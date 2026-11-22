"""5-frame CNN student with a residual recurrent memory on top of its feature vector.

    frames (last n_frames, oldest first) -> StudentNet's CNN + Dense(feat) + ReLU   = f      (unchanged)
    m = LayerNorm(f) -> Dense(mem_width) -> SelectiveSsm (fpv.student_ssm)          (carried state)
    f' = f + Dense(feat, zero-initialised)(m)
    f' (+ previous action) -> StudentNet's hidden layers, action and auxiliary heads    (unchanged)

All layers outside the ``memory`` submodule carry StudentNet's parameter names, so a plain checkpoint's
parameters drop in (`from_plain`); with the zero output projection the memory student then returns
exactly the plain student's actions. Optional ``gate_head``: logits over the 6 gate indices predicting
the target (next) gate from ``m`` (training signal only, weight ``gate_weight``).

The state is carried while flying and reset at every episode start. Training uses windows of
``burn_in + sequence_length`` consecutive steps of one episode (`sample_windows`); the burn-in builds
the entering state without loss and without gradient; padding before the episode start is an
identity update (``present=False``).
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fpv import student as st
from fpv.student_ssm import SelectiveSsm

N_GATES = 6


@dataclasses.dataclass(frozen=True)
class MemConfig(st.StudentConfig):
    mem_width: int = 128
    mem_state: int = 16
    gate_head: bool = False
    gate_weight: float = 0.1
    sequence_length: int = 16
    burn_in: int = 16
    sequence_batch: int = 64
    sequence_microbatch: int = 16  # sequences per compiled gradient; gradients averaged before one update


class MemoryCore(nn.Module):
    feat: int
    width: int
    state_size: int

    @nn.compact
    def __call__(self, f, state, present, sequence: bool):
        z = nn.Dense(self.width, name="in_proj")(nn.LayerNorm(name="in_norm")(f))
        m, final = SelectiveSsm(self.width, self.state_size, name="ssm")(z, state, present, sequence)
        delta = nn.Dense(self.feat, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
                         name="out_proj")(m)
        return delta, m, final


class MemNet(nn.Module):
    cfg: MemConfig

    @nn.compact
    def __call__(self, frames, prev_action, state, present, sequence: bool = False):
        """frames float [B, (T,) K, H, W, 3] in [0, 1]; prev_action [B, (T,) 4]; state [B, mem_width, mem_state];
        present bool [B, (T,)] -> (action, aux, gate_logits or None, final state)."""
        c = self.cfg
        lead = frames.shape[:-4]
        k, h, w, _ = frames.shape[-4:]
        b = int(np.prod(lead))
        x = jnp.transpose(frames.reshape(b, k, h, w, 3) - 0.5, (0, 2, 3, 1, 4)).reshape(b, h, w, k * 3)
        yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, h), jnp.linspace(-1, 1, w), indexing="ij")
        x = jnp.concatenate([x, jnp.broadcast_to(jnp.stack([xx, yy], -1), (b, h, w, 2))], -1)
        for i, ch in enumerate(c.channels):
            x = nn.Conv(ch, (5, 5) if i == 0 else (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.relu(nn.GroupNorm(num_groups=8)(x))
        f = nn.relu(nn.Dense(c.feat)(x.reshape(b, -1))).reshape(*lead, c.feat)
        delta, m, final = MemoryCore(c.feat, c.mem_width, c.mem_state, name="memory")(f, state, present, sequence)
        x = f + delta
        if c.use_prev_action:
            x = jnp.concatenate([x, prev_action], -1)
        for hdim in c.hidden:
            x = nn.relu(nn.Dense(hdim)(x))
        action = jnp.tanh(nn.Dense(4, kernel_init=nn.initializers.variance_scaling(0.01, "fan_in", "uniform"))(x))
        aux = nn.Dense(st.AUX_DIM)(x)
        gate = nn.Dense(N_GATES, name="gate_head")(m) if c.gate_head else None
        return action, aux, gate, final


def empty_state(cfg, batch):
    return jnp.zeros((batch, cfg.mem_width, cfg.mem_state), jnp.float32)


def init_params(cfg, seed=0):
    net = MemNet(cfg)
    params = net.init(jax.random.key(seed), jnp.zeros((1, cfg.n_frames, cfg.height, cfg.width, 3)),
                      jnp.zeros((1, 4)), empty_state(cfg, 1), jnp.ones(1, bool))
    return net, params


def from_plain(plain_cfg, plain_params, seed=0, **mem_kw):
    """Memory student whose non-memory weights are a plain checkpoint's; same actions at init."""
    cfg = MemConfig(**dataclasses.asdict(plain_cfg), **mem_kw)
    _, params = init_params(cfg, seed)
    p = dict(params["params"])
    plain = plain_params["params"]
    extra = set(plain) - set(p)
    if extra:
        raise ValueError(f"plain parameters without a place in the memory student: {sorted(extra)}")
    for name, v in plain.items():
        got = jax.tree_util.tree_map(jnp.shape, p[name])
        if got != jax.tree_util.tree_map(jnp.shape, v):
            raise ValueError(f"{name}: shape mismatch {got}")
        p[name] = v
    return cfg, {"params": p}


# region data
def attach_targets(data):
    """Give every block of a `fpv.dagger_ssm_data.DiskDataset` its flat ``target`` (next gate index)."""
    from pathlib import Path

    for b in data.blocks:
        if hasattr(b, "target"):
            continue
        parts = []
        for m in b.frames.maps:
            with np.load(Path(m.filename)) as z:
                parts.append(z["target"])
        t = np.concatenate(parts, 0)
        b.target = np.ascontiguousarray(np.swapaxes(t, 0, 1)).reshape(-1).astype(np.int32)


def sample_windows(data, rng, batch, length, burn_in, n_frames, weights=None):
    """Windows of T = burn_in + length steps of one episode, with the K-1 frames before them.

    -> frames uint8 [B, T + K - 1, H, W, 3] (frame stack of step t = frames[:, t : t + K]; positions before the
       episode start repeat its first frame, as `Dataset.sample` does), prev_action [B, T, 4], label [B, T, 4],
       aux [B, T, AUX_DIM], present bool [B, T], loss mask bool [B, T], target int32 [B, T]."""
    total = burn_in + length
    back = np.arange(total + n_frames - 2, -1, -1)[None]  # oldest first, incl. K-1 history frames
    counts = data._counts(rng, batch, weights)
    out = [[], [], [], [], [], [], []]
    for b, c in zip(data.blocks, counts):
        if c == 0:
            continue
        i = b.valid_idx[rng.integers(0, len(b.valid_idx), c)]
        pos_ext = i[:, None] - back
        s = b.seq_start[i][:, None]
        frames = b.frames[np.maximum(pos_ext, s)]
        pos = pos_ext[:, n_frames - 1 :]
        present = pos >= s
        idx = np.maximum(pos, s)
        loss = present & b.eligible[idx]
        loss[:, :burn_in] = False
        tgt = b.target[idx] if hasattr(b, "target") else np.zeros(idx.shape, np.int32)
        for o, v in zip(out, (frames, b.prev_action[idx], b.label[idx], b.aux[idx], present, loss, tgt)):
            o.append(v)
    return tuple(np.concatenate(o) for o in out)


def _stack(ext, n_frames):
    """uint8 [B, T + K - 1, H, W, 3] -> [B, T, K, H, W, 3]."""
    t = ext.shape[1] - n_frames + 1
    return ext[:, np.arange(t)[:, None] + np.arange(n_frames)[None]]


# region training
def make_train_step(cfg, total_steps, warmup=200, lr=None):
    net = MemNet(cfg)
    rate = lr or cfg.lr
    schedule = optax.warmup_cosine_decay_schedule(0.0, rate, warmup, max(total_steps, warmup + 1), rate * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=cfg.weight_decay))
    K = cfg.n_frames

    def loss_fn(params, ext, prev_action, label, aux, present, mask, target, key):
        x = ext.astype(jnp.float32) / 255.0  # [b, T+K-1, H, W, 3]
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        b = x.shape[0]
        if cfg.augment:  # the same photometric change for all frames of a window (as StudentNet per sample)
            blur = nn.avg_pool(x.reshape(-1, *x.shape[2:]), (3, 3), strides=(1, 1), padding="SAME").reshape(x.shape)
            x = jnp.where(jax.random.uniform(k7, (b, 1, 1, 1, 1)) < 0.3, blur, x)
            gain = jax.random.uniform(k1, (b, 1, 1, 1, 1), minval=0.7, maxval=1.3)
            tint = jax.random.uniform(k2, (b, 1, 1, 1, 3), minval=0.85, maxval=1.15)
            off = jax.random.uniform(k3, (b, 1, 1, 1, 1), minval=-0.1, maxval=0.1)
            sig = jax.random.uniform(k4, (b, 1, 1, 1, 1), minval=0.0, maxval=0.03)
            con = jax.random.uniform(k8, (b, 1, 1, 1, 1), minval=0.7, maxval=1.3)
            mean = x.mean(axis=(1, 2, 3, 4), keepdims=True)
            x = (x - mean) * con + mean
            x = jnp.clip(x * gain * tint + off + sig * jax.random.normal(k5, x.shape), 0, 1)
        x = _stack(x, K)  # [b, T, K, H, W, 3]
        ka, kb = jax.random.split(k6)
        pa = prev_action + cfg.prev_action_noise * jax.random.normal(ka, prev_action.shape)
        pa = pa * (jax.random.uniform(kb, (*pa.shape[:-1], 1)) >= cfg.prev_action_drop)
        state = empty_state(cfg, b)
        burn = cfg.burn_in
        if burn:
            *_, state = net.apply(params, x[:, :burn], pa[:, :burn], state, present[:, :burn], sequence=True)
            state = jax.lax.stop_gradient(state)
        act, aux_pred, gate, _ = net.apply(params, x[:, burn:], pa[:, burn:], state, present[:, burn:], sequence=True)
        wts = mask[:, burn:].astype(jnp.float32)
        den = jnp.maximum(wts.sum(), 1.0)
        l_act = (optax.huber_loss(act, label[:, burn:], delta=0.1).mean(-1) * wts).sum() / den * 10.0
        l_aux = (jnp.square(aux_pred - aux[:, burn:]).mean(-1) * wts).sum() / den
        mae = (jnp.abs(act - label[:, burn:]).mean(-1) * wts).sum() / den
        loss = l_act + cfg.aux_weight * l_aux
        m = dict(act=l_act, aux=l_aux, mae=mae, targets=den)
        if cfg.gate_head:
            ce = optax.softmax_cross_entropy_with_integer_labels(gate, target[:, burn:])
            m["gate_ce"] = (ce * wts).sum() / den
            m["gate_acc"] = ((jnp.argmax(gate, -1) == target[:, burn:]) * wts).sum() / den
            loss = loss + cfg.gate_weight * m["gate_ce"]
        return loss, dict(m, loss=loss)

    @jax.jit
    def gradient(params, *batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *batch)
        return grads, metrics

    @jax.jit
    def update(params, opt_state, results):
        counts = [m["targets"] for _, m in results]
        total = sum(counts)
        grads = jax.tree_util.tree_map(lambda *g: sum(v * c for v, c in zip(g, counts)) / total,
                                       *(g for g, _ in results))  # fmt: skip
        updates, opt_state = opt.update(grads, opt_state, params)  # clip + AdamW once per full batch
        metrics = {k: sum(m[k] * c for (_, m), c in zip(results, counts)) / total
                   for k in results[0][1] if k != "targets"}  # fmt: skip
        return optax.apply_updates(params, updates), opt_state, dict(metrics, targets=total)

    def step(params, opt_state, *batch_and_key):
        *batch, key = batch_and_key
        micro = cfg.sequence_microbatch or batch[0].shape[0]
        n = batch[0].shape[0]
        assert n % micro == 0, "sequence batch must be divisible by the microbatch"
        keys = jax.random.split(key, n // micro)
        results = [gradient(params, *(v[i : i + micro] for v in batch), k) for i, k in zip(range(0, n, micro), keys)]
        return update(params, opt_state, results)

    return opt, step


def train(cfg, params, data, steps, seed=0, lr=None, log=print, log_every=500, recent_weight=1.0,
          extra_frac=0.25, workers=4):  # fmt: skip
    if cfg.gate_head:
        attach_targets(data)
    opt, step = make_train_step(cfg, steps, lr=lr)
    opt_state = opt.init(params)
    own = [b for b in data.blocks if not b.extra]
    last = max(b.round for b in own)
    weights = np.array([0.0 if b.extra else len(b) * (recent_weight if b.round == last else 1.0) for b in data.blocks])
    n_extra = np.array([len(b) if b.extra else 0.0 for b in data.blocks])
    if n_extra.sum() > 0:
        weights = (1 - extra_frac) * weights / weights.sum() + extra_frac * n_extra / n_extra.sum()
    q: queue.Queue = queue.Queue(maxsize=workers)
    stop = threading.Event()

    def work(i):
        rng = np.random.default_rng([seed, i])
        while not stop.is_set():
            item = sample_windows(data, rng, cfg.sequence_batch, cfg.sequence_length, cfg.burn_in, cfg.n_frames, weights)
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    pass

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(workers)]
    for t in threads:
        t.start()
    key = jax.random.key(seed)
    t0, acc, hist = time.time(), [], []
    try:
        for i in range(steps):
            batch = q.get()
            key, k = jax.random.split(key)
            params, opt_state, m = step(params, opt_state, *batch, k)
            acc.append(m)
            if (i + 1) % log_every == 0 or i + 1 == steps:
                mm = {k: float(np.mean([np.asarray(a[k]) for a in acc])) for k in acc[0]}
                acc = []
                hist.append(dict(step=i + 1, wall_s=time.time() - t0, **mm))
                gate = f" gate ce {mm['gate_ce']:.3f} acc {mm['gate_acc']:.3f}" if "gate_ce" in mm else ""
                log(f"  train {i + 1}/{steps} loss {mm['loss']:.4f} act {mm['act']:.4f} aux {mm['aux']:.4f} "
                    f"mae {mm['mae']:.4f} targets {mm['targets']:.1f}{gate} ({(i + 1) / (time.time() - t0):.2f} it/s)")
    finally:
        stop.set()
        for t in threads:
            t.join()
    return params, hist


# region flying
def _make_act(cfg, n):
    net = MemNet(cfg)

    @jax.jit
    def act(params, state, hist, frame, new, prev_action):
        rolled = jnp.concatenate([hist[:, 1:], frame[:, None]], axis=1)
        first = jnp.broadcast_to(frame[:, None], hist.shape)
        hist = jnp.where(new[:, None, None, None, None], first, rolled)
        state = jnp.where(new[:, None, None], 0.0, state)
        a, aux, _, state = net.apply(params, hist.astype(jnp.float32) / 255.0, prev_action, state, jnp.ones(n, bool))
        return state, hist, a, aux

    return act


class StudentRunner:
    """Closed-loop inference for n parallel episodes: frame history and memory state on the device."""

    def __init__(self, cfg, params, n):
        self.cfg, self.params, self.n = cfg, params, n
        self._act = _make_act(cfg, n)
        self._state = empty_state(cfg, n)
        self._hist = jnp.zeros((n, cfg.n_frames, cfg.height, cfg.width, 3), jnp.uint8)

    def act(self, frames, prev_action, new):
        self._state, self._hist, a, self.last_aux = self._act(
            self.params, self._state, self._hist, jnp.asarray(frames), jnp.asarray(new),
            jnp.asarray(prev_action, jnp.float32))  # fmt: skip
        return np.asarray(a)


def inference_speed(cfg, params, batches=(1, 64), reps=200):
    """Wall time of the runner's one-step update (history roll, state reset mask, CNN, memory, heads)."""
    out = {}
    for b in batches:
        f = _make_act(cfg, b)
        state, hist = empty_state(cfg, b), jnp.zeros((b, cfg.n_frames, cfg.height, cfg.width, 3), jnp.uint8)
        frame, new, pa = jnp.zeros((b, cfg.height, cfg.width, 3), jnp.uint8), jnp.zeros(b, bool), jnp.zeros((b, 4))
        state, hist, a, _ = f(params, state, hist, frame, new, pa)
        a.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(reps):
            state, hist, a, _ = f(params, state, hist, frame, new, pa)
            a.block_until_ready()
        dt = (time.perf_counter() - t0) / reps
        out[f"batch_{b}"] = dict(ms_per_call=1e3 * dt, calls_per_s=1 / dt)
    return out
