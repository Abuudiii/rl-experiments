"""Frame CNN with selective diagonal state-space memory.

The CNN sees the current frame (``n_frames=1``) or the previous frames and the current one of
the same episode packed oldest first along channels (``3 * n_frames`` channels; the episode's
first frame repeats at its start). Training composes affine recurrent updates with an
associative scan. Flying uses the same discretization in one step. Padding is an identity
update; episode resets zero memory. No privileged state or target-gate index is used as a
policy input.
"""
from __future__ import annotations

import dataclasses
import functools
import queue
import threading
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fpv import student as st


@dataclasses.dataclass(frozen=True)
class SsmConfig(st.StudentConfig):
    n_frames: int = 1  # frames packed along channels per step, oldest first: [..., H, W, 3 * n_frames]
    feat: int = 416
    ssm_width: int = 256
    ssm_state: int = 16
    ssm_layers: int = 1
    sequence_length: int = 32
    burn_in: int = 16
    sequence_batch: int = 8
    sequence_microbatch: int = 0  # 0: one compiled gradient; >0: accumulate before one optimizer update


def _compose(left, right):
    """right(left(h)) for diagonal affine updates, in chronological order."""
    a0, b0 = left
    a1, b1 = right
    return a1 * a0, a1 * b0 + b1


class SelectiveSsm(nn.Module):
    width: int
    state_size: int

    @nn.compact
    def __call__(self, x, state, present, sequence: bool):
        z = nn.LayerNorm()(x)
        u, gate = jnp.split(nn.Dense(2 * self.width)(z), 2, -1)
        u = nn.silu(u)
        # Initialize timescales from a few frames through several seconds at 50 Hz.
        dt = nn.softplus(nn.Dense(self.width, bias_init=nn.initializers.constant(-4.0))(z))
        log_a = self.param("log_a", lambda key, shape: jnp.broadcast_to(
            jnp.log(jnp.linspace(0.1, 4.0, self.state_size)), shape), (self.width, self.state_size))
        b, c = jnp.split(nn.Dense(2 * self.state_size)(z), 2, -1)
        decay = jnp.exp(-dt[..., :, None] * jnp.exp(log_a))
        drive = (1.0 - decay) * u[..., :, None] * b[..., None, :]
        decay = jnp.where(present[..., None, None], decay, 1.0)
        drive = jnp.where(present[..., None, None], drive, 0.0)
        if sequence:
            aa, bb = jax.lax.associative_scan(_compose, (decay, drive), axis=1)
            memory = aa * state[:, None] + bb
            final = memory[:, -1]
        else:
            memory = decay * state + drive
            final = memory
        skip = self.param("skip", nn.initializers.ones, (self.width,))
        y = jnp.sum(memory * c[..., None, :], axis=-1) / self.state_size**0.5 + skip * u
        y = nn.Dense(self.width)(y * nn.silu(gate))
        return x + y, final


class StudentSsmNet(nn.Module):
    cfg: SsmConfig

    @nn.compact
    def __call__(self, frames, prev_action, state, present, sequence: bool = False):
        """frames float [B, (T,) H, W, 3 * n_frames] in [0, 1]; state [B, layers, width, state]."""
        c = self.cfg
        leading = frames.shape[:-3]
        h, w, ch = frames.shape[-3:]
        if ch != 3 * c.n_frames:
            raise ValueError(f"expected {3 * c.n_frames} packed frame channels, got {ch}")
        x = frames.reshape(-1, h, w, ch) - 0.5
        yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, h), jnp.linspace(-1, 1, w), indexing="ij")
        x = jnp.concatenate([x, jnp.broadcast_to(jnp.stack([xx, yy], -1), (*x.shape[:-1], 2))], -1)
        for i, ch in enumerate(c.channels):
            x = nn.Conv(ch, (5, 5) if i == 0 else (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.relu(nn.GroupNorm(num_groups=8)(x))
        x = nn.relu(nn.Dense(c.feat)(x.reshape(x.shape[0], -1)))
        x = nn.Dense(c.ssm_width)(x).reshape(*leading, c.ssm_width)
        states = []
        for i in range(c.ssm_layers):
            x, final = SelectiveSsm(c.ssm_width, c.ssm_state, name=f"memory_{i}")(
                x, state[:, i], present, sequence)
            states.append(final)
        if c.use_prev_action:
            x = jnp.concatenate([x, prev_action], -1)
        for hdim in c.hidden:
            x = nn.relu(nn.Dense(hdim)(x))
        action = jnp.tanh(nn.Dense(4, kernel_init=nn.initializers.variance_scaling(
            0.01, "fan_in", "uniform"))(x))
        return action, nn.Dense(st.AUX_DIM)(x), jnp.stack(states, axis=1)


def empty_state(cfg, batch):
    return jnp.zeros((batch, cfg.ssm_layers, cfg.ssm_width, cfg.ssm_state), jnp.float32)


def init_params(cfg, seed=0):
    net = StudentSsmNet(cfg)
    params = net.init(jax.random.key(seed), jnp.zeros((1, cfg.height, cfg.width, 3 * cfg.n_frames)),
                      jnp.zeros((1, 4)), empty_state(cfg, 1), jnp.ones(1, bool))
    return net, params


@functools.lru_cache(maxsize=8)
def make_train_step(cfg, total_steps, warmup=200, lr=None):
    net = StudentSsmNet(cfg)
    rate = lr or cfg.lr
    schedule = optax.warmup_cosine_decay_schedule(0.0, rate, warmup, max(total_steps, warmup + 1), rate * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=cfg.weight_decay))

    def loss_fn(params, frames, prev_action, label, aux, present, mask, key):
        x = frames.astype(jnp.float32) / 255.0
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        b = x.shape[0]
        if cfg.augment:  # consistent photometric transformation throughout a chunk
            blur = nn.avg_pool(x.reshape(-1, *x.shape[2:]), (3, 3), strides=(1, 1), padding="SAME").reshape(x.shape)
            x = jnp.where(jax.random.uniform(k7, (b, 1, 1, 1, 1)) < 0.3, blur, x)
            gain = jax.random.uniform(k1, (b, 1, 1, 1, 1), minval=0.7, maxval=1.3)
            tint = jnp.tile(jax.random.uniform(k2, (b, 1, 1, 1, 3), minval=0.85, maxval=1.15), cfg.n_frames)
            off = jax.random.uniform(k3, (b, 1, 1, 1, 1), minval=-0.1, maxval=0.1)
            sig = jax.random.uniform(k4, (b, 1, 1, 1, 1), minval=0.0, maxval=0.03)
            contrast = jax.random.uniform(k8, (b, 1, 1, 1, 1), minval=0.7, maxval=1.3)
            mean = x.mean(axis=(1, 2, 3, 4), keepdims=True)
            x = jnp.clip(((x - mean) * contrast + mean) * gain * tint + off
                         + sig * jax.random.normal(k5, x.shape), 0, 1)
        ka, kb = jax.random.split(k6)
        pa = prev_action + cfg.prev_action_noise * jax.random.normal(ka, prev_action.shape)
        pa *= jax.random.uniform(kb, (*pa.shape[:-1], 1)) >= cfg.prev_action_drop
        # Burn-in produces the entering state, without supervising its predictions or
        # backpropagating through the prefix. The suffix uses a parallel scan.
        state = empty_state(cfg, b)
        burn = cfg.burn_in
        if burn:
            _, _, state = net.apply(params, x[:, :burn], pa[:, :burn], state, present[:, :burn], sequence=True)
            state = jax.lax.stop_gradient(state)
        act, aux_pred, _ = net.apply(params, x[:, burn:], pa[:, burn:], state, present[:, burn:], sequence=True)
        weights = mask[:, burn:].astype(jnp.float32)
        den = jnp.maximum(weights.sum(), 1.0)
        l_act = (optax.huber_loss(act, label[:, burn:], delta=0.1).mean(-1) * weights).sum() / den * 10.0
        l_aux = (jnp.square(aux_pred - aux[:, burn:]).mean(-1) * weights).sum() / den
        mae = (jnp.abs(act - label[:, burn:]).mean(-1) * weights).sum() / den
        return l_act + cfg.aux_weight * l_aux, dict(act=l_act, aux=l_aux, mae=mae, targets=den)

    @jax.jit
    def step(params, opt_state, frames, pa, label, aux, present, mask, key):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, frames, pa, label, aux, present, mask, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, dict(metrics, loss=loss)

    if not cfg.sequence_microbatch:
        return opt, step

    @jax.jit
    def gradient(params, frames, pa, label, aux, present, mask, key):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, frames, pa, label, aux, present, mask, key)
        return grads, dict(metrics, loss=loss)

    @jax.jit
    def update_accumulated(params, opt_state, results):
        counts = [metrics["targets"] for _, metrics in results]
        total = sum(counts)
        grads = jax.tree_util.tree_map(
            lambda *values: sum(value * count for value, count in zip(values, counts)) / total,
            *(grads for grads, _ in results))
        # Clip and AdamW run ONCE, after target-count-weighted gradient averaging.
        updates, opt_state = opt.update(grads, opt_state, params)
        metrics = {name: sum(m[name] * count for (_, m), count in zip(results, counts)) / total
                   for name in ("act", "aux", "mae", "loss")}
        return optax.apply_updates(params, updates), opt_state, dict(metrics, targets=total)

    def accumulated_step(params, opt_state, frames, pa, label, aux, present, mask, key):
        micro = cfg.sequence_microbatch
        if frames.shape[0] % micro:
            raise ValueError("sequence batch must be divisible by sequence microbatch")
        keys = jax.random.split(key, frames.shape[0] // micro)
        batch = (frames, pa, label, aux, present, mask)
        results = [gradient(params, *(value[i : i + micro] for value in batch), k)
                   for i, k in zip(range(0, frames.shape[0], micro), keys)]
        return update_accumulated(params, opt_state, results)

    return opt, accumulated_step


def train(cfg, params, data, steps, seed=0, lr=None, log=print, log_every=500,
          recent_weight=1.0, extra_frac=0.25, workers=2):
    opt, step = make_train_step(cfg, steps, lr=lr)
    opt_state = opt.init(params)
    own = [b for b in data.blocks if not b.extra]
    last = max(b.round for b in own)
    weights = np.array([0.0 if b.extra else len(b) * (recent_weight if b.round == last else 1.0) for b in data.blocks])
    n_extra = np.array([len(b) if b.extra else 0.0 for b in data.blocks])
    if n_extra.sum() > 0:
        weights = (1 - extra_frac) * weights / weights.sum() + extra_frac * n_extra / n_extra.sum()
    q = queue.Queue(maxsize=workers)
    stop = threading.Event()

    def work(i):
        rng = np.random.default_rng([seed, i])
        while not stop.is_set():
            item = data.sample_sequences(rng, cfg.sequence_batch, cfg.sequence_length, cfg.burn_in, weights,
                                         n_frames=cfg.n_frames)
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    pass

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(workers)]
    for thread in threads:
        thread.start()
    key = jax.random.key(seed)
    t0, acc, hist = time.time(), [], []
    try:
        for i in range(steps):
            batch = q.get()
            key, k = jax.random.split(key)
            params, opt_state, metrics = step(params, opt_state, *batch, k)
            acc.append(metrics)
            if (i + 1) % log_every == 0 or i + 1 == steps:
                mm = {k: float(np.mean([np.asarray(a[k]) for a in acc])) for k in acc[0]}
                acc = []
                hist.append(dict(step=i + 1, wall_s=time.time() - t0, **mm))
                log(f"  train {i + 1}/{steps} loss {mm['loss']:.4f} act {mm['act']:.4f} aux {mm['aux']:.4f} "
                    f"mae {mm['mae']:.4f} targets {mm['targets']:.1f} ({(i + 1) / (time.time() - t0):.1f} it/s)")
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    return params, hist


def _make_act(cfg, n):
    """One jitted closed-loop step for n episodes. ``hist`` holds the previous n_frames - 1 RGB
    frames packed along channels (uint8, on device), or None for current-frame students."""
    net = StudentSsmNet(cfg)

    @jax.jit
    def act(params, state, hist, frame, new, prev_action):
        state = jnp.where(new[:, None, None, None], 0.0, state)
        if hist is None:
            packed = frame
        else:
            first = jnp.tile(frame, cfg.n_frames - 1)
            packed = jnp.concatenate([jnp.where(new[:, None, None, None], first, hist), frame], -1)
            hist = packed[..., 3:]
        action, aux, state = net.apply(params, packed.astype(jnp.float32) / 255.0,
                                      prev_action, state, jnp.ones(n, bool))
        return state, hist, action, aux

    return act


def _empty_hist(cfg, n):
    if cfg.n_frames == 1:
        return None
    return jnp.zeros((n, cfg.height, cfg.width, 3 * (cfg.n_frames - 1)), jnp.uint8)


class StudentRunner:
    def __init__(self, cfg, params, n):
        self.cfg, self.params, self.n = cfg, params, n
        self._act = _make_act(cfg, n)
        self._state = empty_state(cfg, n)
        self._hist = _empty_hist(cfg, n)

    def act(self, frames, prev_action, new):
        """frames uint8 [n, H, W, 3] of the current states, prev_action [n, 4], new [n] bool = first
        frame of an episode -> actions [n, 4] float32 (numpy)."""
        self._state, self._hist, action, self.last_aux = self._act(
            self.params, self._state, self._hist, jnp.asarray(frames), jnp.asarray(new),
            jnp.asarray(prev_action, jnp.float32))
        return np.asarray(action)


def inference_speed(cfg, params, batches=(1, 64), reps=200):
    """Times the runner's single-step update (reset mask, frame packing, recurrence), carrying
    state and frame history; params are a jit argument so weights are not constant-folded."""
    out = {}
    for batch in batches:
        f = _make_act(cfg, batch)
        state, hist = empty_state(cfg, batch), _empty_hist(cfg, batch)
        frame = jnp.zeros((batch, cfg.height, cfg.width, 3), jnp.uint8)
        new = jnp.zeros(batch, bool)
        pa = jnp.zeros((batch, 4))
        state, hist, action, _ = f(params, state, hist, frame, new, pa)
        action.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(reps):
            state, hist, action, _ = f(params, state, hist, frame, new, pa)
            action.block_until_ready()
        dt = (time.perf_counter() - t0) / reps
        out[f"batch_{batch}"] = dict(ms_per_call=dt * 1e3, calls_per_s=1 / dt)
    return out
