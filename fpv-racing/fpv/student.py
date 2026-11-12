"""Camera-only student policy: network, training step, checkpoint IO, closed-loop runner.

Input at inference: the last ``n_frames`` FPV frames (uint8 RGB, 50 Hz, oldest first; at the start
of an episode the first frame is repeated) and the previous applied action. Output: 4 action values
in [-1, 1] (`fpv.env.scale_action`). No state input.

Auxiliary head (training only): regresses privileged quantities taken from the teacher observation
of the current state (`AUX_IDX`): body-frame velocity, the gravity row of the rotation matrix, body
rates and the opening corners of the next two gates in the body frame.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

# teacher_obs entries used as auxiliary targets: velocity (0:3), R[2, :] (9:12), body rates (12:15),
# corners of the next gate (15:27) and of the one after (27:39), all in the body frame
AUX_IDX = np.r_[0:3, 9:39]
AUX_DIM = len(AUX_IDX)


@dataclasses.dataclass(frozen=True)
class StudentConfig:
    width: int = 160
    height: int = 90
    n_frames: int = 3
    use_prev_action: bool = True
    channels: tuple = (32, 64, 96, 128, 192)
    feat: int = 512
    hidden: tuple = (256, 256)
    aux_weight: float = 1.0  # 0 disables the auxiliary head's loss
    # training
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch: int = 256
    augment: bool = True
    prev_action_noise: float = 0.05  # std of the noise added to the previous-action input in training
    prev_action_drop: float = 0.0  # probability of zeroing the previous-action input in training


class StudentNet(nn.Module):
    cfg: StudentConfig

    @nn.compact
    def __call__(self, frames, prev_action):
        """frames float [B, K, H, W, 3] in [0, 1]; prev_action [B, 4] -> (action [B, 4], aux [B, AUX_DIM])."""
        c = self.cfg
        b, k, h, w, _ = frames.shape
        x = jnp.transpose(frames - 0.5, (0, 2, 3, 1, 4)).reshape(b, h, w, k * 3)
        yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, h), jnp.linspace(-1, 1, w), indexing="ij")
        x = jnp.concatenate([x, jnp.broadcast_to(jnp.stack([xx, yy], -1), (b, h, w, 2))], -1)
        for i, ch in enumerate(c.channels):
            x = nn.Conv(ch, (5, 5) if i == 0 else (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.relu(nn.GroupNorm(num_groups=8)(x))
        x = nn.relu(nn.Dense(c.feat)(x.reshape(b, -1)))
        if c.use_prev_action:
            x = jnp.concatenate([x, prev_action], -1)
        for hdim in c.hidden:
            x = nn.relu(nn.Dense(hdim)(x))
        action = jnp.tanh(nn.Dense(4, kernel_init=nn.initializers.variance_scaling(0.01, "fan_in", "uniform"))(x))
        return action, nn.Dense(AUX_DIM)(x)


def init_params(cfg: StudentConfig, seed: int = 0):
    net = StudentNet(cfg)
    params = net.init(jax.random.key(seed), jnp.zeros((1, cfg.n_frames, cfg.height, cfg.width, 3)), jnp.zeros((1, 4)))
    return net, params


def n_params(params) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))


def make_train_step(cfg: StudentConfig, total_steps: int, warmup: int = 200, lr: float | None = None):
    net = StudentNet(cfg)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr or cfg.lr, warmup, max(total_steps, warmup + 1), (lr or cfg.lr) * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=cfg.weight_decay))

    def loss_fn(params, frames, prev_action, label, aux, key):
        x = frames.astype(jnp.float32) / 255.0
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        b = x.shape[0]
        if cfg.augment:  # the same photometric change for all frames of a sample
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
        ka, kb = jax.random.split(k6)
        pa = prev_action + cfg.prev_action_noise * jax.random.normal(ka, prev_action.shape)
        pa = pa * (jax.random.uniform(kb, (b, 1)) >= cfg.prev_action_drop)
        act, aux_pred = net.apply(params, x, pa)
        l_act = optax.huber_loss(act, label, delta=0.1).mean() * 10.0
        l_aux = jnp.square(aux_pred - aux).mean()
        mae = jnp.abs(act - label).mean()
        return l_act + cfg.aux_weight * l_aux, dict(act=l_act, aux=l_aux, mae=mae)

    @jax.jit
    def step(params, opt_state, frames, prev_action, label, aux, key):
        (loss, m), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, frames, prev_action, label, aux, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, dict(m, loss=loss)

    @jax.jit
    def evaluate(params, frames, prev_action, label, aux):
        act, aux_pred = net.apply(params, frames.astype(jnp.float32) / 255.0, prev_action)
        return dict(mae=jnp.abs(act - label).mean(0), aux=jnp.square(aux_pred - aux).mean())

    return opt, step, evaluate


def save_student(path, params, cfg: StudentConfig, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = dict(params=jax.tree_util.tree_map(np.asarray, params), config=dataclasses.asdict(cfg), **(extra or {}))
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(ckpt))
    tmp.rename(path)


def load_student(path):
    """-> (cfg, params, checkpoint dict)."""
    ckpt = pickle.loads(Path(path).read_bytes())
    c = dict(ckpt["config"])
    for k in ("channels", "hidden"):
        c[k] = tuple(c[k])
    cfg = StudentConfig(**c)
    return cfg, jax.tree_util.tree_map(jnp.asarray, ckpt["params"]), ckpt


class StudentRunner:
    """Closed-loop inference for n parallel episodes; keeps the frame history on the device."""

    def __init__(self, cfg: StudentConfig, params, n: int):
        self.cfg, self.params, self.n = cfg, params, n
        net = StudentNet(cfg)

        @jax.jit
        def act(params, hist, frame, new, prev_action):
            rolled = jnp.concatenate([hist[:, 1:], frame[:, None]], axis=1)
            first = jnp.broadcast_to(frame[:, None], hist.shape)
            hist = jnp.where(new[:, None, None, None, None], first, rolled)
            a, aux = net.apply(params, hist.astype(jnp.float32) / 255.0, prev_action)
            return hist, a, aux

        self._act = act
        self._hist = jnp.zeros((n, cfg.n_frames, cfg.height, cfg.width, 3), jnp.uint8)

    def act(self, frames, prev_action, new):
        """frames uint8 [n, H, W, 3] of the current states, prev_action [n, 4], new [n] bool = first
        frame of an episode -> actions [n, 4] float32 (numpy)."""
        self._hist, a, self.last_aux = self._act(self.params, self._hist, jnp.asarray(frames), jnp.asarray(new),
                                                 jnp.asarray(prev_action, jnp.float32))  # fmt: skip
        return np.asarray(a)
