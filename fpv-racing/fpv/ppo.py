"""PPO in JAX (flax / optax) with jitted rollout + update, for `fpv.env.RaceEnv`."""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fpv.env import ACT_DIM, OBS_DIM, EnvConfig, RaceEnv


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    total_steps: float = 5e8  # environment (policy) steps
    n_steps: int = 32  # rollout length per iteration
    n_epochs: int = 4
    n_minibatches: int = 8
    lr: float = 3e-4
    lr_final_frac: float = 0.1  # linear decay of the learning rate to this fraction
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    hidden: tuple = (256, 256, 256)
    log_std_init: float = -0.7
    log_std_min: float = -3.0
    log_std_max: float = 0.5


# region network


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs. Gaussian policy: mean = tanh(MLP), state-independent std."""

    hidden: tuple
    log_std_init: float = -0.7

    def setup(self):
        init = nn.initializers.orthogonal
        self.actor = [nn.Dense(h, kernel_init=init(np.sqrt(2))) for h in self.hidden]
        self.actor_out = nn.Dense(ACT_DIM, kernel_init=init(0.01))
        self.critic = [nn.Dense(h, kernel_init=init(np.sqrt(2))) for h in self.hidden]
        self.critic_out = nn.Dense(1, kernel_init=init(1.0))
        self.log_std = self.param("log_std", nn.initializers.constant(self.log_std_init), (ACT_DIM,))

    def mean(self, x):
        for layer in self.actor:
            x = nn.elu(layer(x))
        return jnp.tanh(self.actor_out(x))

    def value(self, x):
        for layer in self.critic:
            x = nn.elu(layer(x))
        return self.critic_out(x)[..., 0]

    def __call__(self, x):
        return self.mean(x), self.log_std, self.value(x)


class ObsNorm(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def norm_init() -> ObsNorm:
    return ObsNorm(jnp.zeros(OBS_DIM), jnp.ones(OBS_DIM), jnp.asarray(1e-4))


def norm_update(norm: ObsNorm, x) -> ObsNorm:
    x = x.reshape(-1, x.shape[-1])
    b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
    tot = norm.count + b_n
    delta = b_mean - norm.mean
    mean = norm.mean + delta * b_n / tot
    var = (norm.var * norm.count + b_var * b_n + delta**2 * norm.count * b_n / tot) / tot
    return ObsNorm(mean, var, tot)


def norm_apply(norm: ObsNorm, x):
    return jnp.clip((x - norm.mean) / jnp.sqrt(norm.var + 1e-8), -10.0, 10.0)


def policy_action(net: ActorCritic, params, norm: ObsNorm, obs):
    """Deterministic action (the mean), already in [-1, 1]."""
    return net.apply(params, norm_apply(norm, obs), method=net.mean)


# region training


class Transition(NamedTuple):
    nobs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def _logp(mean, log_std, action):
    z = (action - mean) / jnp.exp(log_std)
    return (-0.5 * z**2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


def make_trainer(env: RaceEnv, cfg: PPOConfig):
    """Returns (net, init(key) -> runner, iterate(runner) -> (runner, metrics)), both jitted."""
    n_envs = env.cfg.n_envs
    batch = n_envs * cfg.n_steps
    n_iters = max(1, int(cfg.total_steps // batch))
    net = ActorCritic(tuple(cfg.hidden), cfg.log_std_init)
    n_updates = n_iters * cfg.n_epochs * cfg.n_minibatches
    schedule = optax.linear_schedule(cfg.lr, cfg.lr * cfg.lr_final_frac, n_updates)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule, eps=1e-5))

    def clamp_std(params):
        p = dict(params["params"])
        p["log_std"] = jnp.clip(p["log_std"], cfg.log_std_min, cfg.log_std_max)
        return {"params": p}

    def init(key):
        k_net, k_env, key = jax.random.split(key, 3)
        params = net.init(k_net, jnp.zeros((1, OBS_DIM)))
        env_state, obs = env.reset(k_env, random_t=True)
        return dict(params=params, opt=tx.init(params), norm=norm_init(), env=env_state, obs=obs, key=key)

    def rollout_step(carry, _):
        params, norm, env_state, obs, key = carry
        key, k = jax.random.split(key)
        nobs = norm_apply(norm, obs)
        mean, log_std, value = net.apply(params, nobs)
        action = mean + jnp.exp(log_std) * jax.random.normal(k, mean.shape)
        logp = _logp(mean, log_std, action)
        env_state, next_obs, reward, done, info = env.step(env_state, action)
        # Time limits are not failures: bootstrap from the value of the last observation
        v_final = net.apply(params, norm_apply(norm, info["final_obs"]), method=net.value)
        reward = reward + cfg.gamma * v_final * info["truncated"]
        d = done.astype(jnp.float32)
        stats = dict(
            episodes=d.sum(),
            ep_return=(info["ep_return"] * d).sum(),
            ep_len=(info["ep_len"] * d).sum(),
            ep_gates=(info["ep_gates"] * d).sum(),
            crash_gate=info["crash_gate"].sum(),
            ground=(info["ground"] & ~info["crash_gate"]).sum(),
            oob=(info["oob"] & ~info["crash_gate"]).sum(),
            truncated=info["truncated"].sum(),
            gates=info["passed"].sum(),
            in_fov=info["in_fov"].mean(),
            speed=info["speed"].mean(),
            reward=reward.mean(),
        )
        return (params, norm, env_state, next_obs, key), (Transition(nobs, action, logp, value, reward, d), obs, stats)

    def loss_fn(params, mb: Transition, adv, ret):
        mean, log_std, value = net.apply(params, mb.nobs)
        logp = _logp(mean, log_std, mb.action)
        ratio = jnp.exp(logp - mb.logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        vf = 0.5 * ((value - ret) ** 2).mean()
        entropy = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        loss = pg + cfg.vf_coef * vf - cfg.ent_coef * entropy
        aux = dict(
            pg_loss=pg, vf_loss=vf, entropy=entropy, kl=(mb.logp - logp).mean(),
            clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean(),
        )  # fmt: skip
        return loss, aux

    @jax.jit
    def iterate(runner):
        params, opt, norm = runner["params"], runner["opt"], runner["norm"]
        carry = (params, norm, runner["env"], runner["obs"], runner["key"])
        (_, _, env_state, obs, key), (traj, raw_obs, stats) = jax.lax.scan(rollout_step, carry, None, cfg.n_steps)
        last_value = net.apply(params, norm_apply(norm, obs), method=net.value)

        def gae_step(c, x):
            gae, next_value = c
            tr = x
            delta = tr.reward + cfg.gamma * next_value * (1 - tr.done) - tr.value
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - tr.done) * gae
            return (gae, tr.value), gae

        _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        ret = adv + traj.value
        flat = jax.tree.map(lambda x: x.reshape(batch, *x.shape[2:]), (traj, adv, ret))

        def epoch(c, k):
            params, opt = c
            perm = jax.random.permutation(k, batch)
            mbs = jax.tree.map(lambda x: x[perm].reshape(cfg.n_minibatches, -1, *x.shape[1:]), flat)

            def minibatch(c, mb):
                params, opt = c
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *mb)
                updates, opt = tx.update(grads, opt, params)
                return (clamp_std(optax.apply_updates(params, updates)), opt), aux

            return jax.lax.scan(minibatch, (params, opt), mbs)

        key, k = jax.random.split(key)
        (params, opt), aux = jax.lax.scan(epoch, (params, opt), jax.random.split(k, cfg.n_epochs))
        norm = norm_update(norm, raw_obs)
        ep = jnp.maximum(stats["episodes"].sum(), 1.0)
        crashes = stats["crash_gate"].sum() + stats["ground"].sum() + stats["oob"].sum()
        metrics = dict(
            episodes=stats["episodes"].sum(),
            ep_return=stats["ep_return"].sum() / ep,
            ep_len=stats["ep_len"].sum() / ep,
            ep_gates=stats["ep_gates"].sum() / ep,
            crash_rate=crashes / ep,
            crash_gate=stats["crash_gate"].sum() / ep,
            crash_ground=stats["ground"].sum() / ep,
            crash_oob=stats["oob"].sum() / ep,
            gates_per_s=stats["gates"].sum() / (batch / env.cfg.policy_hz),
            in_fov=stats["in_fov"].mean(),
            speed=stats["speed"].mean(),
            reward=stats["reward"].mean(),
            value=traj.value.mean(),
            explained_var=1 - (ret - traj.value).var() / (ret.var() + 1e-8),
            std=jnp.exp(params["params"]["log_std"]).mean(),
            **{k: v.mean() for k, v in aux.items()},
        )
        runner = dict(params=params, opt=opt, norm=norm, env=env_state, obs=obs, key=key)
        return runner, metrics

    return net, jax.jit(init), iterate, n_iters


# region checkpoints


def config_dict(env_cfg: EnvConfig, ppo_cfg: PPOConfig) -> dict:
    return {"env": dataclasses.asdict(env_cfg), "ppo": dataclasses.asdict(ppo_cfg)}


def save_checkpoint(path: Path, params, norm: ObsNorm, env_cfg: EnvConfig, ppo_cfg: PPOConfig, extra: dict):
    """One pickle of plain numpy: {"params", "obs_norm": {mean, var, count}, "config", **extra}."""
    ckpt = dict(
        params=jax.tree.map(np.asarray, params),
        obs_norm={k: np.asarray(v) for k, v in norm._asdict().items()},
        config=config_dict(env_cfg, ppo_cfg),
        **extra,
    )
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(ckpt))
    tmp.rename(path)


def load_checkpoint(path: Path):
    """Returns (net, params, norm, env_cfg, ppo_cfg, ckpt_dict)."""
    ckpt = pickle.loads(Path(path).read_bytes())
    env_cfg = EnvConfig(**ckpt["config"]["env"])
    ppo = dict(ckpt["config"]["ppo"])
    ppo["hidden"] = tuple(ppo["hidden"])
    ppo_cfg = PPOConfig(**ppo)
    net = ActorCritic(ppo_cfg.hidden, ppo_cfg.log_std_init)
    params = jax.tree.map(jnp.asarray, ckpt["params"])
    norm = ObsNorm(**{k: jnp.asarray(v) for k, v in ckpt["obs_norm"].items()})
    return net, params, norm, env_cfg, ppo_cfg, ckpt


def write_config(path: Path, env_cfg: EnvConfig, ppo_cfg: PPOConfig, extra: dict | None = None):
    Path(path).write_text(json.dumps(config_dict(env_cfg, ppo_cfg) | (extra or {}), indent=2))
