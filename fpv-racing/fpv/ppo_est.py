"""PPO for the estimated-observation environment (`fpv.env_est.EstEnv`): asymmetric actor-critic, teacher
warm start. Same algorithm and hyper-parameters as `fpv.ppo` (its PPOConfig, GAE, clipping, schedule).

Choices (and why):
  * Actor: the teacher's MLP (3 x 256, ELU, tanh mean, state-independent std) on the 46-number estimated
    observation (`fpv.perception.estimator.observation`); its first 43 inputs have the teacher's layout,
    so the actor is WARM-STARTED from the teacher (first-layer rows of the 3 flag inputs zero): at
    iteration 0 the actor is the teacher flying from estimates. Recurrence is not added: the estimator
    already carries the memory explicitly (remembered gates with their age, filtered velocity), and a
    recurrent policy would forfeit the warm start and cost several times the training steps.
  * Critic: ASYMMETRIC (Pinto et al. 2018, "Asymmetric Actor Critic for Image-Based Robot Learning"): it
    sees the 46 estimated numbers AND the true 43-number teacher observation, so the value is not
    confused by estimation errors it cannot see (e.g. a wrong next-gate belief that will end in a crash).
    Warm-started from the teacher's critic on the true part (estimate rows zero). Training only.
  * Observation normaliser over all 89 numbers (running mean / var as `fpv.ppo`), initialised from the
    teacher's for both copies of the 43 teacher-layout numbers.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fpv.env import ACT_DIM, OBS_DIM, EnvConfig
from fpv.env_est import FULL_OBS_DIM, EstEnv, EstEnvConfig
from fpv.perception.estimator import EST_OBS_DIM
from fpv.ppo import ObsNorm, PPOConfig, Transition, _logp, norm_apply, norm_update


class AsymActorCritic(nn.Module):
    """Actor on x[..., :actor_dim], critic on all of x. Parameter names as `fpv.ppo.ActorCritic`."""

    hidden: tuple
    log_std_init: float = -0.7
    actor_dim: int = EST_OBS_DIM

    def setup(self):
        init = nn.initializers.orthogonal
        self.actor = [nn.Dense(h, kernel_init=init(np.sqrt(2))) for h in self.hidden]
        self.actor_out = nn.Dense(ACT_DIM, kernel_init=init(0.01))
        self.critic = [nn.Dense(h, kernel_init=init(np.sqrt(2))) for h in self.hidden]
        self.critic_out = nn.Dense(1, kernel_init=init(1.0))
        self.log_std = self.param("log_std", nn.initializers.constant(self.log_std_init), (ACT_DIM,))

    def mean(self, x):
        x = x[..., : self.actor_dim]
        for layer in self.actor:
            x = nn.elu(layer(x))
        return jnp.tanh(self.actor_out(x))

    def value(self, x):
        for layer in self.critic:
            x = nn.elu(layer(x))
        return self.critic_out(x)[..., 0]

    def __call__(self, x):
        return self.mean(x), self.log_std, self.value(x)


def norm_init(dim: int = FULL_OBS_DIM) -> ObsNorm:
    return ObsNorm(jnp.zeros(dim), jnp.ones(dim), jnp.asarray(1e-4))


def policy_action(net: AsymActorCritic, params, norm: ObsNorm, est_obs):
    """Deterministic action from the ESTIMATED observation only (N, 46): what flies at deployment."""
    n = norm_apply(ObsNorm(norm.mean[:EST_OBS_DIM], norm.var[:EST_OBS_DIM], norm.count), est_obs)
    return net.apply(params, n, method=net.mean)


def warm_start(params, norm: ObsNorm, t_params, t_norm: ObsNorm, log_std: float | None = None):
    """Copy the teacher (fpv.ppo.ActorCritic on 43 inputs) into the asymmetric net / 89-dim normaliser."""
    p = jax.tree.map(lambda a: a, params)["params"]
    t = t_params["params"]
    p = dict(p)
    for name in t:
        if name == "log_std":
            p[name] = t[name] if log_std is None else jnp.full_like(t[name], log_std)
            continue
        tk = dict(t[name])
        if name == "actor_0":
            k = jnp.zeros_like(p[name]["kernel"]).at[:OBS_DIM].set(tk["kernel"])
            tk["kernel"] = k
        elif name == "critic_0":
            k = jnp.zeros_like(p[name]["kernel"]).at[EST_OBS_DIM:].set(tk["kernel"])
            tk["kernel"] = k
        p[name] = tk
    mean = norm.mean.at[:OBS_DIM].set(t_norm.mean).at[EST_OBS_DIM:].set(t_norm.mean)
    var = norm.var.at[:OBS_DIM].set(t_norm.var).at[EST_OBS_DIM:].set(t_norm.var)
    return {"params": p}, ObsNorm(mean, var, t_norm.count)


def make_trainer(env: EstEnv, cfg: PPOConfig):
    """As `fpv.ppo.make_trainer`, for the asymmetric net. Returns (net, init, iterate, n_iters)."""
    n_envs = env.cfg.n_envs
    batch = n_envs * cfg.n_steps
    n_iters = max(1, int(cfg.total_steps // batch))
    net = AsymActorCritic(tuple(cfg.hidden), cfg.log_std_init)
    n_updates = n_iters * cfg.n_epochs * cfg.n_minibatches
    schedule = optax.linear_schedule(cfg.lr, cfg.lr * cfg.lr_final_frac, n_updates)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule, eps=1e-5))

    def clamp_std(params):
        p = dict(params["params"])
        p["log_std"] = jnp.clip(p["log_std"], cfg.log_std_min, cfg.log_std_max)
        return {"params": p}

    def init(key, params=None, norm=None):
        k_net, k_env, key = jax.random.split(key, 3)
        if params is None:
            params = net.init(k_net, jnp.zeros((1, FULL_OBS_DIM)))
        norm = norm_init() if norm is None else norm
        env_state, obs = env.reset(k_env, random_t=True)
        return dict(params=params, opt=tx.init(params), norm=norm, env=env_state, obs=obs, key=key)

    def rollout_step(carry, _):
        params, norm, env_state, obs, key = carry
        key, k = jax.random.split(key)
        nobs = norm_apply(norm, obs)
        mean, log_std, value = net.apply(params, nobs)
        action = mean + jnp.exp(log_std) * jax.random.normal(k, mean.shape)
        logp = _logp(mean, log_std, action)
        env_state, next_obs, reward, done, info = env.step(env_state, action)
        v_final = net.apply(params, norm_apply(norm, info["final_obs"]), method=net.value)
        reward = reward + cfg.gamma * v_final * info["truncated"]
        d = done.astype(jnp.float32)
        stats = dict(
            episodes=d.sum(), ep_return=(info["ep_return"] * d).sum(), ep_len=(info["ep_len"] * d).sum(),
            ep_gates=(info["ep_gates"] * d).sum(), crash_gate=info["crash_gate"].sum(),
            ground=(info["ground"] & ~info["crash_gate"]).sum(), oob=(info["oob"] & ~info["crash_gate"]).sum(),
            truncated=info["truncated"].sum(), gates=info["passed"].sum(), in_fov=info["in_fov"].mean(),
            speed=info["speed"].mean(), reward=reward.mean(), wrong_next=info["wrong_next"].mean(),
        )  # fmt: skip
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
        aux = dict(pg_loss=pg, vf_loss=vf, entropy=entropy, kl=(mb.logp - logp).mean(),
                   clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean())  # fmt: skip
        return loss, aux

    @jax.jit
    def iterate(runner):
        params, opt, norm = runner["params"], runner["opt"], runner["norm"]
        carry = (params, norm, runner["env"], runner["obs"], runner["key"])
        (_, _, env_state, obs, key), (traj, raw_obs, stats) = jax.lax.scan(rollout_step, carry, None, cfg.n_steps)
        last_value = net.apply(params, norm_apply(norm, obs), method=net.value)

        def gae_step(c, tr):
            gae, next_value = c
            delta = tr.reward + cfg.gamma * next_value * (1 - tr.done) - tr.value
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - tr.done) * gae
            return (gae, tr.value), gae

        _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        ret = adv + traj.value
        flat = jax.tree.map(lambda x: x.reshape(batch, *x.shape[2:]), (traj, adv, ret))

        def epoch(c, k):
            perm = jax.random.permutation(k, batch)
            mbs = jax.tree.map(lambda x: x[perm].reshape(cfg.n_minibatches, -1, *x.shape[1:]), flat)

            def minibatch(c, mb):
                params, opt = c
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *mb)
                updates, opt = tx.update(grads, opt, params)
                return (clamp_std(optax.apply_updates(params, updates)), opt), aux

            return jax.lax.scan(minibatch, c, mbs)

        key, k = jax.random.split(key)
        (params, opt), aux = jax.lax.scan(epoch, (params, opt), jax.random.split(k, cfg.n_epochs))
        norm = norm_update(norm, raw_obs)
        ep = jnp.maximum(stats["episodes"].sum(), 1.0)
        crashes = stats["crash_gate"].sum() + stats["ground"].sum() + stats["oob"].sum()
        metrics = dict(
            episodes=stats["episodes"].sum(), ep_return=stats["ep_return"].sum() / ep, ep_len=stats["ep_len"].sum() / ep,
            ep_gates=stats["ep_gates"].sum() / ep, crash_rate=crashes / ep, crash_gate=stats["crash_gate"].sum() / ep,
            crash_ground=stats["ground"].sum() / ep, crash_oob=stats["oob"].sum() / ep,
            gates_per_s=stats["gates"].sum() / (batch / env.cfg.policy_hz), in_fov=stats["in_fov"].mean(),
            speed=stats["speed"].mean(), reward=stats["reward"].mean(), wrong_next=stats["wrong_next"].mean(),
            value=traj.value.mean(), explained_var=1 - (ret - traj.value).var() / (ret.var() + 1e-8),
            std=jnp.exp(params["params"]["log_std"]).mean(), **{k: v.mean() for k, v in aux.items()},
        )  # fmt: skip
        return dict(params=params, opt=opt, norm=norm, env=env_state, obs=obs, key=key), metrics

    return net, init, iterate, n_iters


def config_dict(env_cfg: EnvConfig, ecfg: EstEnvConfig, ppo_cfg: PPOConfig) -> dict:
    return {"env": dataclasses.asdict(env_cfg), "est_env": dataclasses.asdict(ecfg), "ppo": dataclasses.asdict(ppo_cfg)}


def save_checkpoint(path, params, norm: ObsNorm, env_cfg, ecfg, ppo_cfg, extra: dict):
    ckpt = dict(params=jax.tree.map(np.asarray, params), obs_norm={k: np.asarray(v) for k, v in norm._asdict().items()},
                config=config_dict(env_cfg, ecfg, ppo_cfg), kind="est_policy", **extra)  # fmt: skip
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(ckpt))
    tmp.rename(path)


def load_checkpoint(path):
    """Returns (net, params, norm, env_cfg, est_env_cfg, ppo_cfg, ckpt)."""
    ckpt = pickle.loads(Path(path).read_bytes())
    env_cfg = EnvConfig(**ckpt["config"]["env"])
    ecfg = EstEnvConfig.from_dict(ckpt["config"]["est_env"])
    ppo = dict(ckpt["config"]["ppo"])
    ppo["hidden"] = tuple(ppo["hidden"])
    ppo_cfg = PPOConfig(**ppo)
    net = AsymActorCritic(ppo_cfg.hidden, ppo_cfg.log_std_init)
    params = jax.tree.map(jnp.asarray, ckpt["params"])
    norm = ObsNorm(**{k: jnp.asarray(v) for k, v in ckpt["obs_norm"].items()})
    return net, params, norm, env_cfg, ecfg, ppo_cfg, ckpt


def write_config(path, env_cfg, ecfg, ppo_cfg, extra: dict | None = None):
    Path(path).write_text(json.dumps(config_dict(env_cfg, ecfg, ppo_cfg) | (extra or {}), indent=2))
