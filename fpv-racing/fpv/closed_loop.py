"""Crazyflow half of closed-loop flight with an external (e.g. vision) policy.

    fb = FlightBatch(n, "runs/teacher/<run>/ckpt_latest.pkl", seed=0, randomize=True, wide_spawn_p=0.0)
    s = fb.reset()                 # all environments; or fb.reset(mask) for a boolean [n] subset
    while ...:
        s = fb.step(actions)       # float32 [n, 4] in [-1, 1], fpv.env.scale_action mapping, 50 Hz
        if s["done"].any():
            s = fb.reset(s["done"])

numpy in, numpy out, JAX inside. Device: "gpu" (the first GPU visible through CUDA_VISIBLE_DEVICES),
"cpu", or None = gpu if jax has one (``JAX_PLATFORMS=cpu`` forces cpu). The environment (reward
terms aside), randomisation ranges, gate logic, time limit and wide-spawn ranges are those of the
teacher checkpoint's training config; ``randomize=False`` gives the nominal circuit and nominal
physics (spawn noise stays), ``wide_spawn_p`` overrides ``p_wide``.

State dict (leading dim n, describes the CURRENT state, i.e. after the step / reset):
    pos [n,3], quat [n,4] xyzw body-to-world, vel [n,3] world, ang_vel [n,3] body,
    prev_action [n,4]   last applied (clipped) action; after a reset the spawn's previous action
    target [n]          index of the next gate
    gates_pos [n,6,3], gates_yaw [n,6]   this episode's layout
    teacher_obs [n,43]  fpv.env.teacher_obs of the current state
    teacher_action [n,4]  deterministic teacher action for the current state (the label for NOW)
    done [n] bool       episode over (latched until reset); outcome [n]: 0 running or time limit,
                        1 gate crash, 2 ground, 3 out of bounds (as in rollouts.npz)
    passed [n] bool     the last step carried the drone through its target gate
    gates_passed_total [n], t [n] (policy steps since the reset; time = t / 50 s)

There is no auto-reset: environments that are ``done`` are frozen (further ``step`` calls leave
them unchanged and ignore their actions) until the caller resets them. A reset draws a fresh gate
layout, physical parameters and spawn exactly as training does.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from fpv import ppo
from fpv.env import RaceEnv

_EP_FIELDS = ("gates_pos", "gates_yaw", "target", "prev_action", "delay", "act_noise", "t", "ep_return", "ep_gates")


class FlightBatch:
    def __init__(self, n: int, teacher_ckpt, seed: int = 0, randomize: bool = True, wide_spawn_p: float = 0.0,
                 device: str | None = None):  # fmt: skip
        from crazyflow.utils import pytree_replace, world_mask

        net, params, norm, env_cfg, _, _ = ppo.load_checkpoint(teacher_ckpt)
        cfg = dataclasses.replace(env_cfg, n_envs=n, p_wide=wide_spawn_p, rand_action_noise=0.0)
        if not randomize:
            cfg = cfg.nominal()
        self.n, self.cfg = n, cfg
        env = self.env = RaceEnv(cfg, device=device)
        dev = env.default.states.pos.device
        params, norm = jax.device_put((params, norm), dev)
        self._key = jax.device_put(jax.random.key(seed), dev)
        self._dev = dev

        def view(state, done, outcome, passed):
            s = state.data.states
            obs = env.obs(state)
            return dict(
                pos=s.pos[:, 0], quat=s.quat[:, 0], vel=s.vel[:, 0], ang_vel=s.ang_vel[:, 0],
                prev_action=state.prev_action, target=state.target, gates_pos=state.gates_pos,
                gates_yaw=state.gates_yaw, teacher_obs=obs,
                teacher_action=ppo.policy_action(net, params, norm, obs), done=done, outcome=outcome,
                passed=passed, gates_passed_total=state.ep_gates, t=state.t,
            )  # fmt: skip

        @jax.jit
        def reset(state, done, outcome, key, mask):
            state = env._reset_masked(state, key, mask)
            done, outcome = done & ~mask, jnp.where(mask, 0, outcome)
            return state, done, outcome, view(state, done, outcome, jnp.zeros_like(done))

        @jax.jit
        def step(state, done, outcome, action):
            new, _, ev = env.step_core(state, action)
            # Finished environments stay frozen until the caller resets them
            data = pytree_replace(new.data, state.data, world_mask(new.data), done)
            keep = {
                f: jnp.where(done.reshape(-1, *[1] * (getattr(new, f).ndim - 1)), getattr(state, f), getattr(new, f))
                for f in _EP_FIELDS
            }
            new = new.replace(data=data, **keep)
            ended = ~done & (ev["terminated"] | ev["truncated"])
            code = jnp.where(ev["crash_gate"], 1, jnp.where(ev["ground"], 2, jnp.where(ev["oob"], 3, 0)))
            outcome = jnp.where(ended, code, outcome)
            done_new = done | ended
            return new, done_new, outcome, view(new, done_new, outcome, ev["passed"] & ~done)

        self._reset, self._step = reset, step
        self._state, _ = jax.jit(env.reset)(self._key)  # placeholder, replaced by the first reset()
        self._done = jax.device_put(jnp.ones(n, bool), dev)
        self._outcome = jax.device_put(jnp.zeros(n, jnp.int32), dev)
        self._started = False

    def reset(self, mask=None) -> dict:
        """Start new episodes in all environments (``mask=None``) or where ``mask`` [n] is True."""
        if mask is None or not self._started:
            mask = np.ones(self.n, bool)
        self._started = True
        self._key, k = jax.random.split(self._key)
        mask = jax.device_put(jnp.asarray(np.asarray(mask, bool)), self._dev)
        self._state, self._done, self._outcome, out = self._reset(self._state, self._done, self._outcome, k, mask)
        return {k: np.asarray(v) for k, v in out.items()}

    def step(self, actions) -> dict:
        """Apply ``actions`` [n, 4] in [-1, 1] for one 20 ms policy step."""
        assert self._started, "call reset() first"
        a = jax.device_put(jnp.asarray(np.asarray(actions, np.float32).reshape(self.n, 4)), self._dev)
        self._state, self._done, self._outcome, out = self._step(self._state, self._done, self._outcome, a)
        return {k: np.asarray(v) for k, v in out.items()}
