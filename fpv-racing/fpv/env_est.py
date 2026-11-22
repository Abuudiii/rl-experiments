"""Estimated-observation racing environment: RaceEnv dynamics, rewards, randomisation and tracks unchanged,
but the policy sees only the perception pipeline's estimates (`fpv.perception`).

Per policy step (50 Hz): Crazyflow steps (`RaceEnv.step_core`), the simulated IMU samples the step
(`perception.imu`), the detection-noise model turns the true projected gate corners into detector-like
outputs (`perception.detnoise`), and the estimator (`perception.estimator`) updates from detections + IMU +
the episode's NOMINAL gate-order list (``EnvState.track_pos/track_yaw``; the per-episode gate randomisation
and the spawn are unknown to it).

Observation returned by ``reset`` / ``step``: (N, 89) = [estimated observation (46) | true teacher
observation (43)]. The policy (actor) reads only the first 46 (`fpv.perception.estimator.observation`);
the critic may read all 89 (asymmetric actor-critic, training only). Rewards, termination and the true
next gate are the simulator's (privileged, as for the teacher).

Noise levels: every episode draws a detection-noise scale in [det_scale_min, det_scale_max] and an IMU
noise scale in [imu_scale_min, imu_scale_max] (uniform), a one-frame detector latency with probability
p_latency; dropouts of the true next gate follow ``det.dropout_rate`` (`DetNoiseConfig`).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import struct

from fpv.env import OBS_DIM, EnvConfig, EnvState, RaceEnv
from fpv.perception import detnoise, estimator, imu
from fpv.perception.detnoise import DetNoiseConfig
from fpv.perception.estimator import EST_OBS_DIM, EstConfig
from fpv.perception.imu import IMUConfig

FULL_OBS_DIM = EST_OBS_DIM + OBS_DIM


@dataclasses.dataclass(frozen=True)
class EstEnvConfig:
    det_scale_min: float = 1.0
    det_scale_max: float = 1.0
    imu_scale_min: float = 1.0
    imu_scale_max: float = 1.0
    p_latency: float = 0.0
    # Contract addition (reported separately): the pipeline is told at episode start which list index is the
    # FIRST gate to fly (a race start box); still no pose. Used only for the first lock of an episode.
    known_start: bool = False
    det: DetNoiseConfig = DetNoiseConfig()
    imu: IMUConfig = IMUConfig()
    est: EstConfig = EstConfig()

    @staticmethod
    def from_dict(d: dict) -> "EstEnvConfig":
        """Rebuild from `dataclasses.asdict` output; fields unknown to this version (older checkpoints) are
        dropped, missing ones take the current defaults."""
        d = dict(d)

        def known(cls, x):
            names = {f.name for f in dataclasses.fields(cls)}
            return cls(**{k: v for k, v in x.items() if k in names})

        det, imu_, est = d.pop("det"), d.pop("imu"), d.pop("est")
        return known(EstEnvConfig, d | dict(det=known(DetNoiseConfig, det), imu=known(IMUConfig, imu_),
                                            est=known(EstConfig, est)))  # fmt: skip


# Named noise conditions (training variants and evaluation conditions of stage 4)
def condition(name: str, base: EstEnvConfig = EstEnvConfig()) -> EstEnvConfig:
    det = base.det

    def fixed(s):
        return dict(det_scale_min=s, det_scale_max=s, imu_scale_min=s, imu_scale_max=s)

    table = {
        "perfect": fixed(0.0) | dict(p_latency=0.0, det=dataclasses.replace(det, dropout_rate=0.0)),
        "x1": fixed(1.0) | dict(det=dataclasses.replace(det, dropout_rate=0.0)),
        "x2": fixed(2.0) | dict(det=dataclasses.replace(det, dropout_rate=0.0)),
        "x4": fixed(4.0) | dict(det=dataclasses.replace(det, dropout_rate=0.0)),
        "x05": fixed(0.5) | dict(det=dataclasses.replace(det, dropout_rate=0.0)),
        # next-gate dropouts of up to 0.5 s at 1x noise: on average one per second of flight
        "drop": fixed(1.0) | dict(det=dataclasses.replace(det, dropout_rate=1.0, dropout_max_steps=25)),
        # training mix: scales U(0, 4), dropouts, latency on a quarter of the episodes
        "rand": dict(det_scale_min=0.0, det_scale_max=4.0, imu_scale_min=0.0, imu_scale_max=4.0, p_latency=0.25,
                     det=dataclasses.replace(det, dropout_rate=0.5, dropout_max_steps=25)),
    }  # fmt: skip
    return dataclasses.replace(base, **table[name])


@struct.dataclass
class EstEnvState:
    race: EnvState
    det: detnoise.DetState
    imu: imu.IMUState
    est: estimator.EstState
    latency: jax.Array  # (N,) bool
    est_obs: jax.Array  # (N, 46) last estimated observation
    start_idx: jax.Array  # (N,) int first gate of the episode if known_start, else -1
    key: jax.Array


class EstEnv:
    """Same interface as `RaceEnv` (reset / step), plus ``step_core_est`` for evaluation."""

    def __init__(self, cfg: EnvConfig, ecfg: EstEnvConfig = EstEnvConfig(), circuit=None, device=None):
        self.race = RaceEnv(cfg, circuit=circuit, device=device)
        self.cfg, self.ecfg = cfg, ecfg
        self.dt = 1.0 / cfg.policy_hz

    def _scales(self, key, n):
        e = self.ecfg
        k1, k2, k3 = jax.random.split(key, 3)
        ds = jax.random.uniform(k1, (n,), minval=e.det_scale_min, maxval=e.det_scale_max)
        is_ = jax.random.uniform(k2, (n,), minval=e.imu_scale_min, maxval=e.imu_scale_max)
        lat = jax.random.uniform(k3, (n,)) < e.p_latency
        return ds, is_, lat

    def _start_idx(self, race: EnvState):
        return jnp.where(self.ecfg.known_start, race.target, -1)

    def _perceive(self, race: EnvState, det_st, est_st, latency, gyro, acc, dt, key, start_idx):
        s = race.data.states
        prev_c, prev_conf = det_st.prev_corners, det_st.prev_conf
        det_st, corners, conf, daux = detnoise.detect(det_st, key, s.pos[:, 0], s.quat[:, 0], race.gates_pos,
                                                      race.gates_yaw, race.target, self.ecfg.det, self.dt)  # fmt: skip
        corners = jnp.where(latency[:, None, None, None], prev_c, corners)
        conf = jnp.where(latency[:, None], prev_conf, conf)
        est_st, einfo = estimator.est_update(est_st, corners, conf, gyro, acc, race.track_pos, race.track_yaw, dt,
                                             self.ecfg.est, start_idx)  # fmt: skip
        obs = estimator.observation(est_st, race.track_pos, race.track_yaw, race.prev_action, self.ecfg.est)
        return det_st, est_st, obs, dict(det=daux, est=einfo, corners=corners, conf=conf)

    def reset(self, key, random_t: bool = False):
        race, true_obs = self.race.reset(key, random_t)
        n = self.cfg.n_envs
        k = jax.random.fold_in(key, 0xE57)
        k1, k2, k3, k4, k5 = jax.random.split(k, 5)
        ds, is_, lat = self._scales(k1, n)
        det_st = detnoise.det_init(k2, ds, self.race.n_gates)
        imu_st = imu.imu_init(k3, is_, self.ecfg.imu)
        est_st = estimator.est_init(n, self.ecfg.est)
        z = jnp.zeros((n, 3))
        sidx = self._start_idx(race)
        det_st, est_st, obs, _ = self._perceive(race, det_st, est_st, lat, z, z, jnp.zeros(n), k4, sidx)
        st = EstEnvState(race=race, det=det_st, imu=imu_st, est=est_st, latency=lat, est_obs=obs, start_idx=sidx,
                         key=k5)  # fmt: skip
        return st, jnp.concatenate([obs, true_obs], -1)

    def step_core_est(self, state: EstEnvState, action, reset_done: bool = True):
        """Physics + perception. With ``reset_done`` finished episodes restart (training); without, the
        estimator keeps running on the ended episode (evaluation masks it out). Returns
        (state, full_obs, reward, done, info)."""
        n = self.cfg.n_envs
        key, kr, ki, kd, ks, kn = jax.random.split(state.key, 6)
        s0 = state.race.data.states
        race, reward, ev = self.race.step_core(state.race, action)
        s1 = race.data.states
        imu_st, gyro, acc = imu.imu_measure(state.imu, ki, s0.quat[:, 0], s1.quat[:, 0], s0.vel[:, 0], s1.vel[:, 0],
                                            s0.ang_vel[:, 0], s1.ang_vel[:, 0], self.dt, self.ecfg.imu)  # fmt: skip
        done = ev["terminated"] | ev["truncated"]
        true_final = self.race.obs(race)
        final_full = jnp.concatenate([state.est_obs, true_final], -1)  # critic bootstrap (previous estimate)
        det_st, est_st, lat, sidx = state.det, state.est, state.latency, state.start_idx
        dt = jnp.full(n, self.dt)
        if reset_done:
            race = self.race._reset_masked(race, kr, done)
            ds, is_, lat_new = self._scales(ks, n)
            k_d, k_i = jax.random.split(kn)
            det_st = detnoise.det_reset_masked(det_st, k_d, done, ds)
            imu_st = imu.imu_reset_masked(imu_st, k_i, done, is_)
            est_st = estimator.est_reset_masked(est_st, done, self.ecfg.est)
            lat = jnp.where(done, lat_new, lat)
            sidx = jnp.where(done, self._start_idx(race), sidx)
            gyro, acc = jnp.where(done[:, None], 0.0, gyro), jnp.where(done[:, None], 0.0, acc)
            dt = jnp.where(done, 0.0, dt)
        det_st, est_st, obs, pinfo = self._perceive(race, det_st, est_st, lat, gyro, acc, dt, kd, sidx)
        full = jnp.concatenate([obs, self.race.obs(race)], -1)
        state = EstEnvState(race=race, det=det_st, imu=imu_st, est=est_st, latency=lat, est_obs=obs, start_idx=sidx,
                            key=key)  # fmt: skip
        info = ev | dict(final_obs=final_full, ep_return=race.ep_return, ep_len=race.t, ep_gates=race.ep_gates,
                         wrong_next=est_st.nxt != race.target, gyro=gyro, acc=acc, percep=pinfo)  # fmt: skip
        return state, full, reward, done, info

    def step(self, state: EstEnvState, action):
        """RaceEnv.step semantics: auto-reset; ``info["final_obs"]`` for bootstrapping at truncation (its
        estimated part is the previous step's estimate). ``ep_*`` hold the finished episode's totals where
        done (read before the reset)."""
        r = state.race
        state2, full, reward, done, info = self.step_core_est(state, action, reset_done=True)
        # episode totals of the finished episode (the reset zeroed them)
        info["ep_return"] = jnp.where(done, r.ep_return + reward, state2.race.ep_return)
        info["ep_len"] = jnp.where(done, r.t + 1, state2.race.t)
        info["ep_gates"] = jnp.where(done, r.ep_gates + info["passed"], state2.race.ep_gates)
        return state2, full, reward, done, info
