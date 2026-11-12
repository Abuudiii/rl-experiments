"""Batched, jittable gate-racing environment on top of Crazyflow (the only dynamics).

Action (policy output, 50 Hz), ``a`` in [-1, 1]^4 = [collective thrust, roll rate, pitch rate,
yaw rate]. `scale_action` is THE mapping to physical commands (use it for every policy):

    thrust_N = (a[0] + 1) / 2 * MAX_THRUST_N          in [0, 0.8] N (cf21B_500, 4 motors)
    wx, wy   = a[1] * 10 rad/s, a[2] * 10 rad/s        body roll / pitch rate
    wz       = a[3] * 5 rad/s                          body yaw rate

Teacher observation, `teacher_obs`, 43 float32 (before the running normaliser of the policy):

    [0:3]    linear velocity in the body frame / 5 m/s
    [3:12]   body-to-world rotation matrix R, row-major (R[0,0], R[0,1], ..., R[2,2])
    [12:15]  body angular velocity / [10, 10, 5] rad/s
    [15:27]  next gate: 4 opening corners in the body frame / 5 m, corner order top-left,
             top-right, bottom-right, bottom-left (seen along the travel direction), xyz each
    [27:39]  gate after the next: same layout
    [39:43]  previous action (clipped to [-1, 1])

Body frame: x forward, y left, z up (thrust axis). Quaternions are xyzw (scipy / Crazyflow).
"""

from __future__ import annotations

import dataclasses
import math
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["SCIPY_ARRAY_API"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from fpv import track

G = 9.81
MAX_THRUST_N = 0.8  # 4 * thrust_max of cf21B_500 (asserted against Crazyflow in RaceEnv)
MAX_RATE_XY = 10.0  # rad/s
MAX_RATE_Z = 5.0  # rad/s
OBS_DIM = 43
ACT_DIM = 4
VEL_SCALE = 5.0
POS_SCALE = 5.0
CAM_PITCH = math.radians(track.CAMERA["pitch_up_deg"])
# Camera axes in the body frame: optical axis (forward, pitched up), left, up
CAM_FWD = np.array([math.cos(CAM_PITCH), 0.0, math.sin(CAM_PITCH)], dtype=np.float32)
CAM_LEFT = np.array([0.0, 1.0, 0.0], dtype=np.float32)
CAM_UP = np.array([-math.sin(CAM_PITCH), 0.0, math.cos(CAM_PITCH)], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    n_envs: int = 8192
    sim_hz: int = 500
    policy_hz: int = 50
    episode_s: float = 30.0  # time limit (truncation)
    # Spawn: `spawn_dist` m before a random gate, roughly facing it, roughly at rest
    spawn_dist_min: float = 1.0
    spawn_dist_max: float = 2.0
    spawn_lateral: float = 0.3  # +/- m, sideways and vertical offset from the gate axis
    spawn_yaw: float = 0.4  # +/- rad about the gate travel direction
    spawn_tilt: float = 0.15  # +/- rad roll / pitch
    spawn_vel: float = 0.3  # +/- m/s per axis
    spawn_ang_vel: float = 0.3  # +/- rad/s per axis
    # Wide spawn (off by default): with probability p_wide the episode starts in flight at a random
    # point on the straight segment between two consecutive gate centres, target = the gate at the
    # end of the segment, with a random attitude, velocity, body rates and previous action
    p_wide: float = 0.0
    wide_lateral: float = 1.0  # +/- m horizontal offset perpendicular to the segment
    wide_vertical: float = 0.5  # +/- m (kept >= ground_z + 0.3)
    wide_yaw: float = 1.0472  # +/- rad about the direction to the target gate centre
    wide_tilt: float = 0.5236  # +/- rad roll and pitch
    wide_speed_max: float = 5.0  # speed uniform in [0, max] m/s
    wide_vel_bias: float = 1.0  # direction = normalise(bias * segment direction + random unit vector)
    wide_ang_vel: float = 3.0  # +/- rad/s per axis
    # Per-episode randomisation (relative unless noted)
    rand_mass: float = 0.10
    rand_thrust: float = 0.05  # per-motor thrust curve
    rand_rotor: float = 0.10  # per-motor rotor lag coefficients
    rand_drag: float = 0.30
    rand_gate_pos: float = 0.3  # +/- m per axis
    rand_gate_yaw: float = 0.2  # +/- rad
    delay_max_substeps: int = 3  # action delay drawn from {0..n} physics steps (2 ms each)
    # Action perturbation (off by default): Gaussian noise added to the policy action every step,
    # std drawn per episode from [0, rand_action_noise]; the perturbed action is what is flown and
    # what appears as "previous action". Trains tolerance to an imperfect pilot.
    rand_action_noise: float = 0.0
    p_action_noise: float = 1.0  # fraction of episodes that are perturbed at all
    # Termination
    gate_margin: float = 0.05  # drone radius: opening shrinks and frame grows by this [m]
    ground_z: float = 0.10
    box_margin: float = 3.0  # horizontal margin around the nominal circuit
    z_max: float = 3.5
    # Reward
    w_progress: float = 1.0  # per metre of distance reduction to the next gate centre
    w_gate: float = 1.0  # per gate passed
    w_crash: float = 5.0  # on gate crash / ground / out of bounds
    w_rate: float = 5e-4  # * |body rate|^2 [rad/s]
    w_dact: float = 0.02  # * |a_t - a_{t-1}|^2
    # Camera alignment: w_cam * (exp(-cam_lambda * theta^4) - cam_offset), theta = angle between the
    # FPV optical axis and the next gate centre. cam_offset=1 makes it a pure penalty (<= 0), so
    # hovering in front of a gate earns nothing.
    w_cam: float = 0.03
    cam_lambda: float = 10.0
    cam_offset: float = 1.0
    # Optional soft speed limit: - w_speed * max(0, |v| - speed_limit)^2 (off by default)
    w_speed: float = 0.0
    speed_limit: float = 100.0

    @property
    def substeps(self) -> int:
        return self.sim_hz // self.policy_hz

    @property
    def max_steps(self) -> int:
        return round(self.episode_s * self.policy_hz)

    def nominal(self, **kw) -> "EnvConfig":
        """Same task without dynamics / gate randomisation (spawn noise stays)."""
        zero = dict(rand_mass=0.0, rand_thrust=0.0, rand_rotor=0.0, rand_drag=0.0, rand_gate_pos=0.0,
                    rand_gate_yaw=0.0, delay_max_substeps=0)  # fmt: skip
        return dataclasses.replace(self, **(zero | kw))


@struct.dataclass
class EnvState:
    data: object  # Crazyflow SimData
    gates_pos: jax.Array  # (N, G, 3) per-environment layout
    gates_yaw: jax.Array  # (N, G)
    target: jax.Array  # (N,) int32 index of the next gate
    prev_action: jax.Array  # (N, 4) clipped
    delay: jax.Array  # (N,) int32 physics steps
    t: jax.Array  # (N,) int32 policy steps in this episode
    ep_return: jax.Array  # (N,)
    ep_gates: jax.Array  # (N,) int32 gates passed in this episode
    act_noise: jax.Array  # (N,) std of the action perturbation in this episode
    key: jax.Array


# region pure helpers (shared with evaluation and the later vision policy)


def scale_action(action):
    """Policy action in [-1, 1]^4 -> physical command [thrust N, wx, wy, wz rad/s] (clips first)."""
    a = jnp.clip(action, -1.0, 1.0)
    return jnp.stack(
        [(a[..., 0] + 1) * (MAX_THRUST_N / 2), a[..., 1] * MAX_RATE_XY, a[..., 2] * MAX_RATE_XY, a[..., 3] * MAX_RATE_Z],
        axis=-1,
    )


def quat_to_mat(q):
    """xyzw quaternion (..., 4) -> body-to-world rotation matrix (..., 3, 3)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            jnp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            jnp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def _take_gate(arr, idx):
    """arr (N, G, ...) indexed per environment by idx (N,)."""
    return jnp.take_along_axis(arr, idx.reshape(-1, *[1] * (arr.ndim - 1)), axis=1)[:, 0]


def teacher_obs(pos, quat, vel, ang_vel, gates_pos, gates_yaw, target, prev_action):
    """Teacher observation (N, 43); layout in the module docstring.

    Args: pos (N, 3) world, quat (N, 4) xyzw, vel (N, 3) world, ang_vel (N, 3) body,
    gates_pos (N, G, 3), gates_yaw (N, G), target (N,) int, prev_action (N, 4).
    """
    rot = quat_to_mat(quat)  # body -> world
    n_gates = gates_pos.shape[1]
    vel_b = jnp.einsum("nji,nj->ni", rot, vel)
    parts = [vel_b / VEL_SCALE, rot.reshape(-1, 9), ang_vel / jnp.asarray([MAX_RATE_XY, MAX_RATE_XY, MAX_RATE_Z])]
    for k in range(2):
        idx = (target + k) % n_gates
        corners = track.gate_corners(_take_gate(gates_pos, idx), _take_gate(gates_yaw, idx))  # (N, 4, 3)
        rel = jnp.einsum("nji,ncj->nci", rot, corners - pos[:, None, :])
        parts.append(rel.reshape(-1, 12) / POS_SCALE)
    parts.append(prev_action)
    return jnp.concatenate(parts, axis=-1)


def camera_view(pos, quat, point):
    """Angle [rad] between the FPV optical axis and ``point``, and whether it is inside the FOV."""
    rot = quat_to_mat(quat)
    d = jnp.einsum("nji,nj->ni", rot, point - pos)
    d = d / (jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-6)
    fwd, left, up = d @ jnp.asarray(CAM_FWD), d @ jnp.asarray(CAM_LEFT), d @ jnp.asarray(CAM_UP)
    tan_h, tan_v = track.camera_half_fov_tan()
    in_fov = (fwd > 1e-6) & (jnp.abs(left) < tan_h * fwd) & (jnp.abs(up) < tan_v * fwd)
    return jnp.arccos(jnp.clip(fwd, -1.0, 1.0)), in_fov


# region environment


class RaceEnv:
    """Pure ``reset`` / ``step`` functions over a Crazyflow ``Sim`` (closed over, not mutated)."""

    def __init__(self, cfg: EnvConfig, circuit=track.CIRCUIT, device: str | None = None):
        """``device``: "gpu" / "cpu" for the simulation data (default: gpu if jax has one)."""
        import crazyflow.sim.functional as F
        from crazyflow.control import Control
        from crazyflow.control.transform import motor_force2rotor_vel
        from crazyflow.drones import Drone
        from crazyflow.dynamics import Dynamics, load_params
        from crazyflow.sim import Sim
        from crazyflow.utils import leaf_replace

        self.cfg, self._F, self._leaf_replace = cfg, F, leaf_replace
        n = cfg.n_envs
        drone = Drone.cf21B_500
        sim = Sim(
            n_worlds=n,
            n_drones=1,
            drone=drone,
            dynamics=Dynamics.first_principles,
            control=Control.body_rate,
            freq=cfg.sim_hz,
            body_rate_freq=cfg.sim_hz,
            force_torque_freq=cfg.sim_hz,
            device=device or ("cpu" if jax.default_backend() == "cpu" else "gpu"),
            fused_mjx_model=True,
        )
        p = load_params(Dynamics.first_principles, drone)
        self.mass = float(p["mass"])
        assert abs(4 * float(p["thrust_max"]) - MAX_THRUST_N) < 0.01, p["thrust_max"]
        self.hover_action = 2 * self.mass * G / MAX_THRUST_N - 1
        data = sim.data
        # Pure rate tracking: drop the firmware's attitude-levelling terms
        br = data.controls.body_rate
        br = br.replace(params=br.params | {"kR": jnp.zeros(3), "ki_m": jnp.zeros(3)})
        # Rotors spin at hover RPM after a reset
        rpm = motor_force2rotor_vel(np.full(4, self.mass * G / 4), np.asarray(p["rpm2thrust"]))
        states = data.states.replace(
            pos=data.states.pos.at[..., 2].set(1.0),
            rotor_vel=jnp.broadcast_to(jnp.asarray(rpm, dtype=jnp.float32), (n, 1, 4)),
        )
        # Per-world parameter shapes, fixed from the start (nominal values)
        d = data.params
        params = d.replace(
            mass=jnp.broadcast_to(d.mass, (n, 1, 1)),
            rpm2thrust=jnp.broadcast_to(d.rpm2thrust, (n, 1, 4, 3)),
            rotor_dyn_coef=jnp.broadcast_to(d.rotor_dyn_coef, (n, 1, 4, 4)),
            drag_matrix=jnp.broadcast_to(d.drag_matrix, (n, 1, 3, 3)),
        )
        self.default = data.replace(states=states, params=params, controls=data.controls.replace(body_rate=br))
        self._sim, self._sim_step, self._sim_reset = sim, sim._step, sim._reset
        gp, gy = track.circuit_arrays(circuit)
        self.gates_pos, self.gates_yaw = jnp.asarray(gp), jnp.asarray(gy)
        self.n_gates = gp.shape[0]
        m = cfg.box_margin
        self.box_lo = jnp.asarray([gp[:, 0].min() - m, gp[:, 1].min() - m, cfg.ground_z])
        self.box_hi = jnp.asarray([gp[:, 0].max() + m, gp[:, 1].max() + m, cfg.z_max])

    # region reset

    def _sample(self, key):
        """Draw a new episode for every environment (masked in by the caller)."""
        c, n, ng = self.cfg, self.cfg.n_envs, self.n_gates
        ks = jax.random.split(key, 16)

        def u(k, shape, r):
            return jax.random.uniform(k, shape, minval=-1.0, maxval=1.0) * r

        gates_pos = self.gates_pos + u(ks[0], (n, ng, 3), c.rand_gate_pos)
        gates_yaw = self.gates_yaw + u(ks[1], (n, ng), c.rand_gate_yaw)
        target = jax.random.randint(ks[2], (n,), 0, ng)
        gp, gy = _take_gate(gates_pos, target), _take_gate(gates_yaw, target)
        normal, left = track.gate_axes(gy)
        dist = jax.random.uniform(ks[3], (n, 1), minval=c.spawn_dist_min, maxval=c.spawn_dist_max)
        off = u(ks[4], (n, 2), c.spawn_lateral)
        pos = gp - dist * normal + off[:, :1] * left
        pos = pos.at[:, 2].set(jnp.maximum(gp[:, 2] + off[:, 1], c.ground_z + 0.3))
        yaw = gy + u(ks[5], (n,), c.spawn_yaw)
        tilt = u(ks[6], (n, 2), c.spawn_tilt)
        zero = jnp.zeros_like(yaw)
        q_yaw = jnp.stack([zero, zero, jnp.sin(yaw / 2), jnp.cos(yaw / 2)], -1)
        q_tilt = jnp.concatenate([tilt / 2, zero[:, None], jnp.ones((n, 1))], -1)
        quat = quat_mul(q_yaw, q_tilt / jnp.linalg.norm(q_tilt, axis=-1, keepdims=True))
        vel, ang_vel = u(ks[7], (n, 3), c.spawn_vel), u(ks[8], (n, 3), c.spawn_ang_vel)
        prev_action = jnp.zeros((n, 4)).at[:, 0].set(self.hover_action)
        if c.p_wide > 0:
            kw = jax.random.split(ks[14], 12)
            wide = jax.random.uniform(kw[0], (n,)) < c.p_wide
            start = _take_gate(gates_pos, (target - 1) % ng)
            seg = gp - start
            seg_dir = seg / (jnp.linalg.norm(seg, axis=-1, keepdims=True) + 1e-6)
            side = jnp.stack([-seg_dir[:, 1], seg_dir[:, 0], jnp.zeros(n)], -1)
            side = side / (jnp.linalg.norm(side, axis=-1, keepdims=True) + 1e-6)
            w_pos = start + jax.random.uniform(kw[1], (n, 1)) * seg + u(kw[2], (n, 1), c.wide_lateral) * side
            w_pos = w_pos.at[:, 2].set(jnp.maximum(w_pos[:, 2] + u(kw[3], (n,), c.wide_vertical), c.ground_z + 0.3))
            to_gate = gp - w_pos
            w_yaw = jnp.arctan2(to_gate[:, 1], to_gate[:, 0]) + u(kw[4], (n,), c.wide_yaw)
            rv = jnp.concatenate([u(kw[5], (n, 2), c.wide_tilt), zero[:, None]], -1)  # rotation vector
            ang = jnp.linalg.norm(rv, axis=-1, keepdims=True)
            w_tilt = jnp.concatenate([jnp.sin(ang / 2) * rv / (ang + 1e-9), jnp.cos(ang / 2)], -1)
            w_quat = quat_mul(jnp.stack([zero, zero, jnp.sin(w_yaw / 2), jnp.cos(w_yaw / 2)], -1), w_tilt)
            rnd = jax.random.normal(kw[6], (n, 3))
            v_dir = c.wide_vel_bias * seg_dir + rnd / (jnp.linalg.norm(rnd, axis=-1, keepdims=True) + 1e-6)
            v_dir = v_dir / (jnp.linalg.norm(v_dir, axis=-1, keepdims=True) + 1e-6)
            w_vel = v_dir * jax.random.uniform(kw[7], (n, 1)) * c.wide_speed_max
            sel = wide[:, None]
            pos, quat, vel = jnp.where(sel, w_pos, pos), jnp.where(sel, w_quat, quat), jnp.where(sel, w_vel, vel)
            ang_vel = jnp.where(sel, u(kw[8], (n, 3), c.wide_ang_vel), ang_vel)
            prev_action = jnp.where(sel, u(kw[9], (n, 4), 1.0), prev_action)
        d = self.default.params
        return dict(
            gates_pos=gates_pos,
            gates_yaw=gates_yaw,
            target=target,
            pos=pos,
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
            prev_action=prev_action,
            mass=d.mass * (1 + u(ks[9], (n, 1, 1), c.rand_mass)),
            rpm2thrust=d.rpm2thrust * (1 + u(ks[10], (n, 1, 4, 1), c.rand_thrust)),
            rotor_dyn_coef=d.rotor_dyn_coef * (1 + u(ks[11], (n, 1, 4, 1), c.rand_rotor)),
            drag_matrix=d.drag_matrix * (1 + u(ks[12], (n, 1, 1, 1), c.rand_drag)),
            delay=jax.random.randint(ks[13], (n,), 0, c.delay_max_substeps + 1),
            act_noise=jax.random.uniform(ks[15], (n,)) * c.rand_action_noise
            * (jax.random.uniform(jax.random.fold_in(ks[15], 1), (n,)) < c.p_action_noise),
        )

    def _reset_masked(self, state: EnvState, key, mask) -> EnvState:
        ep, lr = self._sample(key), self._leaf_replace
        data = self._sim_reset(state.data, self.default, mask)
        states = lr(
            data.states, mask, pos=ep["pos"][:, None], quat=ep["quat"][:, None], vel=ep["vel"][:, None],
            ang_vel=ep["ang_vel"][:, None],
        )  # fmt: skip
        params = lr(
            data.params, mask, mass=ep["mass"], rpm2thrust=ep["rpm2thrust"],
            rotor_dyn_coef=ep["rotor_dyn_coef"], drag_matrix=ep["drag_matrix"],
        )  # fmt: skip
        br = lr(data.controls.body_rate, mask, prev_ang_vel=ep["ang_vel"][:, None])
        data = data.replace(states=states, params=params, controls=data.controls.replace(body_rate=br))

        def sel(new, old):
            return jnp.where(mask.reshape(-1, *[1] * (old.ndim - 1)), new, old)

        return state.replace(
            data=data,
            gates_pos=sel(ep["gates_pos"], state.gates_pos),
            gates_yaw=sel(ep["gates_yaw"], state.gates_yaw),
            target=sel(ep["target"], state.target),
            prev_action=sel(ep["prev_action"], state.prev_action),
            delay=sel(ep["delay"], state.delay),
            act_noise=sel(ep["act_noise"], state.act_noise),
            t=sel(0, state.t),
            ep_return=sel(0.0, state.ep_return),
            ep_gates=sel(0, state.ep_gates),
        )

    def reset(self, key, random_t: bool = False) -> tuple[EnvState, jax.Array]:
        """Start a new episode everywhere. ``random_t`` staggers the time limits (training)."""
        n, ng = self.cfg.n_envs, self.n_gates
        key, k1, k2 = jax.random.split(key, 3)
        state = EnvState(
            data=self.default,
            gates_pos=jnp.zeros((n, ng, 3)),
            gates_yaw=jnp.zeros((n, ng)),
            target=jnp.zeros(n, jnp.int32),
            prev_action=jnp.zeros((n, 4)),
            delay=jnp.zeros(n, jnp.int32),
            act_noise=jnp.zeros(n),
            t=jnp.zeros(n, jnp.int32),
            ep_return=jnp.zeros(n),
            ep_gates=jnp.zeros(n, jnp.int32),
            key=key,
        )
        state = self._reset_masked(state, k1, jnp.ones(n, bool))
        if random_t:
            state = state.replace(t=jax.random.randint(k2, (n,), 0, self.cfg.max_steps))
        return state, self.obs(state)

    # region step

    def obs(self, state: EnvState) -> jax.Array:
        s = state.data.states
        o = teacher_obs(
            s.pos[:, 0], s.quat[:, 0], s.vel[:, 0], s.ang_vel[:, 0], state.gates_pos, state.gates_yaw,
            state.target, state.prev_action,
        )  # fmt: skip
        return jnp.nan_to_num(o, posinf=0.0, neginf=0.0)

    def _physics(self, data, cmd_new, cmd_old, delay):
        """One policy step. The new command takes effect after ``delay`` physics steps."""
        F, sub = self._F, self.cfg.substeps

        def to_cf(cmd):  # [thrust, wx, wy, wz] -> Crazyflow's [wx, wy, wz, thrust], (N, 1, 4)
            return jnp.concatenate([cmd[:, 1:], cmd[:, :1]], axis=-1)[:, None, :]

        if self.cfg.delay_max_substeps == 0:
            return self._sim_step(F.body_rate_control(data, to_cf(cmd_new)), n_steps=sub)
        new, old = to_cf(cmd_new), to_cf(cmd_old)

        def substep(data, i):
            cmd = jnp.where((i >= delay)[:, None, None], new, old)
            return self._sim_step(F.body_rate_control(data, cmd), n_steps=1), None

        return jax.lax.scan(substep, data, jnp.arange(sub))[0]

    def step_core(self, state: EnvState, action) -> tuple[EnvState, jax.Array, dict]:
        """Advance one policy step without resetting. Returns (state, reward, events)."""
        c = self.cfg
        if c.rand_action_noise > 0:
            key, k = jax.random.split(state.key)
            action = action + state.act_noise[:, None] * jax.random.normal(k, action.shape)
            state = state.replace(key=key)
        a = jnp.clip(action, -1.0, 1.0)
        pos0 = state.data.states.pos[:, 0]
        data = self._physics(state.data, scale_action(a), scale_action(state.prev_action), state.delay)
        s = data.states
        pos, quat, vel, ang_vel = s.pos[:, 0], s.quat[:, 0], s.vel[:, 0], s.ang_vel[:, 0]

        passed_g, wrong_g, hit_g = track.gate_crossing(
            pos0[:, None, :], pos[:, None, :], state.gates_pos, state.gates_yaw,
            track.GATE_OPENING / 2 - c.gate_margin, track.GATE_OUTER / 2 + c.gate_margin,
        )  # fmt: skip
        passed = _take_gate(passed_g, state.target)
        crash_gate = (wrong_g | hit_g).any(axis=1)
        finite = jnp.isfinite(pos).all(-1) & jnp.isfinite(vel).all(-1) & jnp.isfinite(quat).all(-1)
        ground = finite & (pos[:, 2] < c.ground_z)
        oob = ~finite | (((pos < self.box_lo) | (pos > self.box_hi)).any(-1) & ~ground)
        terminated = crash_gate | ground | oob
        t = state.t + 1
        truncated = (t >= c.max_steps) & ~terminated

        gate = _take_gate(state.gates_pos, state.target)
        progress = jnp.linalg.norm(pos0 - gate, axis=-1) - jnp.linalg.norm(pos - gate, axis=-1)
        target = jnp.where(passed, (state.target + 1) % self.n_gates, state.target)
        theta, in_fov = camera_view(pos, quat, _take_gate(state.gates_pos, target))
        rate_sq = (ang_vel**2).sum(-1)
        dact_sq = ((a - state.prev_action) ** 2).sum(-1)
        reward = (
            c.w_progress * progress
            + c.w_gate * passed
            - c.w_crash * terminated
            - c.w_rate * rate_sq
            - c.w_dact * dact_sq
            + c.w_cam * (jnp.exp(-c.cam_lambda * theta**4) - c.cam_offset)
            - c.w_speed * jnp.maximum(jnp.linalg.norm(vel, axis=-1) - c.speed_limit, 0.0) ** 2
        )
        reward = jnp.where(finite, reward, -c.w_crash)
        state = state.replace(
            data=data,
            target=target,
            prev_action=a,
            t=t,
            ep_return=state.ep_return + reward,
            ep_gates=state.ep_gates + passed,
        )
        events = dict(
            passed=passed, crash_gate=crash_gate, ground=ground, oob=oob, terminated=terminated,
            truncated=truncated, in_fov=in_fov, speed=jnp.linalg.norm(vel, axis=-1),
        )  # fmt: skip
        return state, reward, events

    def step(self, state: EnvState, action):
        """Step with auto-reset. Returns (state, obs, reward, done, info).

        ``obs`` belongs to the new episode where ``done``; ``info["final_obs"]`` is the last
        observation of the old one (for bootstrapping at ``info["truncated"]``). ``info["ep_*"]``
        hold the finished episode's totals where ``done``.
        """
        state, reward, ev = self.step_core(state, action)
        done = ev["terminated"] | ev["truncated"]
        final_obs = self.obs(state)
        info = ev | dict(final_obs=final_obs, ep_return=state.ep_return, ep_len=state.t, ep_gates=state.ep_gates)
        key, k = jax.random.split(state.key)
        state = self._reset_masked(state.replace(key=key), k, done)
        obs = jnp.where(done[:, None], self.obs(state), final_obs)
        return state, obs, reward, done, info
