"""Simulated 6-axis IMU (gyroscope + accelerometer) at the 50 Hz policy rate, batched JAX.

From Crazyflow's true state at the start and end of a policy step (dt = 20 ms):

    gyro  = (1 + s_g) * w_true + b_g + n_g        w_true = mean body rate over the step = log(R0^T R1) / dt
                                                   (the delta angle of the step; rates change within a step:
                                                   (w0 + w1) / 2 is off by up to 1.7 deg per step, see checks)
    accel = f_true + b_a + n_a                     f_true = specific force in the body frame, mean over the step:
                                                   R_mid^T ((v1 - v0) / dt + g e_z),  R_mid = body attitude at mid-step
    b_g, b_a: initial offset per episode, then a random walk; n: white noise per sample; s_g: gyro scale error.

The sample is the step mean (a flight controller low-pass filters and decimates its kHz IMU stream), so
integrating ``accel`` rotated by the mid-step attitude reproduces the velocity change exactly; this is
checked against the simulator in `scripts/check_estimator.py`.

Defaults are those of a Bosch BMI270 (the IMU of most current FPV flight controllers), datasheet
BST-BMI270-DS000 rev 1.3, section 1 "Specifications":
  gyro  zero-rate offset +-0.5 dps (over life), offset drift 0.015 dps/K, noise density 0.007 dps/sqrt(Hz)
        (0.07 dps rms at 74.6 Hz bandwidth), sensitivity error after CRT trimming 0.4 %;
  accel zero-g offset +-20 mg, output noise 0.16 mg/sqrt(Hz).
The sensor's own white noise is negligible at 50 Hz (0.035 dps and 0.008 m/s^2 rms at 25 Hz bandwidth); on a
flying quadrotor propeller vibration dominates, so the white-noise defaults are ASSUMED in-flight values
(0.5 dps, 0.3 m/s^2 rms per filtered 50 Hz sample), 7x and 40x the datasheet figures. The random walks
are assumed (a few K of temperature change per minute times the datasheet drift). ``scale`` multiplies
every noise and bias term (0 = perfect IMU); per-episode values are drawn by the caller.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import struct

from fpv.env import G, quat_mul, quat_to_mat

DEG = math.pi / 180


@dataclasses.dataclass(frozen=True)
class IMUConfig:
    gyro_noise: float = 0.5 * DEG  # rad/s rms per 50 Hz sample (vibration, assumed)
    gyro_bias0: float = 0.5 * DEG  # rad/s, initial bias uniform +- per axis (datasheet zero-rate offset)
    gyro_bias_rw: float = 0.05 * DEG  # rad/s per sqrt(s) (assumed)
    gyro_scale_err: float = 0.004  # +- relative per axis (datasheet, after CRT)
    acc_noise: float = 0.3  # m/s^2 rms per sample (vibration, assumed)
    acc_bias0: float = 0.020 * G  # m/s^2, uniform +- per axis (datasheet zero-g offset 20 mg)
    acc_bias_rw: float = 0.01  # m/s^2 per sqrt(s) (assumed)


@struct.dataclass
class IMUState:
    gyro_bias: jax.Array  # (N, 3) rad/s
    acc_bias: jax.Array  # (N, 3) m/s^2
    gyro_scale: jax.Array  # (N, 3) relative error
    scale: jax.Array  # (N,) noise scale of the episode


def imu_init(key, scale, cfg: IMUConfig = IMUConfig()) -> IMUState:
    """New per-episode IMU errors; ``scale`` (N,) multiplies every error term (0 = perfect)."""
    n = scale.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)

    def u(k, r):
        return jax.random.uniform(k, (n, 3), minval=-1.0, maxval=1.0) * r * scale[:, None]

    return IMUState(u(k1, cfg.gyro_bias0), u(k2, cfg.acc_bias0), u(k3, cfg.gyro_scale_err), scale)


def imu_reset_masked(state: IMUState, key, mask, scale) -> IMUState:
    new = imu_init(key, scale)
    return jax.tree.map(lambda a, b: jnp.where(mask.reshape(-1, *[1] * (a.ndim - 1)), a, b), new, state)


def mid_quat(q0, q1):
    """Normalised mean of two xyzw quaternions (sign-aligned): the attitude at mid-step."""
    q1 = jnp.where(((q0 * q1).sum(-1, keepdims=True) < 0), -q1, q1)
    q = q0 + q1
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def quat_log_delta(quat0, quat1):
    """Rotation vector (N, 3) of R0^T R1, body frame, from xyzw quaternions."""
    x0, y0, z0, w0 = quat0[..., 0], quat0[..., 1], quat0[..., 2], quat0[..., 3]
    conj = jnp.stack([-x0, -y0, -z0, w0], -1)
    dq = quat_mul(conj, quat1)
    dq = jnp.where(dq[..., 3:] < 0, -dq, dq)
    v = dq[..., :3]
    s = jnp.sqrt((v * v).sum(-1, keepdims=True) + 1e-18)
    return 2 * jnp.arctan2(s, dq[..., 3:]) * v / s


def true_imu(quat0, quat1, vel0, vel1, ang_vel0, ang_vel1, dt):
    """Error-free step means: body rate (N, 3) and body specific force (N, 3). ``ang_vel*`` are unused (kept
    for the call signature; the step-mean rate follows exactly from the attitude change)."""
    gyro = quat_log_delta(quat0, quat1) / dt
    f_world = (vel1 - vel0) / dt + jnp.asarray([0.0, 0.0, G])
    rot = quat_to_mat(mid_quat(quat0, quat1))  # body -> world
    acc = jnp.einsum("nji,nj->ni", rot, f_world)
    return gyro, acc


def imu_measure(state: IMUState, key, quat0, quat1, vel0, vel1, ang_vel0, ang_vel1, dt,
                cfg: IMUConfig = IMUConfig()):  # fmt: skip
    """One 50 Hz sample. Returns (state with the random-walked biases, gyro (N, 3) rad/s, accel (N, 3) m/s^2)."""
    gyro_t, acc_t = true_imu(quat0, quat1, vel0, vel1, ang_vel0, ang_vel1, dt)
    s = state.scale[:, None]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    sq = math.sqrt(dt)
    gb = state.gyro_bias + cfg.gyro_bias_rw * sq * s * jax.random.normal(k1, gyro_t.shape)
    ab = state.acc_bias + cfg.acc_bias_rw * sq * s * jax.random.normal(k2, acc_t.shape)
    gyro = (1 + state.gyro_scale) * gyro_t + gb + cfg.gyro_noise * s * jax.random.normal(k3, gyro_t.shape)
    acc = acc_t + ab + cfg.acc_noise * s * jax.random.normal(k4, acc_t.shape)
    return state.replace(gyro_bias=gb, acc_bias=ab), gyro, acc
