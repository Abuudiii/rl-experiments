"""Fast synthetic FPV renderer (JAX ray caster) with the shared renderer interface.

    r = SynthRenderer(n, width=160, height=90, seed=0)
    r.set_gates(slots, gates_pos, gates_yaw)        # per-slot layout (+ a fresh random "look")
    frames = r.render(pos, quat_xyzw, active=None)  # uint8 [n, H, W, 3] RGB
    frames = r.render_trajectories(slots, pos, quat_xyzw)   # [k, T, H, W, 3]
    r.close()

Every pixel casts one ray (``supersample``^2 rays, box filtered) from the camera of `fpv.track.CAMERA`
(at the body origin, along body +x, pitched up, pinhole) against: the six gate frames (square rings,
opening `GATE_OPENING`, outer `GATE_OUTER`, magenta, zero thickness), a few coloured background
rectangles outside the circuit (distractors), a textured ground plane z = 0 and a gradient sky. The
nearest hit wins, so perspective and occlusion order are exact. Each slot has its own random look
(sky / ground colours, ground texture, gate brightness, distractors, exposure, pixel noise),
redrawn at every `set_gates`; ``randomize=False`` gives a clean fixed look.

`project_points` is the same pinhole model as a point projection (numpy), used to check the ray
caster and to build image-space labels.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from fpv import track
from fpv.env import CAM_FWD, CAM_LEFT, CAM_UP, quat_to_mat

N_GATES = 6
N_DISTRACTORS = 8
CIRCUIT_CENTRE = np.array([3.25, 2.75], np.float32)


def project_points(pos, quat, points, width: int, height: int):
    """World points (..., P, 3) seen from body pose pos (..., 3), quat xyzw (..., 4).

    Returns (uv (..., P, 2), depth (..., P)): pixel coordinates (integer = pixel centre, u right,
    v down) and distance along the optical axis (<= 0: behind the camera, uv meaningless).
    """
    rot = np.asarray(quat_to_mat(jnp.asarray(quat)))
    rel = np.einsum("...ji,...pj->...pi", rot, np.asarray(points) - np.asarray(pos)[..., None, :])
    fwd, left, up = rel @ CAM_FWD, rel @ CAM_LEFT, rel @ CAM_UP
    tan_h, tan_v = track.camera_half_fov_tan()
    safe = np.where(np.abs(fwd) > 1e-6, fwd, 1e-6)
    u = (-left / (safe * tan_h) + 1) / 2 * width - 0.5
    v = (-up / (safe * tan_v) + 1) / 2 * height - 0.5
    return np.stack([u, v], -1), fwd


def _pixel_dirs(width: int, height: int, ss: int) -> np.ndarray:
    """Ray directions in the body frame, (H*ss * W*ss, 3), unit component along the optical axis."""
    tan_h, tan_v = track.camera_half_fov_tan()
    x = ((np.arange(width * ss) + 0.5) / (width * ss) * 2 - 1) * tan_h
    y = ((np.arange(height * ss) + 0.5) / (height * ss) * 2 - 1) * tan_v
    xx, yy = np.meshgrid(x, y)
    d = CAM_FWD[None, None] - xx[..., None] * CAM_LEFT - yy[..., None] * CAM_UP
    return d.reshape(-1, 3).astype(np.float32)


def _make_render(width: int, height: int, ss: int):
    dirs_b = jnp.asarray(_pixel_dirs(width, height, ss))
    half_open, half_outer = track.GATE_OPENING / 2, track.GATE_OUTER / 2

    def render(pos, quat, gates_pos, gates_yaw, sc, key):
        n = pos.shape[0]
        rot = quat_to_mat(quat)
        d = jnp.einsum("nij,pj->npi", rot, dirs_b)  # (n, P, 3) world, |component along the axis| = 1
        dlen = jnp.linalg.norm(d, axis=-1)
        dz = d[..., 2]
        # background: sky gradient above the horizon, textured ground plane z = 0 below
        elev = dz / dlen
        sky = sc["horizon"][:, None] + (sc["sky"] - sc["horizon"])[:, None] * jnp.clip(elev * 2.5, 0, 1)[..., None]
        t_g = jnp.where(dz < -1e-6, -pos[:, 2:3] / jnp.minimum(dz, -1e-6), jnp.inf)
        t_gc = jnp.minimum(t_g, 1e4)
        gx, gy = pos[:, 0:1] + t_gc * d[..., 0], pos[:, 1:2] + t_gc * d[..., 1]
        ca, sa, k = jnp.cos(sc["tex_angle"])[:, None], jnp.sin(sc["tex_angle"])[:, None], sc["tex_freq"][:, None]
        px, py = k * (ca * gx + sa * gy), k * (-sa * gx + ca * gy)
        tex = jnp.clip(jnp.sin(px) * jnp.sin(py) * 3.0, -1, 1) * 0.35 + jnp.sin(0.37 * px + 1.3) * jnp.sin(0.23 * py) * 0.15
        tex = tex * jnp.exp(-t_gc * dlen / 12.0)  # fade the pattern with distance (no aliasing far away)
        ground = sc["ground_a"][:, None] + (sc["ground_b"] - sc["ground_a"])[:, None] * (0.5 + tex)[..., None]
        col = jnp.where((dz < -1e-6)[..., None], ground, sky)
        best = t_g
        rect_id = jnp.where(dz < -1e-6, -1, -2)  # -2 sky, -1 ground, 0..5 gates, 6.. distractors

        def rect(col, best, rect_id, centre, yaw, colour, lo, hi_l, hi_u, idx):
            c, s = jnp.cos(yaw), jnp.sin(yaw)
            rel = centre - pos  # (n, 3)
            dn = d[..., 0] * c[:, None] + d[..., 1] * s[:, None]
            num = rel[:, 0] * c + rel[:, 1] * s
            t = num[:, None] / jnp.where(jnp.abs(dn) > 1e-6, dn, 1e-6)
            left = t * (-d[..., 0] * s[:, None] + d[..., 1] * c[:, None]) - (-rel[:, 0] * s + rel[:, 1] * c)[:, None]
            up = t * dz - rel[:, 2:3]
            al, au = jnp.abs(left), jnp.abs(up)
            inside = (al < hi_l) & (au < hi_u) & (jnp.maximum(al, au) >= lo)
            hit = inside & (t > 0.03) & (t < best)
            shade = 0.7 + 0.3 * jnp.abs(dn) / dlen
            col = jnp.where(hit[..., None], colour[:, None] * shade[..., None], col)
            return col, jnp.where(hit, t, best), jnp.where(hit, idx, rect_id)

        for g in range(N_GATES):
            col, best, rect_id = rect(col, best, rect_id, gates_pos[:, g], gates_yaw[:, g], sc["gate_col"],
                                      half_open, half_outer, half_outer, g)  # fmt: skip
        for j in range(N_DISTRACTORS):
            col, best, rect_id = rect(col, best, rect_id, sc["d_pos"][:, j], sc["d_yaw"][:, j], sc["d_col"][:, j],
                                      0.0, sc["d_size"][:, j, 0:1], sc["d_size"][:, j, 1:2], N_GATES + j)  # fmt: skip
        fog = 1 - jnp.exp(-jnp.minimum(best, 1e4) * dlen / sc["fog"][:, None])
        col = jnp.where((rect_id > -2)[..., None], col + (sc["horizon"][:, None] - col) * fog[..., None], col)
        img = col.reshape(n, height, ss, width, ss, 3).mean(axis=(2, 4))
        img = img * sc["gain"][:, None, None, None] + sc["noise"][:, None, None, None] * jax.random.normal(key, img.shape)
        ids = rect_id.reshape(n, height, ss, width, ss)[:, :, 0, :, 0]
        return jnp.clip(img * 255.0 + 0.5, 0, 255).astype(jnp.uint8), ids

    return jax.jit(render)


class SynthRenderer:
    def __init__(self, n: int, width: int = 160, height: int = 90, seed: int = 0, supersample: int = 2,
                 randomize: bool = True, chunk: int = 512, unreal_like_p: float = 0.5):  # fmt: skip
        self.n, self.width, self.height, self.randomize = n, width, height, randomize
        self.unreal_like_p = unreal_like_p
        self._rng = np.random.default_rng(seed)
        self._key = jax.random.key(seed)
        self._fn = _make_render(width, height, supersample)
        self._chunk = chunk
        self.gates_pos = np.tile(track.circuit_arrays()[0], (n, 1, 1))
        self.gates_yaw = np.tile(track.circuit_arrays()[1], (n, 1))
        self.scene = self._draw_scene(n)

    # region look
    def _draw_scene(self, m: int) -> dict:
        r, f = self._rng, np.float32
        if not self.randomize:
            return dict(
                sky=np.tile(f([0.35, 0.55, 0.9]), (m, 1)), horizon=np.tile(f([0.75, 0.82, 0.9]), (m, 1)),
                ground_a=np.tile(f([0.3, 0.42, 0.25]), (m, 1)), ground_b=np.tile(f([0.42, 0.5, 0.33]), (m, 1)),
                tex_angle=np.zeros(m, f), tex_freq=np.full(m, 2.0, f), gate_col=np.tile(f([1, 0, 1]), (m, 1)),
                d_pos=np.full((m, N_DISTRACTORS, 3), 1e3, f), d_yaw=np.zeros((m, N_DISTRACTORS), f),
                d_size=np.zeros((m, N_DISTRACTORS, 2), f), d_col=np.zeros((m, N_DISTRACTORS, 3), f),
                fog=np.full(m, 80.0, f), gain=np.ones(m, f), noise=np.zeros(m, f),
            )  # fmt: skip
        u = lambda lo, hi, *shape: r.uniform(lo, hi, (m, *shape)).astype(f)  # noqa: E731
        sky = np.clip(f([0.4, 0.6, 0.9]) + u(-0.25, 0.1, 3), 0, 1)
        base = u(0.15, 0.6, 1) + u(-0.12, 0.12, 3)
        ang, rad = u(0, 2 * math.pi, N_DISTRACTORS), u(5.5, 14.0, N_DISTRACTORS)
        size = u(0.3, 2.5, N_DISTRACTORS, 2)
        d_pos = np.stack([CIRCUIT_CENTRE[0] + rad * np.cos(ang), CIRCUIT_CENTRE[1] + rad * np.sin(ang),
                          size[..., 1] * u(0.3, 1.6, N_DISTRACTORS)], -1)  # fmt: skip
        d_col = u(0.05, 0.95, N_DISTRACTORS, 3)
        # keep distractors away from the gate colour: never "red and blue high, green low"
        mag = (d_col[..., 0] > 0.5) & (d_col[..., 2] > 0.5) & (d_col[..., 1] < 0.45)
        d_col[..., 1] = np.where(mag, 0.6, d_col[..., 1])
        scene = dict(
            sky=sky, horizon=np.clip(sky + u(0.1, 0.4, 1), 0, 1),
            ground_a=np.clip(base, 0.03, 1), ground_b=np.clip(base + u(-0.2, 0.2, 3), 0.03, 1),
            tex_angle=u(0, math.pi), tex_freq=u(0.8, 6.0),
            # magenta of varying brightness and blue share (the Unreal gates are (1, 0, 0.8) under scene lighting)
            gate_col=np.clip(f([1, 0, 1]) * u(0.5, 1.0, 1) * np.stack([u(0.85, 1), u(0, 1), u(0.6, 1)], -1)
                             + np.abs(u(-0.12, 0.12, 3)) * f([-1, 1, -1]), 0, 1),
            d_pos=d_pos, d_yaw=u(-math.pi, math.pi, N_DISTRACTORS), d_size=size / 2, d_col=d_col,
            fog=u(25.0, 120.0), gain=u(0.8, 1.15), noise=u(0.0, 0.03),
        )  # fmt: skip
        # Half of the slots look like the Unreal scene: grey floor, blue sky, dark grey blocks standing
        # on the ground around the circuit, bright magenta gates
        ue = r.random(m) < self.unreal_like_p
        grey = u(0.25, 0.6, 1)
        half = np.stack([u(0.5, 3.0, N_DISTRACTORS), u(0.4, 2.5, N_DISTRACTORS)], -1)
        like = dict(
            sky=np.clip(f([0.3, 0.5, 0.9]) + u(-0.08, 0.08, 3), 0, 1),
            ground_a=grey + u(-0.02, 0.02, 3), ground_b=grey + u(-0.06, 0.06, 1) + u(-0.02, 0.02, 3),
            gate_col=np.stack([u(0.8, 1.0), u(0.0, 0.15), u(0.6, 0.9)], -1),
            d_col=u(0.08, 0.3, N_DISTRACTORS, 1) + u(-0.02, 0.02, N_DISTRACTORS, 3), d_size=half,
            d_pos=np.concatenate([d_pos[..., :2], half[..., 1:2]], -1),
        )  # fmt: skip
        like["horizon"] = np.clip(like["sky"] + u(0.2, 0.45, 1), 0, 1)
        for k, v in like.items():
            scene[k] = np.where(ue.reshape(-1, *[1] * (v.ndim - 1)), v, scene[k]).astype(f)
        return scene

    # region interface
    def set_gates(self, slots, gates_pos, gates_yaw):
        slots = np.atleast_1d(np.asarray(slots))
        if slots.dtype == bool:
            slots = np.flatnonzero(slots)
        self.gates_pos[slots] = np.asarray(gates_pos, np.float32).reshape(len(slots), N_GATES, 3)
        self.gates_yaw[slots] = np.asarray(gates_yaw, np.float32).reshape(len(slots), N_GATES)
        new = self._draw_scene(len(slots))
        for k, v in new.items():
            self.scene[k][slots] = v

    def _call(self, pos, quat, gates_pos, gates_yaw, scene):
        self._key, k = jax.random.split(self._key)
        return self._fn(jnp.asarray(pos, jnp.float32), jnp.asarray(quat, jnp.float32), jnp.asarray(gates_pos),
                        jnp.asarray(gates_yaw), {a: jnp.asarray(b) for a, b in scene.items()}, k)  # fmt: skip

    def render_device(self, pos, quat):
        """`render` without the copy to the host: a uint8 jax array on the GPU."""
        return self._call(pos, quat, self.gates_pos, self.gates_yaw, self.scene)[0]

    def render(self, pos, quat_xyzw, active=None):
        return np.asarray(self.render_device(pos, quat_xyzw))

    def render_ids(self, pos, quat_xyzw):
        """Frames and the per-pixel hit map (-2 sky, -1 ground, 0..5 gate index, >= 6 distractor)."""
        img, ids = self._call(pos, quat_xyzw, self.gates_pos, self.gates_yaw, self.scene)
        return np.asarray(img), np.asarray(ids)

    def render_trajectories(self, slots, pos, quat_xyzw):
        slots = np.atleast_1d(np.asarray(slots))
        pos, quat = np.asarray(pos, np.float32), np.asarray(quat_xyzw, np.float32)
        k, t = pos.shape[:2]
        sl = np.repeat(slots, t)
        pos, quat = pos.reshape(-1, 3), quat.reshape(-1, 4)
        out = np.empty((k * t, self.height, self.width, 3), np.uint8)
        c = self._chunk
        for i in range(0, k * t, c):
            j = min(i + c, k * t)
            idx = np.r_[i:j, np.full(c - (j - i), j - 1)]  # pad the last chunk: one compiled shape
            s = sl[idx]
            img, _ = self._call(pos[idx], quat[idx], self.gates_pos[s], self.gates_yaw[s],
                                {a: b[s] for a, b in self.scene.items()})  # fmt: skip
            out[i:j] = np.asarray(img)[: j - i]
        return out.reshape(k, t, self.height, self.width, 3)

    def close(self):
        pass
