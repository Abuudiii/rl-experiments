"""Fast synthetic FPV renderer (JAX ray caster) with the shared renderer interface.

    r = SynthRenderer(n, width=160, height=90, seed=0)
    r.set_gates(slots, gates_pos, gates_yaw)        # per-slot layout (+ a fresh random "look")
    frames = r.render(pos, quat_xyzw, active=None)  # uint8 [n, H, W, 3] RGB
    frames = r.render_trajectories(slots, pos, quat_xyzw)   # [k, T, H, W, 3]
    r.close()

Every pixel casts one ray (``supersample``^2 rays, box filtered) from the camera of `fpv.track.CAMERA`
(at the body origin, along body +x, pitched up, pinhole) against: the six gate frames (four bars of
`GATE_DEPTH` x bar width each around the `GATE_OPENING` square, outer `GATE_OUTER`, lit by a sun),
`N_BLOCKS` boxes standing on the ground outside the track volume, a grainy textured ground plane
z = 0 with soft shadows of the gate frames, and a sky gradient with a sun. The nearest hit wins, so
perspective and occlusion order are exact.

Track-general: the blocks are placed outside `fpv.track.VOLUME_LO` / `VOLUME_HI` grown by
`CLEAR_MARGIN` on every side (all gate centres of every track lie in that volume), so a block never
intersects a gate and never stands between a camera inside the grown volume and a gate.

Each slot has its own random look, redrawn at every `set_gates`: with probability ``unreal_like_p``
one modelled on the Unreal Blocks scene of `fpv.render_airsim` (grey grainy floor, blue sky fading to
a pale horizon, dark grey block stacks and a few small coloured blocks, magenta gates, soft gate
shadows), otherwise random sky / ground / block colours, textures and sun; both with random
exposure, haze and pixel noise. ``randomize=False`` gives a clean fixed Unreal-like look.

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
N_BLOCKS = 12  # boxes standing on the ground around the flying area (hit-map ids 6..)
GATE_DEPTH = 0.05  # thickness of a gate frame along the travel direction [m] (as in Unreal)
# Blocks keep at least this distance [m] horizontally from the track volume: the drone goes out of
# bounds 3 m outside its nominal track (env box_margin), and randomised gates move up to 0.3 m.
CLEAR_MARGIN = 3.3
# Target gate with ``highlight_target``: pure green, as the Unreal renderer's highlight (measured
# there: median RGB (0, 226, 9) on screen); brightness randomised per slot like the magenta gates.
HIGHLIGHT_RGB = (0.0, 1.0, 0.0)
VOL_LO, VOL_HI = np.asarray(track.VOLUME_LO, np.float32), np.asarray(track.VOLUME_HI, np.float32)


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


def _slab(o, d, half):
    """Ray / axis-aligned box (centred at the origin) in a local frame. o: 3 x (n, 1), d: 3 x (n, P),
    half: 3 x (n, 1) -> (t_near (inf if missed), axis of the entry face, sign of d along it)."""
    lo, hi = [], []
    for o_, d_, h_ in zip(o, d, half):
        inv = 1.0 / jnp.where(jnp.abs(d_) > 1e-9, d_, 1e-9)
        t1, t2 = (-h_ - o_) * inv, (h_ - o_) * inv
        lo.append(jnp.minimum(t1, t2))
        hi.append(jnp.maximum(t1, t2))
    t_near = jnp.maximum(jnp.maximum(lo[0], lo[1]), lo[2])
    t_far = jnp.minimum(jnp.minimum(hi[0], hi[1]), hi[2])
    axis = jnp.where((lo[0] >= lo[1]) & (lo[0] >= lo[2]), 0, jnp.where(lo[1] >= lo[2], 1, 2))
    hit = (t_far >= t_near) & (t_near > 0.03)
    return jnp.where(hit, t_near, jnp.inf), axis


def _make_render(width: int, height: int, ss: int):
    dirs_b = jnp.asarray(_pixel_dirs(width, height, ss))
    h_open, h_out = track.GATE_OPENING / 2, track.GATE_OUTER / 2
    bar_c, bar_h = (h_open + h_out) / 2, (h_out - h_open) / 2
    # gate bars in the gate frame (normal, left, up): (centre, half extents)
    bars = [((0.0, 0.0, bar_c), (GATE_DEPTH / 2, h_out, bar_h)), ((0.0, 0.0, -bar_c), (GATE_DEPTH / 2, h_out, bar_h)),
            ((0.0, bar_c, 0.0), (GATE_DEPTH / 2, bar_h, h_open)), ((0.0, -bar_c, 0.0), (GATE_DEPTH / 2, bar_h, h_open))]  # fmt: skip

    def local(centre, yaw, pos, d):
        """Camera position and ray directions in the frame of an object at centre (n, 3), yaw (n,)."""
        c, s = jnp.cos(yaw)[:, None], jnp.sin(yaw)[:, None]
        rel = pos - centre
        o = (rel[:, 0:1] * c + rel[:, 1:2] * s, -rel[:, 0:1] * s + rel[:, 1:2] * c, rel[:, 2:3])
        dl = (d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c, d[..., 2])
        return o, dl, c, s

    def lambert(axis, dl, c, s, sun, amb):
        """Brightness of the entered face: ambient + diffuse sun light; normal = -sign(d) along axis."""
        sx, sy, sz = sun[:, 0:1], sun[:, 1:2], sun[:, 2:3]
        n_dot = jnp.where(axis == 0, -jnp.sign(dl[0]) * (c * sx + s * sy),
                          jnp.where(axis == 1, -jnp.sign(dl[1]) * (-s * sx + c * sy), -jnp.sign(dl[2]) * sz))  # fmt: skip
        return amb[:, None] + (1 - amb[:, None]) * jnp.clip(n_dot, 0, 1)

    def render(pos, quat, gates_pos, gates_yaw, sc, key):
        n = pos.shape[0]
        k_noise, k_grain = jax.random.split(key)
        rot = quat_to_mat(quat)
        d = jnp.einsum("nij,pj->npi", rot, dirs_b)  # (n, P, 3) world, |component along the axis| = 1
        dlen = jnp.linalg.norm(d, axis=-1)
        dz = d[..., 2]
        sun, amb = sc["sun"], sc["amb"]
        # sky: zenith colour fading to the horizon colour, sun disc with a halo
        elev = dz / dlen
        sky = sc["horizon"][:, None] + (sc["sky"] - sc["horizon"])[:, None] * jnp.clip(elev * sc["sky_k"][:, None], 0, 1)[..., None]
        cos_sun = jnp.einsum("npi,ni->np", d, sun) / dlen
        glow = jnp.exp((cos_sun - 1) / 2e-4) + 0.35 * jnp.exp((cos_sun - 1) / 0.02)
        sky = sky + (sc["sun_glow"][:, None] * glow)[..., None]
        # ground plane z = 0: two-colour world texture, grain, soft shadows of the gate frames
        t_g = jnp.where(dz < -1e-6, -pos[:, 2:3] / jnp.minimum(dz, -1e-6), jnp.inf)
        t_gc = jnp.minimum(t_g, 1e4)
        gx, gy = pos[:, 0:1] + t_gc * d[..., 0], pos[:, 1:2] + t_gc * d[..., 1]
        ca, sa, k = jnp.cos(sc["tex_angle"])[:, None], jnp.sin(sc["tex_angle"])[:, None], sc["tex_freq"][:, None]
        px, py = k * (ca * gx + sa * gy), k * (-sa * gx + ca * gy)
        tex = jnp.clip(jnp.sin(px) * jnp.sin(py) * 3.0, -1, 1) * 0.35 + jnp.sin(0.37 * px + 1.3) * jnp.sin(0.23 * py) * 0.15
        tex = tex * sc["tex_amp"][:, None] * jnp.exp(-t_gc * dlen / 12.0)  # fade with distance (no aliasing)
        tex = tex + sc["grain"][:, None] * jax.random.normal(k_grain, tex.shape)
        ground = sc["ground_a"][:, None] + (sc["ground_b"] - sc["ground_a"])[:, None] * (0.5 + tex)[..., None]
        shadow = jnp.zeros_like(t_gc)
        for g in range(N_GATES):  # ray from the ground point towards the sun against the gate plane
            c, s = jnp.cos(gates_yaw[:, g])[:, None], jnp.sin(gates_yaw[:, g])[:, None]
            rx, ry, rz = gx - gates_pos[:, g, 0:1], gy - gates_pos[:, g, 1:2], -gates_pos[:, g, 2:3]
            sn = sun[:, 0:1] * c + sun[:, 1:2] * s
            t = -(rx * c + ry * s) / jnp.where(jnp.abs(sn) > 0.05, sn, 0.05)
            ul = -(rx + t * sun[:, 0:1]) * s + (ry + t * sun[:, 1:2]) * c
            uz = rz + t * sun[:, 2:3]
            soft = 0.04 + 0.08 * jnp.abs(t)  # penumbra grows with the distance to the frame
            m = jnp.maximum(jnp.abs(ul), jnp.abs(uz))
            ring = jax.nn.sigmoid((h_out - m) / soft) * jax.nn.sigmoid((m - h_open) / soft)
            shadow = jnp.maximum(shadow, jnp.where(t > 0, ring, 0))
        ground = ground * (1 - sc["shadow"][:, None] * shadow)[..., None]
        col = jnp.where((dz < -1e-6)[..., None], ground, sky)
        best = t_g
        hit_id = jnp.where(dz < -1e-6, -1, -2)  # -2 sky, -1 ground, 0..5 gates, 6.. blocks
        for g in range(N_GATES):
            o, dl, c, s = local(gates_pos[:, g], gates_yaw[:, g], pos, d)
            for (bc, bh) in bars:
                t, axis = _slab([o[i] - bc[i] for i in range(3)], dl, [jnp.full((n, 1), h) for h in bh])
                hit = t < best
                shade = lambert(axis, dl, c, s, sun, sc["gate_amb"])
                g_col = jnp.where((sc["tgt"] == g)[:, None], sc["hl_col"], sc["gate_col"])
                col = jnp.where(hit[..., None], g_col[:, None] * shade[..., None], col)
                best, hit_id = jnp.where(hit, t, best), jnp.where(hit, g, hit_id)
        for j in range(N_BLOCKS):
            o, dl, c, s = local(sc["b_pos"][:, j], sc["b_yaw"][:, j], pos, d)
            half = [sc["b_half"][:, j, i : i + 1] for i in range(3)]
            t, axis = _slab(o, dl, half)
            hit = t < best
            shade = lambert(axis, dl, c, s, sun, amb)
            # dark seams of a stack of cubes (cell size b_cell) on the faces
            cell = sc["b_cell"][:, j : j + 1]
            seam = jnp.zeros(t.shape, bool)
            for i in range(3):
                q = (o[i] + jnp.minimum(t, 1e4) * dl[i] + half[i]) / cell
                seam |= (jnp.abs(q - jnp.round(q)) < 0.025 / cell) & (axis != i)
            bcol = sc["b_col"][:, j][:, None] * (shade * jnp.where(seam, 0.55, 1.0))[..., None]
            col = jnp.where(hit[..., None], bcol, col)
            best, hit_id = jnp.where(hit, t, best), jnp.where(hit, N_GATES + j, hit_id)
        fog = 1 - jnp.exp(-jnp.minimum(best, 1e4) * dlen / sc["fog"][:, None])
        col = jnp.where((hit_id > -2)[..., None], col + (sc["horizon"][:, None] - col) * fog[..., None], col)
        img = col.reshape(n, height, ss, width, ss, 3).mean(axis=(2, 4))
        img = img * sc["gain"][:, None, None, None] + sc["noise"][:, None, None, None] * jax.random.normal(k_noise, img.shape)
        ids = hit_id.reshape(n, height, ss, width, ss)[:, :, 0, :, 0]
        return jnp.clip(img * 255.0 + 0.5, 0, 255).astype(jnp.uint8), ids

    return jax.jit(render)


def place_blocks(rng, half_xy: np.ndarray, extra: float = 12.0, along: float = 8.0) -> np.ndarray:
    """Ground-plane centres (..., 2) for boxes with horizontal half extents half_xy (..., 2), each
    outside the track volume grown by `CLEAR_MARGIN`: a side of that rectangle is chosen in
    proportion to its length, a point along it (extended by ``along`` m), and the box centre is put
    outward from that side by its circumscribed radius + U(0, ``extra``) m."""
    shape = half_xy.shape[:-1]
    lo, hi = VOL_LO[:2] - CLEAR_MARGIN, VOL_HI[:2] + CLEAR_MARGIN
    w, h = hi - lo
    side = rng.choice(4, shape, p=np.array([w, w, h, h]) / (2 * (w + h)))  # 0 y-low, 1 y-high, 2 x-low, 3 x-high
    out = np.hypot(half_xy[..., 0], half_xy[..., 1]) + rng.uniform(0, extra, shape)
    ax = rng.uniform(lo[0] - along, hi[0] + along, shape)
    ay = rng.uniform(lo[1] - along, hi[1] + along, shape)
    x = np.where(side == 2, lo[0] - out, np.where(side == 3, hi[0] + out, ax))
    y = np.where(side == 0, lo[1] - out, np.where(side == 1, hi[1] + out, ay))
    return np.stack([x, y], -1).astype(np.float32)


def blocks_clear_of_volume(b_pos, b_yaw, b_half) -> np.ndarray:
    """True where a box (any yaw) lies entirely outside the grown track volume (horizontally)."""
    lo, hi = VOL_LO[:2] - CLEAR_MARGIN, VOL_HI[:2] + CLEAR_MARGIN
    c, s = np.abs(np.cos(b_yaw)), np.abs(np.sin(b_yaw))
    ex = c * b_half[..., 0] + s * b_half[..., 1]  # half extents of the axis-aligned bounding box
    ey = s * b_half[..., 0] + c * b_half[..., 1]
    x, y = b_pos[..., 0], b_pos[..., 1]
    return (x + ex < lo[0]) | (x - ex > hi[0]) | (y + ey < lo[1]) | (y - ey > hi[1])


class SynthRenderer:
    def __init__(self, n: int, width: int = 160, height: int = 90, seed: int = 0, supersample: int = 2,
                 randomize: bool = True, chunk: int = 128, unreal_like_p: float = 0.7, highlight_target: bool = False,
                 highlight_rgb=HIGHLIGHT_RGB):  # fmt: skip
        """``highlight_target``: draw the gate given as ``target`` to `render` / `render_trajectories` in
        ``highlight_rgb`` instead of magenta (the next gate is marked, as lit race gates do)."""
        self.highlight_rgb = np.asarray(highlight_rgb, np.float32)
        self.n, self.width, self.height, self.randomize = n, width, height, randomize
        self.highlight_target = bool(highlight_target)
        self.unreal_like_p = unreal_like_p
        self._rng = np.random.default_rng(seed)
        self._key = jax.random.key(seed)
        self._fn = _make_render(width, height, supersample)
        self._chunk = chunk
        self.gates_pos = np.tile(track.circuit_arrays()[0], (n, 1, 1))
        self.gates_yaw = np.tile(track.circuit_arrays()[1], (n, 1))
        self.scene = self._draw_scene(n)

    # region look
    @staticmethod
    def _blocks_unreal(m: int, r) -> dict:
        """Dark grey stacks of cubes (cell 0.9-1.5 m) and a few small coloured blocks."""
        f, B = np.float32, N_BLOCKS
        u = lambda lo, hi, *shape: r.uniform(lo, hi, (m, *shape)).astype(f)  # noqa: E731
        small = r.random((m, B)) < 0.3
        cell = u(0.9, 1.5, B)
        big = np.stack([cell * r.integers(1, 5, (m, B)), cell * r.integers(1, 5, (m, B)),
                        cell * r.integers(1, 4, (m, B))], -1) / 2  # fmt: skip
        little = u(0.15, 0.45, B, 3)
        half = np.where(small[..., None], little, big).astype(f)
        grey = u(0.06, 0.22, B, 1) + u(-0.015, 0.015, B, 3)
        bright = u(0.1, 0.95, B, 3)
        green = (bright[..., 1] > 0.5) & (np.maximum(bright[..., 0], bright[..., 2]) < 0.45)  # not the target's green
        bright[..., 0] = np.where(green, 0.6, bright[..., 0])
        mag = (bright[..., 0] > 0.5) & (bright[..., 2] > 0.5) & (bright[..., 1] < 0.45)  # nor magenta
        bright[..., 1] = np.where(mag, 0.6, bright[..., 1])
        return dict(b_half=half, b_cell=np.where(small, 1e3, cell).astype(f),
                    b_col=np.where(small[..., None], bright, grey).astype(f))  # fmt: skip

    @staticmethod
    def _blocks_random(m: int, r) -> dict:
        """Boxes of any proportion (poles, panels, cubes) and colour, plain or with seams."""
        f, B = np.float32, N_BLOCKS
        u = lambda lo, hi, *shape: r.uniform(lo, hi, (m, *shape)).astype(f)  # noqa: E731
        half = np.exp(u(np.log(0.1), np.log(3.0), B, 3)).astype(f)
        col = u(0.05, 0.95, B, 3)
        mag = (col[..., 0] > 0.5) & (col[..., 2] > 0.5) & (col[..., 1] < 0.45)  # never the gate colours:
        col[..., 1] = np.where(mag, 0.6, col[..., 1])  # magenta ...
        green = (col[..., 1] > 0.5) & (np.maximum(col[..., 0], col[..., 2]) < 0.45)  # ... or the target's green
        col[..., 0] = np.where(green, 0.6, col[..., 0])
        cell = np.where(r.random((m, B)) < 0.4, u(0.5, 2.0, B), 1e3).astype(f)
        return dict(b_half=half, b_cell=cell, b_col=col)

    def _draw_scene(self, m: int) -> dict:
        r, f = self._rng, np.float32
        u = lambda lo, hi, *shape: r.uniform(lo, hi, (m, *shape)).astype(f)  # noqa: E731
        sun_az, sun_el = u(-math.pi, math.pi), u(0.6, 1.2)
        sun = np.stack([np.cos(sun_el) * np.cos(sun_az), np.cos(sun_el) * np.sin(sun_az), np.sin(sun_el)], -1)
        if not self.randomize:
            rng0 = np.random.default_rng(0)  # the same blocks in every slot
            blocks = {k: np.repeat(v, m, 0) for k, v in self._blocks_unreal(1, rng0).items()}
            yaw = np.repeat(rng0.uniform(-math.pi, math.pi, (1, N_BLOCKS)).astype(f), m, 0)
            xy = np.repeat(place_blocks(rng0, blocks["b_half"][:1, :, :2]), m, 0)
            return dict(
                sky=np.tile(f([0.12, 0.3, 0.7]), (m, 1)), horizon=np.tile(f([0.72, 0.8, 0.88]), (m, 1)),
                sky_k=np.full(m, 2.0, f), sun=np.tile(f([0.45, -0.3, 0.84]), (m, 1)), sun_glow=np.full(m, 0.8, f),
                ground_a=np.tile(f([0.36, 0.35, 0.34]), (m, 1)), ground_b=np.tile(f([0.46, 0.45, 0.44]), (m, 1)),
                tex_angle=np.zeros(m, f), tex_freq=np.full(m, 2.0, f), tex_amp=np.full(m, 0.3, f),
                grain=np.zeros(m, f), shadow=np.full(m, 0.4, f), amb=np.full(m, 0.6, f), gate_amb=np.full(m, 0.75, f),
                gate_col=np.tile(f([1, 0, 0.85]), (m, 1)), hl_col=np.tile(self.highlight_rgb, (m, 1)),
                b_pos=np.concatenate([xy, blocks["b_half"][..., 2:3]], -1), b_yaw=yaw, **blocks,
                fog=np.full(m, 80.0, f), gain=np.ones(m, f), noise=np.zeros(m, f),
            )  # fmt: skip
        # random look
        sky = np.clip(f([0.4, 0.6, 0.9]) + u(-0.3, 0.1, 3), 0, 1)
        base = u(0.15, 0.6, 1) + u(-0.12, 0.12, 3)
        scene = dict(
            sky=sky, horizon=np.clip(sky + u(0.1, 0.45, 1), 0, 1), sky_k=u(1.0, 4.0),
            sun=sun.astype(f), sun_glow=u(0.0, 1.0) * (r.random(m) < 0.6),
            ground_a=np.clip(base, 0.03, 1), ground_b=np.clip(base + u(-0.2, 0.2, 3), 0.03, 1),
            tex_angle=u(0, math.pi), tex_freq=u(0.8, 6.0), tex_amp=u(0.2, 1.0), grain=u(0.0, 0.25),
            shadow=u(0.0, 0.6), amb=u(0.4, 0.9), gate_amb=u(0.4, 0.9),
            # magenta of varying brightness and blue share (the Unreal gates are (1, 0, 0.8) under scene lighting)
            gate_col=np.clip(f([1, 0, 1]) * u(0.6, 1.0, 1) * np.stack([u(0.85, 1), u(0, 1), u(0.6, 1)], -1)
                             + np.abs(u(-0.12, 0.12, 3)) * f([-1, 1, -1]), 0, 1),
            fog=u(25.0, 120.0), gain=u(0.8, 1.15), noise=u(0.0, 0.03),
            **self._blocks_random(m, r),
        )  # fmt: skip
        scene["hl_col"] = np.clip(self.highlight_rgb * u(0.6, 1.0, 1) + np.abs(u(-0.12, 0.12, 3)) * f([1, -1, 1]), 0, 1)
        # Unreal-like slots (see the module docstring)
        ue = r.random(m) < self.unreal_like_p
        grey = u(0.3, 0.5, 1)
        like = dict(
            sky=np.clip(f([0.12, 0.3, 0.7]) + u(-0.06, 0.06, 3), 0, 1), sky_k=u(1.5, 3.0), sun_glow=u(0.5, 1.0),
            ground_a=grey + u(-0.02, 0.02, 3), ground_b=grey + u(0.04, 0.12, 1) + u(-0.015, 0.015, 3),
            tex_amp=u(0.1, 0.5), grain=u(0.1, 0.3), shadow=u(0.25, 0.55), amb=u(0.45, 0.7), gate_amb=u(0.7, 0.9),
            gate_col=np.stack([u(0.85, 1.0), u(0.0, 0.1), u(0.7, 0.9)], -1),
            hl_col=np.clip(self.highlight_rgb * u(0.8, 1.0, 1) + u(0.0, 0.05, 3) * f([1, 0, 1]), 0, 1),
            **self._blocks_unreal(m, r),
        )  # fmt: skip
        like["horizon"] = np.clip(f([0.72, 0.8, 0.88]) + u(-0.08, 0.08, 1), 0, 1)
        for k, v in like.items():
            scene[k] = np.where(ue.reshape(-1, *[1] * (v.ndim - 1)), v, scene[k]).astype(f)
        scene["b_yaw"] = u(-math.pi, math.pi, N_BLOCKS)
        xy = place_blocks(r, scene["b_half"][..., :2])
        scene["b_pos"] = np.concatenate([xy, scene["b_half"][..., 2:3]], -1).astype(f)  # standing on the ground
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

    def render_device(self, pos, quat, target=None):
        """`render` without the copy to the host: a uint8 jax array on the GPU."""
        return self._chunked(pos, quat, np.arange(self.n), target)[0]

    def _chunked(self, pos, quat, slots, target=None):
        """Render pose i with the scene of slots[i] (and target gate target[i]), ``chunk`` poses per
        call (bounded memory; the last call is padded so that every call has one compiled shape)."""
        pos, quat, m, c = np.asarray(pos, np.float32), np.asarray(quat, np.float32), len(slots), self._chunk
        tgt = np.full(m, -1, np.int32)
        if self.highlight_target:
            if target is None:
                raise ValueError("highlight_target renderer: pass the target gate of every pose")
            tgt = np.asarray(target, np.int32).reshape(m)
        sc = lambda idx, s: dict({k: v[s] for k, v in self.scene.items()}, tgt=tgt[idx])  # noqa: E731
        if m <= c:
            return self._call(pos, quat, self.gates_pos[slots], self.gates_yaw[slots], sc(np.arange(m), slots))
        imgs, ids = [], []
        for i in range(0, m, c):
            j = min(i + c, m)
            idx = np.r_[i:j, np.full(c - (j - i), j - 1)]
            s = slots[idx]
            a, b = self._call(pos[idx], quat[idx], self.gates_pos[s], self.gates_yaw[s], sc(idx, s))
            imgs.append(a[: j - i])
            ids.append(b[: j - i])
        return jnp.concatenate(imgs), jnp.concatenate(ids)

    def render(self, pos, quat_xyzw, active=None, target=None):
        return np.asarray(self.render_device(pos, quat_xyzw, target))

    def render_ids(self, pos, quat_xyzw, target=None):
        """Frames and the per-pixel hit map (-2 sky, -1 ground, 0..5 gate index, >= 6 blocks)."""
        img, ids = self._chunked(pos, quat_xyzw, np.arange(self.n), target)
        return np.asarray(img), np.asarray(ids)

    def render_trajectories(self, slots, pos, quat_xyzw, target=None):
        """target: [k, T] gate index per pose (used with ``highlight_target``)."""
        slots = np.atleast_1d(np.asarray(slots))
        pos, quat = np.asarray(pos, np.float32), np.asarray(quat_xyzw, np.float32)
        k, t = pos.shape[:2]
        sl = np.repeat(slots, t)
        out = np.asarray(self._chunked(pos.reshape(-1, 3), quat.reshape(-1, 4), sl,
                                       None if target is None else np.asarray(target).reshape(-1))[0])  # fmt: skip
        return out.reshape(k, t, self.height, self.width, 3)

    def close(self):
        pass
