"""True gate-corner labels in the FPV image, and the pose / look sources of the detector dataset.

Image convention (the same as `fpv.render_synth.project_points` and the ray caster): pixel
coordinates (u, v), u right, v down, integer values = pixel centres, so the image spans
u in [-0.5, W - 0.5], v in [-0.5, H - 0.5]. The FPV camera is a pinhole with square pixels:

    u = cx + f * x / z,  v = cy + f * y / z,  f = (W / 2) / tan(hfov / 2),  cx = (W - 1) / 2,  cy = (H - 1) / 2

with camera coordinates x right (= -body-camera left), y down (= -camera up), z along the optical
axis (`fpv.env.CAM_FWD`, pitched up 20 deg from body +x). 320x180: f = 112.03 px, cx = 159.5, cy = 89.5.

Corner order (contract): the 4 corners of the gate OPENING, clockwise in the image (v down) starting
from the corner whose direction from the quad's centroid is nearest to image top-left (-135 deg);
independent of the side the gate is seen from. The start corner flips at about 45 deg of image roll
of the gate (inherent to an image-order rule for a square).

Visibility (contract): a gate is VISIBLE if all four opening corners are in front of the camera
(depth > `MIN_DEPTH`), at least 2 project inside the image, and its centre is within `MAX_DIST` m of
the camera. Occlusion by other gates is ignored. DONT-CARE gates are seen in part but not visible by
that rule (exactly one corner inside, beyond `MAX_DIST`, or a corner behind the camera with the
centre in front): neither a target nor a false detection.

Works on numpy and jax arrays alike (``xp`` follows the input type); pose sources below use JAX on
the GPU.
"""

from __future__ import annotations

import math

import numpy as np

from fpv import track
from fpv.env import CAM_FWD, CAM_LEFT, CAM_UP

W, H = 320, 180
MAX_DIST = 12.0  # m, camera to gate centre
MIN_DEPTH = 0.1  # m, every corner in front of the camera by at least this
N_GATES = 6


def _xp(a):
    return track._xp(a)


def intrinsics(width: int = W, height: int = H) -> tuple[float, float, float]:
    """(f, cx, cy) in pixels for the FPV camera at this resolution (square pixels)."""
    tan_h, _ = track.camera_half_fov_tan()
    return width / 2 / tan_h, (width - 1) / 2, (height - 1) / 2


def quat_to_mat(q):
    """xyzw quaternion (..., 4) -> body-to-world rotation (..., 3, 3) (as `fpv.env.quat_to_mat`, any xp)."""
    xp = _xp(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack(
        [
            xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


# rows: camera x (right), y (down), z (optical axis) in the body frame
_BODY_TO_CAM = np.stack([-CAM_LEFT, -CAM_UP, CAM_FWD]).astype(np.float32)


def to_camera(pos, quat, points):
    """World points (..., P, 3) -> camera coordinates (..., P, 3) (x right, y down, z forward)."""
    xp = _xp(points)
    rot = quat_to_mat(quat)  # body -> world
    rel = points - pos[..., None, :]
    body = xp.einsum("...ji,...pj->...pi", rot, rel)
    return body @ xp.asarray(_BODY_TO_CAM).T


def project(cam, width: int = W, height: int = H):
    """Camera points (..., 3) -> pixels (..., 2) (meaningless where z <= 0)."""
    xp = _xp(cam)
    f, cx, cy = intrinsics(width, height)
    z = xp.where(xp.abs(cam[..., 2]) > 1e-6, cam[..., 2], 1e-6)
    return xp.stack([cx + f * cam[..., 0] / z, cy + f * cam[..., 1] / z], -1)


def canonical_order(uv):
    """Reorder quads (..., 4, 2) (corners in either cyclic order) into the contract's image order."""
    xp = _xp(uv)
    c = uv.mean(-2, keepdims=True)
    ang = xp.arctan2(uv[..., 1] - c[..., 1], uv[..., 0] - c[..., 0])  # v down: increasing = clockwise
    srt = xp.argsort(ang, axis=-1)
    ang_s = xp.take_along_axis(ang, srt, -1)
    d = xp.abs((ang_s + 0.75 * math.pi + math.pi) % (2 * math.pi) - math.pi)  # distance to -135 deg
    start = xp.argmin(d, axis=-1)[..., None]
    idx = xp.take_along_axis(srt, (start + xp.arange(4)) % 4, -1)
    return xp.take_along_axis(uv, idx[..., None], -2)


def in_image(uv, width: int = W, height: int = H):
    return (uv[..., 0] >= -0.5) & (uv[..., 0] <= width - 0.5) & (uv[..., 1] >= -0.5) & (uv[..., 1] <= height - 0.5)


def gate_labels(pos, quat, gates_pos, gates_yaw, width: int = W, height: int = H, max_dist: float = MAX_DIST):
    """Labels of every gate for poses pos (N, 3), quat xyzw (N, 4), layouts gates_pos (N, G, 3), gates_yaw (N, G).

    Returns dict (leading N, G):
        corners (N, G, 4, 2) opening corners in contract image order (pixels),
        centre (N, G, 2) projected gate centre (pixels), centre_cam (N, G, 3) camera coordinates [m],
        dist (N, G) camera to gate centre [m], n_inside (N, G) corners inside the image,
        visible, dontcare (N, G) bool (module docstring), side_px (N, G) sqrt of the quad's area.
    """
    xp = _xp(gates_pos)
    n, g = gates_pos.shape[:2]
    cw = track.gate_corners(gates_pos, gates_yaw)  # (N, G, 4, 3)
    pts = xp.concatenate([cw.reshape(n, g * 4, 3), gates_pos], 1)  # corners then centres
    cam = to_camera(pos, quat, pts)
    uv = project(cam, width, height)
    c_uv, c_cam = uv[:, : g * 4].reshape(n, g, 4, 2), cam[:, : g * 4].reshape(n, g, 4, 3)
    ctr_uv, ctr_cam = uv[:, g * 4 :], cam[:, g * 4 :]
    front = (c_cam[..., 2] > MIN_DEPTH).all(-1)
    inside = in_image(c_uv, width, height) & (c_cam[..., 2] > MIN_DEPTH)
    n_in = inside.sum(-1)
    dist = xp.sqrt((ctr_cam**2).sum(-1))
    visible = front & (n_in >= 2) & (dist <= max_dist)
    seen = (n_in >= 1) | (front & (ctr_cam[..., 2] > 0) & in_image(ctr_uv, width, height))
    dontcare = seen & ~visible & (ctr_cam[..., 2] > 0)
    corners = canonical_order(c_uv)
    x, y = corners[..., 0], corners[..., 1]
    area = 0.5 * xp.abs((x * xp.roll(y, -1, -1) - xp.roll(x, -1, -1) * y).sum(-1))
    return dict(corners=corners, centre=ctr_uv, centre_cam=ctr_cam, dist=dist, n_inside=n_in, visible=visible,
                dontcare=dontcare, side_px=xp.sqrt(area))  # fmt: skip

def hidden_fraction(pos, quat, gates_pos, gates_yaw, ids, side_px, width: int = W, height: int = H):
    """Fraction (N, G) of each gate's frame hidden (by a nearer gate), from the renderer's hit map ids
    (N, H, W) (one ray per pixel). Exact where the frame's outer square is entirely in the image
    (expected frame area = outer quad - opening quad); elsewhere 1 if no frame pixel is seen, else 0."""
    xp = _xp(gates_pos)
    n, g = gates_pos.shape[:2]
    cw = track.gate_corners(gates_pos, gates_yaw, track.GATE_OUTER)
    cam = to_camera(pos, quat, cw.reshape(n, g * 4, 3)).reshape(n, g, 4, 3)
    uv = project(cam, width, height)
    full = (cam[..., 2] > MIN_DEPTH).all(-1) & in_image(uv, width, height).all(-1)
    x, y = uv[..., 0], uv[..., 1]
    area = 0.5 * xp.abs((x * xp.roll(y, -1, -1) - xp.roll(x, -1, -1) * y).sum(-1)) - side_px**2
    px = (ids[..., None] == xp.arange(g)).sum((1, 2))
    frac = xp.clip(1 - px / xp.maximum(area, 1.0), 0.0, 1.0)
    return xp.where(full & (area >= 20), frac, xp.where(px > 0, 0.0, 1.0))



# region pose sources (JAX, GPU)


def collect_flights(teacher_ckpt, n_envs: int, steps: int, seed: int, scenario=None, wide_spawn_p: float = 0.5,
                    teacher_noise: float = 0.3, burst_p: float = 0.01, burst_mag: float = 0.3, burst_len=(3, 12),
                    log=print) -> dict:  # fmt: skip
    """Teacher flights in Crazyflow with stick noise (std per episode U(0, teacher_noise), bursts as in
    `fpv.dagger`) and wide / off-course starts; records every live step of every environment.

    ``scenario``: None = the teacher's random-track distribution (a new track per episode), or a
    name in `fpv.track.SCENARIOS` (randomised gates around that layout).
    Returns numpy arrays (M = recorded frames): pos (M, 3), quat (M, 4), gates_pos (M, 6, 3),
    gates_yaw (M, 6), target (M,), env (M,), t (M,) step in episode, episode (M,) global episode id.
    """
    from fpv.closed_loop import FlightBatch

    rng = np.random.default_rng(seed)
    fb = FlightBatch(n_envs, teacher_ckpt, seed=seed, randomize=True, wide_spawn_p=wide_spawn_p, scenario=scenario)
    s = fb.reset()
    noise = rng.uniform(0, teacher_noise, n_envs).astype(np.float32)
    burst_left, burst_off = np.zeros(n_envs, int), np.zeros((n_envs, 4), np.float32)
    ep = np.arange(n_envs)
    next_ep = n_envs
    keys = ("pos", "quat", "gates_pos", "gates_yaw", "target", "t")
    out = {k: [] for k in (*keys, "env", "episode")}
    for i in range(steps):
        live = ~s["done"]
        for k in keys:
            out[k].append(s[k][live])
        out["env"].append(np.flatnonzero(live))
        out["episode"].append(ep[live])
        a = s["teacher_action"] + noise[:, None] * rng.standard_normal((n_envs, 4)).astype(np.float32)
        start = (rng.random(n_envs) < burst_p) & (burst_left == 0)
        burst_left[start] = rng.integers(burst_len[0], burst_len[1] + 1, int(start.sum()))
        burst_off[start] = rng.uniform(-burst_mag, burst_mag, (int(start.sum()), 4))
        a = a + np.where(burst_left[:, None] > 0, burst_off, 0)
        burst_left = np.maximum(burst_left - 1, 0)
        s = fb.step(np.clip(a, -1, 1))
        if s["done"].any():
            d = s["done"]
            s = fb.reset(d)
            noise[d] = rng.uniform(0, teacher_noise, int(d.sum()))
            burst_left[d] = 0
            ep[d] = next_ep + np.arange(int(d.sum()))
            next_ep += int(d.sum())
        if log and (i + 1) % 250 == 0:
            log(f"  flights {scenario or 'random'}: step {i + 1}/{steps}, frames {sum(len(x) for x in out['env'])}")
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["pos"], res["quat"] = res["pos"].astype(np.float32), res["quat"].astype(np.float32)
    return res


def perturb_poses(key, pos, quat, p: float = 0.5, tilt: float = 0.35, yaw: float = 0.5, offset: float = 0.4,
                  ground: float = 0.08):  # fmt: skip
    """With probability p per pose: rotate the body by a random small rotation (roll / pitch uniform in
    +-tilt rad, yaw +-yaw rad, applied in the body frame) and shift it by U(+-offset) m per axis
    (kept above ``ground`` m). jax arrays in, jax arrays out."""
    import jax
    import jax.numpy as jnp

    from fpv.env import quat_mul

    n = pos.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    on = jax.random.uniform(k1, (n,)) < p
    rv = jax.random.uniform(k2, (n, 3), minval=-1.0, maxval=1.0) * jnp.asarray([tilt, tilt, yaw])
    ang = jnp.linalg.norm(rv, axis=-1, keepdims=True)
    axis = rv / jnp.maximum(ang, 1e-9)
    dq = jnp.concatenate([axis * jnp.sin(ang / 2), jnp.cos(ang / 2)], -1)
    q2 = quat_mul(quat, dq)
    p2 = pos + jax.random.uniform(k3, (n, 3), minval=-offset, maxval=offset)
    p2 = p2.at[:, 2].set(jnp.maximum(p2[:, 2], ground))
    return jnp.where(on[:, None], p2, pos), jnp.where(on[:, None], q2, quat)


def scene_pool(n: int, seed: int, width: int = W, height: int = H) -> dict:
    """``n`` random looks of `fpv.render_synth.SynthRenderer` (its brightness / texture / noise
    randomisation, Unreal-like share 0.7), highlight off (``tgt`` = -1), as numpy arrays (n, ...)."""
    from fpv.render_synth import SynthRenderer

    r = SynthRenderer(1, width=width, height=height, seed=seed)
    sc = r._draw_scene(n)
    sc["tgt"] = np.full(n, -1, np.int32)
    return sc


def make_renderer(width: int = W, height: int = H, supersample: int = 2):
    """The ray caster of `fpv.render_synth` as a pure function
    render(pos (n,3), quat (n,4), gates_pos (n,6,3), gates_yaw (n,6), scene dict (n,...), key) -> (uint8 (n,H,W,3), ids)."""
    from fpv.render_synth import _make_render

    return _make_render(width, height, supersample)
