"""Detection-noise model: detector-like outputs from the true projected gate corners (batched JAX).

Output = the detector contract: ``corners`` (N, K, 4, 2) pixels at 320x180 (u right, v down, integer =
pixel centre, `fpv.perception.labels` convention), the 4 opening corners of each detected gate in image
order (clockwise from the corner nearest image top-left, `labels.canonical_order`); ``conf`` (N, K) in
[0, 1]; slots unordered; empty slots have conf 0.

Ground truth visibility is the contract rule (`labels.gate_labels`: all corners in front, >= 2 corners
inside the image, centre within 12 m; occlusion ignored). Every term below scales with the per-episode
noise scale ``s`` (0 = perfect detections: exactly the visible gates, true corners, conf 1):

  corner noise   per corner and axis, sigma = s * (px_base + px_far * max(dist - 6, 0) + px_rel * side_px
                 + px_outside * [corner outside the image]); split into a white part and an AR(1) part
                 (correlation ``rho`` per 20 ms frame, variance share ``corr_share``); plus a heavy-tailed
                 mixture: with probability s * p_mid a corner gets an extra N(0, px_mid) error, with
                 s * p_outlier an extra N(0, px_outlier). side_px = sqrt(area of the true quad). Range errors
                 are not injected: they follow from the corner errors through the estimator's own geometry.
  misses         a visible gate is not reported with probability s * (p_miss + p_miss_partial *
                 [< 4 corners inside] + p_miss_far * clip((dist - 3) / 4, 0, 1)), and with p_miss_stay
                 after a miss of the same gate (misses come in bursts).
  false gates    with probability s * p_false per frame a spurious quad (random place, size 6-60 px,
                 roughly square, mildly skewed) appears; it persists with p_false_stay per frame.
  confidence     true: 0.95 - 0.25 * [partial] - 0.02 * dist + s * 0.1 * N(0, 1); false: U(0.3, 0.9).
  dropouts       (variant d) with rate ``dropout_rate`` per second a dropout starts that hides the TRUE
                 NEXT gate for U(1, dropout_max_steps) frames (independent of ``s``).
  latency        with ``latency`` the outputs are those of the previous frame (one frame = 20 ms).
  occlusion      (every s, also "perfect") a gate with >= ``occl_hidden`` of its frame behind a nearer
                 gate's frame (16 samples on the ring mid-line against the projected rings) is not reported.
  slot shuffle   the K slots are randomly permuted every frame.

Defaults (1x): the detector worker's PRELIMINARY measured numbers (see `DetNoiseConfig`), the
pessimistic end; `calibrate` replaces terms when the final detector is measured.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import struct

from fpv import track
from fpv.perception import labels

K = 6


@dataclasses.dataclass(frozen=True)
class DetNoiseConfig:
    # 1x = the FINAL stage-1 detector (pilot A, 40k steps; detector worker, 2026-11-21: 32,000 held-out random-
    # track frames, threshold 0.4): corner error mean/median/p90/p99 0.46/0.12/0.99/5.4 px (8-12 m median 0.49);
    # corners outside the image (3 / 2 inside: median 0.52 / 1.26 px); errors white (lag-1 corr 0.04); misses
    # excluding >= 70 %-occluded gates 0.5 / 2.5 / 5.1 / 3.6 % at 0-2 / 2-4 / 4-8 / 8-12 m, 83-85 % of
    # >= 70 %-occluded gates missed; next gate missed in 0.29 % of visible frames, P(miss | miss) 0.53-0.71,
    # P(miss | hit) 0.07-0.13 %; 0.006-0.009 false detections per frame.
    # (Before: calibrated to the 600-step PRELIMINARY model, ~5x worse: core 0.55 px, outliers 2.5 % x 6 px,
    # misses 1.4 % + far 5 %, false 0.025 / frame; policies est_*_v2 were TRAINED with that older scale.)
    px_base: float = 0.10  # Gaussian core sigma per axis (median corner error ~0.12 px)
    px_far: float = 0.06  # px per m beyond 6 m
    px_rel: float = 0.0  # px per px of apparent size
    px_outside: float = 1.0  # extrapolated corners outside the image
    p_mid: float = 0.12  # per corner: moderate error component (p90 ~1 px)
    px_mid: float = 0.8
    p_outlier: float = 0.015  # per corner: heavy tail (p99 ~5 px)
    px_outlier: float = 3.5
    rho: float = 0.85  # AR(1) per frame of the correlated part
    corr_share: float = 0.05
    p_miss: float = 0.0008  # fresh miss of a fully visible near gate
    p_miss_partial: float = 0.005
    p_miss_far: float = 0.015  # fresh-miss probability added at >= 6 m, ramping from 2 m
    p_miss_stay: float = 0.6  # P(miss | missed in the previous frame)
    p_false: float = 0.005
    p_false_stay: float = 0.3
    conf_noise: float = 0.1
    dropout_rate: float = 0.0  # per second, dropouts of the true next gate (variant d)
    dropout_max_steps: int = 25  # 0.5 s at 50 Hz
    latency: bool = False
    occl_hidden: float = 0.7  # gates with this share of their frame behind a nearer gate ...
    p_occl_miss: float = 0.85  # ... are missed with this probability (any noise scale)


@struct.dataclass
class DetState:
    corr: jax.Array  # (N, G, 4, 2) AR(1) corner noise in units of sigma, per world corner
    missed: jax.Array  # (N, G) bool, gate missed in the previous frame
    false_on: jax.Array  # (N,) bool
    false_quad: jax.Array  # (N, 4, 2)
    dropout_left: jax.Array  # (N,) int frames
    prev_corners: jax.Array  # (N, K, 4, 2) last output (latency)
    prev_conf: jax.Array  # (N, K)
    scale: jax.Array  # (N,)


def det_init(key, scale, n_gates: int = 6) -> DetState:
    n = scale.shape[0]
    return DetState(
        corr=jax.random.normal(key, (n, n_gates, 4, 2)), missed=jnp.zeros((n, n_gates), bool),
        false_on=jnp.zeros(n, bool), false_quad=jnp.zeros((n, 4, 2)), dropout_left=jnp.zeros(n, jnp.int32),
        prev_corners=jnp.zeros((n, K, 4, 2)), prev_conf=jnp.zeros((n, K)), scale=scale,
    )  # fmt: skip


def det_reset_masked(state: DetState, key, mask, scale) -> DetState:
    new = det_init(key, scale, state.corr.shape[1])
    return jax.tree.map(lambda a, b: jnp.where(mask.reshape(-1, *[1] * (a.ndim - 1)), a, b), new, state)


def project_gates(pos, quat, gates_pos, gates_yaw):
    """True opening corners per world corner (N, G, 4, 2) px, inside (N, G, 4), visible (N, G), dist (N, G),
    side_px (N, G); the contract visibility rule of `labels.gate_labels`."""
    n, g = gates_pos.shape[:2]
    cw = track.gate_corners(gates_pos, gates_yaw)
    cam = labels.to_camera(pos, quat, jnp.concatenate([cw.reshape(n, g * 4, 3), gates_pos], 1))
    uv = labels.project(cam)
    c_uv, c_z = uv[:, : g * 4].reshape(n, g, 4, 2), cam[:, : g * 4, 2].reshape(n, g, 4)
    front = (c_z > labels.MIN_DEPTH).all(-1)
    inside = labels.in_image(c_uv) & (c_z > labels.MIN_DEPTH)
    dist = jnp.linalg.norm(cam[:, g * 4 :], axis=-1)
    visible = front & (inside.sum(-1) >= 2) & (dist <= labels.MAX_DIST)
    x, y = c_uv[..., 0], c_uv[..., 1]
    area = 0.5 * jnp.abs((x * jnp.roll(y, -1, -1) - jnp.roll(x, -1, -1) * y).sum(-1))
    return c_uv, inside, visible, dist, jnp.sqrt(area)


def _in_quad(p, quad):
    """Points p (..., P, 2) inside convex quads (..., 4, 2) (either winding) -> (..., P)."""
    a = quad[..., None, :, :]
    b = jnp.roll(quad, -1, -2)[..., None, :, :]
    pp = p[..., :, None, :]
    cr = (b[..., 0] - a[..., 0]) * (pp[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (pp[..., 0] - a[..., 0])
    return (cr >= 0).all(-1) | (cr <= 0).all(-1)


def occluded_fraction(pos, quat, gates_pos, gates_yaw, n_side: int = 4):
    """Share (N, G) of each gate's frame hidden behind a NEARER gate's frame: 4 * n_side sample points on the
    mid-line of the frame ring are tested against the projected ring (outer square minus opening) of every
    nearer gate. Points behind the camera count as not hidden."""
    n, g = gates_pos.shape[:2]
    mid = (track.GATE_OPENING + track.GATE_OUTER) / 4  # half side of the ring's mid-line
    s = jnp.linspace(-1.0, 1.0, n_side, endpoint=False)
    edge = jnp.concatenate([jnp.stack([s, jnp.ones_like(s)], -1), jnp.stack([jnp.ones_like(s), -s], -1),
                            jnp.stack([-s, -jnp.ones_like(s)], -1), jnp.stack([-jnp.ones_like(s), s], -1)])  # (S, 2)  # fmt: skip
    _, left = track.gate_axes(gates_yaw)
    up = jnp.asarray([0.0, 0.0, 1.0])
    pts = gates_pos[:, :, None] + mid * (edge[None, None, :, :1] * left[:, :, None] + edge[None, None, :, 1:] * up)
    S = pts.shape[2]
    outer = track.gate_corners(gates_pos, gates_yaw, track.GATE_OUTER)
    inner = track.gate_corners(gates_pos, gates_yaw)
    allp = jnp.concatenate([pts.reshape(n, -1, 3), outer.reshape(n, -1, 3), inner.reshape(n, -1, 3), gates_pos], 1)
    cam = labels.to_camera(pos, quat, allp)
    uv = labels.project(cam)
    o = g * S
    p_uv, p_z = uv[:, :o].reshape(n, g, S, 2), cam[:, :o, 2].reshape(n, g, S)
    out_uv = uv[:, o : o + 4 * g].reshape(n, g, 4, 2)
    out_z = cam[:, o : o + 4 * g, 2].reshape(n, g, 4)
    in_uv = uv[:, o + 4 * g : o + 8 * g].reshape(n, g, 4, 2)
    depth = jnp.linalg.norm(cam[:, o + 8 * g :], axis=-1)  # (N, G)
    # sample s of gate a inside the ring of gate b: (N, A, B, S)
    in_out = _in_quad(p_uv[:, :, None], out_uv[:, None])
    in_in = _in_quad(p_uv[:, :, None], in_uv[:, None])
    ok_b = (out_z > labels.MIN_DEPTH).all(-1)  # occluder fully in front of the camera
    nearer = depth[:, None, :] < depth[:, :, None] - 0.05  # (N, A, B)
    hid = in_out & ~in_in & (nearer & ok_b[:, None, :])[..., None] & (p_z > labels.MIN_DEPTH)[:, :, None]
    return hid.any(2).mean(-1)


def detect(state: DetState, key, pos, quat, gates_pos, gates_yaw, target, cfg: DetNoiseConfig = DetNoiseConfig(),
           dt: float = 0.02):  # fmt: skip
    """One frame. ``target`` (N,) = true next gate (only used by dropouts). Returns (state, corners, conf, aux);
    aux: gate (N, K) true gate index of each slot (-1 = empty / false), visible (N, G)."""
    n, g = gates_pos.shape[:2]
    s = state.scale
    ks = jax.random.split(key, 16)
    uv, inside, visible, dist, side = project_gates(pos, quat, gates_pos, gates_yaw)
    sig = s[:, None, None] * (cfg.px_base + cfg.px_far * jnp.maximum(dist - 6, 0)[..., None] + cfg.px_rel * side[..., None]
                              + cfg.px_outside * (~inside))  # (N, G, 4)  # fmt: skip
    corr = cfg.rho * state.corr + jnp.sqrt(1 - cfg.rho**2) * jax.random.normal(ks[0], state.corr.shape)
    white = jax.random.normal(ks[1], uv.shape)
    # heavy-tailed mixture per corner: moderate (p_mid, px_mid) and outlier (p_outlier, px_outlier) components
    u_c = jax.random.uniform(ks[12], uv.shape[:-1])
    is_out = u_c < jnp.minimum(cfg.p_outlier * s, 0.5)[:, None, None]
    is_mid = ~is_out & (u_c < jnp.minimum((cfg.p_outlier + cfg.p_mid) * s, 0.9)[:, None, None])
    extra = jnp.where(is_out, cfg.px_outlier, jnp.where(is_mid, cfg.px_mid, 0.0))
    noise = ((jnp.sqrt(cfg.corr_share) * corr + jnp.sqrt(1 - cfg.corr_share) * white) * sig[..., None]
             + extra[..., None] * jax.random.normal(ks[13], uv.shape))  # fmt: skip
    uv_n = labels.canonical_order(uv + noise)
    partial = inside.sum(-1) < 4
    p = s[:, None] * (cfg.p_miss + cfg.p_miss_partial * partial + cfg.p_miss_far * jnp.clip((dist - 2) / 4, 0, 1))
    p = jnp.where(state.missed, jnp.maximum(p, cfg.p_miss_stay * (s[:, None] > 0)), p)
    miss = visible & (jax.random.uniform(ks[2], (n, g)) < jnp.clip(p, 0, 0.95))
    # gates with >= occl_hidden of their frame behind a nearer gate are mostly missed (detector: 83-85 %)
    occl = visible & (occluded_fraction(pos, quat, gates_pos, gates_yaw) >= cfg.occl_hidden)
    miss = miss | (occl & (jax.random.uniform(ks[14], (n, g)) < cfg.p_occl_miss))
    # Dropouts of the true next gate
    start = (state.dropout_left == 0) & (jax.random.uniform(ks[3], (n,)) < cfg.dropout_rate * dt)
    length = jax.random.randint(ks[4], (n,), 1, cfg.dropout_max_steps + 1)
    left = jnp.where(start, length, state.dropout_left)
    drop = (left > 0)[:, None] & (jnp.arange(g)[None] == target[:, None])
    det = visible & ~miss & ~drop
    conf = 0.95 - 0.25 * partial - 0.02 * dist + s[:, None] * cfg.conf_noise * jax.random.normal(ks[5], (n, g))
    conf = jnp.where(det, jnp.clip(conf, 0.05, 1.0), 0.0)
    conf = jnp.where(det & (s[:, None] == 0), 1.0, conf)
    # False gate: random roughly square quad
    new_f = jax.random.uniform(ks[6], (n,)) < s * cfg.p_false
    keep_f = state.false_on & (jax.random.uniform(ks[7], (n,)) < cfg.p_false_stay)
    ctr = jax.random.uniform(ks[8], (n, 1, 2)) * jnp.asarray([labels.W, labels.H])
    half = jax.random.uniform(ks[9], (n, 1, 1), minval=3.0, maxval=30.0)
    sq = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    skew = 1 + 0.3 * jax.random.uniform(ks[10], (n, 4, 2), minval=-1.0, maxval=1.0)
    quad_new = ctr + half * sq * skew
    jitter = jax.random.normal(ks[11], (n, 4, 2))
    f_quad = jnp.where(new_f[:, None, None] & ~keep_f[:, None, None], quad_new, state.false_quad + jitter)
    f_on = keep_f | new_f
    f_conf = jnp.where(f_on, jax.random.uniform(jax.random.fold_in(key, 1), (n,), minval=0.3, maxval=0.9), 0.0)
    # Slots: the G true gates + 1 false candidate, keep the K most confident, then shuffle
    cand = jnp.concatenate([uv_n, labels.canonical_order(f_quad)[:, None]], 1)  # (N, G + 1, 4, 2)
    cconf = jnp.concatenate([conf, f_conf[:, None]], 1)
    gid = jnp.concatenate([jnp.broadcast_to(jnp.arange(g), (n, g)), jnp.full((n, 1), -1)], 1)
    top = jnp.argsort(-cconf, axis=1)[:, :K]
    perm = jnp.argsort(jax.random.uniform(jax.random.fold_in(key, 2), (n, K)), axis=1)
    sel = jnp.take_along_axis(top, perm, 1)
    corners = jnp.take_along_axis(cand, sel[..., None, None], 1)
    out_conf = jnp.take_along_axis(cconf, sel, 1)
    gate = jnp.where(out_conf > 0, jnp.take_along_axis(gid, sel, 1), -1)
    corners = jnp.where((out_conf > 0)[..., None, None], corners, 0.0)
    new_state = state.replace(corr=corr, missed=visible & (miss | drop), false_on=f_on, false_quad=f_quad,
                              dropout_left=jnp.maximum(left - 1, 0), prev_corners=corners, prev_conf=out_conf)  # fmt: skip
    if cfg.latency:
        corners, out_conf = state.prev_corners, state.prev_conf
    return new_state, corners, out_conf, dict(gate=gate, visible=visible)


def calibrate(cfg: DetNoiseConfig, **measured) -> DetNoiseConfig:
    """Replace default terms by measured ones (e.g. px_base, px_per_m, p_miss, rho)."""
    return dataclasses.replace(cfg, **measured)
