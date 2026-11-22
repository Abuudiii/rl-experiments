"""Gate geometry from detections: pose of a square gate from its 4 image corners (batched JAX).

Input: corners (..., 4, 2) in pixels (320x180, `fpv.perception.labels` convention: u right, v down, integer =
pixel centre, f = 112.03 px, cx = 159.5, cy = 89.5), in contract image order (clockwise from the corner
nearest image top-left). Clockwise in the image means the model square, seen from the camera, is
(-h, -h), (h, -h), (h, h), (-h, h) in (right, down) plane coordinates, h = GATE_OPENING / 2, whatever side
the gate is seen from: the plane normal of the fit always points away from the camera, so the travel
direction is NOT observable from one gate (the estimator resolves it with the gate-order list).

Method (near gates): planar homography from the 4 point pairs (exact DLT, 8x8 solve), decomposed into
rotation and translation (scale = mean of the first two column norms, symmetric orthonormalisation of the
two plane axes), then ``refine_iters`` Gauss-Newton steps on the 6-dof pose minimising the pixel
reprojection error. Far gates (a few px): DLT range is dominated by corner noise (detector worker,
preliminary model: -41 % bias at 4-8 m), so the range comes from the apparent size, 1.07 * f * 0.4 m /
longest quad edge, along the ray through the intersection of the diagonals (bearing is good at every
distance). The two ranges are blended linearly between PNP_NEAR and PNP_FAR (by the size-based range);
``w_pnp`` is that blend weight, and the PnP orientation (normal, in-plane axes) should be trusted with it.
Quality in [0, 1] (0 = unusable) combines:
  * apparent size sqrt(area) in px (tiny gates -> 0) and a convex quad,
  * weighted by w_pnp: viewing angle |cos| between line of sight and plane normal (edge-on -> 0) and
    squareness of the homography (ratio and orthogonality of its plane axes, before orthonormalisation).
Partly visible gates need no special case: the contract's corners may lie outside the image (the
detector extrapolates them) and the fit uses them like the others; their noise enters through the
squareness term.

Outputs are in the BODY frame (x forward, y left, z up), the frame of `fpv.env.teacher_obs`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fpv import track
from fpv.perception import labels

F_PX, CX, CY = labels.intrinsics()
_H = track.GATE_OPENING / 2
MODEL = np.array([[-_H, -_H], [_H, -_H], [_H, _H], [-_H, _H]], np.float32)  # (right, down) in the plane
CAM_TO_BODY = labels._BODY_TO_CAM.T  # body = CAM_TO_BODY @ cam


def _homography(xy):
    """xy (..., 4, 2) normalised image points of MODEL -> H (..., 3, 3) with h33 = 1."""
    X, Y = MODEL[:, 0], MODEL[:, 1]
    x, y = xy[..., 0], xy[..., 1]
    zero, one = jnp.zeros_like(x), jnp.ones_like(x)
    X, Y = jnp.broadcast_to(X, x.shape), jnp.broadcast_to(Y, x.shape)
    r1 = jnp.stack([X, Y, one, zero, zero, zero, -x * X, -x * Y], -1)
    r2 = jnp.stack([zero, zero, zero, X, Y, one, -y * X, -y * Y], -1)
    A = jnp.concatenate([r1, r2], -2)  # (..., 8, 8)
    b = jnp.concatenate([x, y], -1)
    A = A + 1e-9 * jnp.eye(8)
    h = jnp.linalg.solve(A, b[..., None])[..., 0]
    return jnp.concatenate([h, jnp.ones_like(h[..., :1])], -1).reshape(*h.shape[:-1], 3, 3)


def _rodrigues(w):
    """Rotation matrix of rotation vectors w (..., 3); smooth (differentiable) at w = 0."""
    th = jnp.sqrt((w * w).sum(-1) + 1e-12)[..., None, None]
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    z = jnp.zeros_like(wx)
    W = jnp.stack([jnp.stack([z, -wz, wy], -1), jnp.stack([wz, z, -wx], -1), jnp.stack([-wy, wx, z], -1)], -2)
    a = jnp.sin(th) / th
    b = 0.5 * (jnp.sin(th / 2) / (th / 2)) ** 2
    return jnp.eye(3) + a * W + b * W @ W


def _project(R, t):
    """Model corners under pose (R (..., 3, 3), t (..., 3)) -> normalised image points (..., 4, 2)."""
    P = jnp.asarray(MODEL)
    pc = jnp.einsum("...ij,pj->...pi", R[..., :, :2], P) + t[..., None, :]
    z = jnp.maximum(pc[..., 2:], 1e-4)
    return pc[..., :2] / z


def _refine(R, t, xy, iters: int):
    """Gauss-Newton on (rotation vector perturbation, translation), 8 residuals, 6 unknowns."""

    def resid(p, R0, xy):
        return (_project(_rodrigues(p[:3]) @ R0, p[3:]) - xy).reshape(-1)

    def one(R0, t0, xy):
        def body(carry, _):
            R0, t0 = carry
            p = jnp.concatenate([jnp.zeros(3), t0])
            r = resid(p, R0, xy)
            J = jax.jacfwd(resid)(p, R0, xy)
            JTJ = J.T @ J + 1e-6 * jnp.eye(6)
            dp = -jnp.linalg.solve(JTJ, J.T @ r)
            dp = jnp.clip(dp, -0.5, 0.5)
            return (_rodrigues(dp[:3]) @ R0, t0 + dp[3:]), None

        (R1, t1), _ = jax.lax.scan(body, (R0, t0), None, iters)
        return R1, t1

    shp = R.shape[:-2]
    f = jax.vmap(one)
    R1, t1 = f(R.reshape(-1, 3, 3), t.reshape(-1, 3), xy.reshape(-1, 4, 2))
    return R1.reshape(*shp, 3, 3), t1.reshape(*shp, 3)


# Size-based range: the longest quad edge over-reads the opening for oblique views; with TRUE corners the
# detector worker measured a -6 to -8 % range bias of f * 0.4 / longest_edge, corrected by SIZE_BIAS.
SIZE_BIAS = 1.07
PNP_NEAR, PNP_FAR = 2.5, 4.0  # m: full PnP below, size-based range beyond (DLT range std 0.14 at 2-4 m, -41 % bias at 4-8 m)


def gate_pose(corners, refine_iters: int = 2):
    """Pose of square gates from image corners (..., 4, 2) px.

    Returns dict (leading shape ...):
      centre (..., 3) gate centre in the body frame [m]; normal (..., 3) unit plane normal in the body frame,
      pointing away from the camera; axis_r, axis_d (..., 3) unit in-plane axes (image-right-like and
      image-down-like, body frame); dist (..., ); quality (..., ) in [0, 1]; side_px, cos_view, square_err.
    """
    xy = jnp.stack([(corners[..., 0] - CX) / F_PX, (corners[..., 1] - CY) / F_PX], -1)
    Hm = _homography(xy)
    h1, h2, h3 = Hm[..., :, 0], Hm[..., :, 1], Hm[..., :, 2]
    n1, n2 = jnp.linalg.norm(h1, axis=-1), jnp.linalg.norm(h2, axis=-1)
    lam = 0.5 * (n1 + n2)
    sgn = jnp.where(h3[..., 2] >= 0, 1.0, -1.0)
    a = sgn[..., None] * h1 / jnp.maximum(n1, 1e-9)[..., None]
    b = sgn[..., None] * h2 / jnp.maximum(n2, 1e-9)[..., None]
    t = sgn[..., None] * h3 / jnp.maximum(lam, 1e-9)[..., None]
    c = a + b
    c = c / jnp.maximum(jnp.linalg.norm(c, axis=-1, keepdims=True), 1e-9)
    d = a - b
    d = d / jnp.maximum(jnp.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    r1, r2 = (c + d) / jnp.sqrt(2.0), (c - d) / jnp.sqrt(2.0)
    r3 = jnp.cross(r1, r2)
    R = jnp.stack([r1, r2, r3], -1)
    square_err = jnp.abs(jnp.log(jnp.maximum(n1, 1e-9) / jnp.maximum(n2, 1e-9))) + jnp.abs((a * b).sum(-1))
    if refine_iters:
        R, t = _refine(R, t, xy, refine_iters)
    r1, r2, r3 = R[..., :, 0], R[..., :, 1], R[..., :, 2]
    # Normal away from the camera
    flip = jnp.where((r3 * t).sum(-1) < 0, -1.0, 1.0)[..., None]
    r3 = r3 * flip
    dist_pnp = jnp.linalg.norm(t, axis=-1)
    cos_view = jnp.abs((r3 * t).sum(-1)) / jnp.maximum(dist_pnp, 1e-6)
    x, y = corners[..., 0], corners[..., 1]
    cross = (jnp.roll(x, -1, -1) - x) * (jnp.roll(y, -2, -1) - jnp.roll(y, -1, -1)) - (
        jnp.roll(y, -1, -1) - y) * (jnp.roll(x, -2, -1) - jnp.roll(x, -1, -1))  # fmt: skip
    convex = (cross > 0).all(-1) | (cross < 0).all(-1)
    area = 0.5 * jnp.abs((x * jnp.roll(y, -1, -1) - jnp.roll(x, -1, -1) * y).sum(-1))
    side = jnp.sqrt(area)
    # Size-based range (robust on small quads): depth = SIZE_BIAS * f * 0.4 m / longest edge, along the ray
    # through the intersection of the diagonals. Blended with the PnP range by w_pnp (1 below PNP_NEAR m,
    # 0 beyond PNP_FAR m), the PnP orientation is trusted with the same weight.
    edge = jnp.sqrt(((corners - jnp.roll(corners, -1, -2)) ** 2).sum(-1)).max(-1)
    p0, p1, p2, p3 = xy[..., 0, :], xy[..., 1, :], xy[..., 2, :], xy[..., 3, :]
    d1, d2 = p2 - p0, p3 - p1
    den = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    s_ = ((p1 - p0)[..., 0] * d2[..., 1] - (p1 - p0)[..., 1] * d2[..., 0]) / jnp.where(jnp.abs(den) > 1e-9, den, 1e-9)
    ctr = jnp.where((jnp.abs(den) > 1e-9)[..., None], p0 + s_[..., None] * d1, xy.mean(-2))
    ray = jnp.concatenate([ctr, jnp.ones_like(ctr[..., :1])], -1)  # unit depth (camera z = 1)
    depth_size = SIZE_BIAS * F_PX * track.GATE_OPENING / jnp.maximum(edge, 1e-3)  # f L / edge is a DEPTH
    t_size = ray * depth_size[..., None]
    dist_size = jnp.linalg.norm(t_size, axis=-1)
    w_pnp = jnp.clip((PNP_FAR - dist_size) / (PNP_FAR - PNP_NEAR), 0.0, 1.0)
    ok_pnp = jnp.isfinite(t).all(-1) & (t[..., 2] > 0.05)
    w_pnp = jnp.where(ok_pnp, w_pnp, 0.0)
    t = w_pnp[..., None] * jnp.nan_to_num(t) + (1 - w_pnp[..., None]) * t_size
    dist = jnp.linalg.norm(t, axis=-1)
    q_pnp = jnp.clip(cos_view / 0.25, 0, 1) * jnp.exp(-square_err / 0.3)
    q = (jnp.nan_to_num(w_pnp * q_pnp) + (1 - w_pnp)) * jnp.clip((side - 2.0) / 4.0, 0, 1) * convex * (dist < 20.0)
    M = jnp.asarray(CAM_TO_BODY)
    q = jnp.where(jnp.isfinite(t).all(-1), q, 0.0)

    def body(v):
        return jnp.nan_to_num(jnp.einsum("ij,...j->...i", M, v))

    return dict(centre=body(t), normal=body(r3), axis_r=body(r1), axis_d=body(r2), dist=jnp.nan_to_num(dist),
                quality=q, w_pnp=jnp.nan_to_num(w_pnp), dist_pnp=jnp.nan_to_num(dist_pnp), dist_size=dist_size,
                side_px=side, cos_view=cos_view, square_err=square_err)  # fmt: skip
