"""State estimator of the perception pipeline (batched JAX, runs inside PPO for thousands of drones).

Inputs per 50 Hz step: detections (contract: corners (N, K, 4, 2) px, conf (N, K)), the IMU sample (gyro,
accel, `fpv.perception.imu`), the gate-order list (NOMINAL gate centres (N, 6, 3) and yaws (N, 6) in the
list's own frame, as walked before the flight) and dt. The drone's world pose is never used.

Frames. Body b: x forward, y left, z up. Heading frame H: origin at the drone, z up (gravity), x = the body
x axis projected on the horizontal; body -> H is the tilt-only rotation T(u) built from the estimated up
vector u (world z in body coordinates). Everything the estimator remembers lives in H and is moved by
ego-motion every step; no world frame is needed and heading drift is irrelevant by construction.

Per step:
 1. Attitude: u is rotated by the bias-corrected gyro (u <- Rb^T u, Rb = exp(w dt)); the heading change
    dpsi follows exactly from T(u_old) Rb T(u_new)^T. Tilt correction (complementary filter): the up edge
    of every good gate fit nearer than ``tilt_range`` is an absolute measurement of u (gates stand
    upright); only its component in the gate plane is used (the out-of-plane part, rotation about the
    gate's horizontal axis, is the badly conditioned one of a 4-point fit): u <- normalise(u + kp *
    mean_w(P_plane (u_meas - u))), gyro bias b <- b - ki * (u_meas x u). The accelerometer is NOT used for
    tilt: on a quadrotor it measures thrust + drag, not gravity (steps where the accelerometer direction is
    within 5 deg of true up are rare in flight: `scripts/check_estimator.py`, imu physics).
 2. Kalman filter (linear given the attitude) on x = [v (3), p_0 .. p_5 (3 each)] in H: velocity and the
    relative positions of the six list gates. Prediction: v' = Rz(-dpsi)(v + a dt) with a = T(u) f - g e_z
    from the accelerometer; p' = Rz(-dpsi)(p - v dt - a dt^2 / 2); process noise ``sig_acc`` on v and
    ``sig_rot`` * |p| on the gates (heading error). Measurement: the PnP centre of a matched gate, with a
    covariance that is ``sig_px`` / f * d across the line of sight and d^2 * sig_px / (f * 0.4) along it
    (range from apparent size), inflated by 1 / quality. Velocity is thus "how gate positions change over
    time", fused with the accelerometer integration, with the correct weighting of range against bearing.
 3. Detections: square-gate PnP (`geometry.gate_pose`) -> centre, plane normal (travel direction
    ambiguous), quality; usable if conf > ``conf_min`` and quality > ``q_min``.
 4. Track placement: the list is placed in H by a 2D rigid transform (yaw pth, translation pt), refitted
    to the matched gates every frame (weighted 2D Kabsch on centres + gate yaws, low-pass
    ``place_gain``) and moved by ego-motion in between. It predicts the gates not tracked by the filter.
 5. Association: detections are matched to predicted list gates (the filter's gate if matched within
    ``mem_steps``, else the placement's prediction) by mutual nearest neighbour inside a gate of
    tau0 + tau_range * d + (3 sigma of the filter's gate | tau_place). A matched gate the filter did not
    track is (re)initialised at the measurement. Gate yaws: low-pass ``alpha_yaw`` (travel direction
    resolved against the prediction).
 6. Lock and relocalisation: hypotheses "detection k is list gate j, travelled in direction f" (K x 6 x 2)
    each fix a placement; a placement scores the detections it explains (inliers within a gate that grows
    with the lever arm to the anchor detection), minus ``neg_weight`` per list gate it predicts in the
    camera (< ``neg_range`` m, facing the camera) that no detection explains, + 0.2 for approaching the
    anchor gate from its entry side. First lock: best hypothesis with score >= 0.9. Relocalisation: when
    for ``reloc_steps`` consecutive frames either no detection is matched although a good near one
    exists, or the best hypothesis explains ``reloc_margin`` more detections than were matched, the
    estimator jumps to the best hypothesis (filter gates reset to the placement).
 7. Which gate is next (index nxt into the list), never from the world pose:
    a) passing: the estimate of gate nxt crosses the drone: signed distance of the drone along the gate's
       travel direction goes from < 0 to >= 0 with lateral and vertical offset < ``pass_window`` ->
       nxt + 1 (the filter dead-reckons the gate through the last metres, where it leaves the image);
    b) consistency / recovery: every frame the gate the drone is approaching is computed from the
       estimated gate poses: cost_j = 0.5 * distance + 3 if already past gate j's plane + 2 if gate j is
       not in the camera + 1 if not past gate j-1's plane; if argmin_j beats nxt by ``geo_margin`` for
       ``geo_steps`` consecutive frames, nxt jumps to it (missed or spurious pass);
    c) every (re)lock sets nxt by rule b at once.
 8. Output (`observation`): the teacher's 43 numbers with estimates (body velocity, tilt-only attitude
    matrix T(u) in place of the world rotation, bias-corrected step-mean rates, body-frame corners of gates
    nxt and nxt + 1, previous action) + 3 flags (age of the next / after-next gate information, lock).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import struct

from fpv import track
from fpv.env import CAM_FWD, CAM_LEFT, CAM_UP, G, MAX_RATE_XY, MAX_RATE_Z, POS_SCALE, VEL_SCALE
from fpv.perception import geometry

N_GATES = 6
NX = 3 + 3 * N_GATES + 2  # [v (3), gates (18), tilt error about H x / y (2)]
TL = slice(3 + 3 * N_GATES, NX)
EST_OBS_DIM = 46
BIG_AGE = 10_000


@dataclasses.dataclass(frozen=True)
class EstConfig:
    conf_min: float = 0.5
    q_min: float = 0.05
    refine_iters: int = 2
    # Kalman filter (design values; the estimator does not know the episode's true noise level)
    sig_acc: float = 1.0  # m/s^2, velocity process noise (accelerometer noise + tilt error * g)
    sig_rot: float = 0.02  # rad/s, heading process noise on the gates (gyro noise)
    sig_px: float = 1.5  # px, corner noise assumed for the measurement covariance
    r_floor: float = 0.02  # m, minimum measurement sigma
    sig_v0: float = 2.0  # m/s, initial velocity sigma
    sig_tilt0: float = 0.2  # rad, initial tilt sigma (episodes spawn in the air with an unknown tilt)
    sig_tilt_rw: float = 0.02  # rad/s, tilt process noise (gyro noise + bias)
    chi2_gate: float = 16.0  # Mahalanobis^2 (3 dof) above which a tracked gate's measurement is rejected
    alpha_yaw: float = 0.3
    # tilt
    kp_tilt: float = 0.05
    ki_bias: float = 0.003
    tilt_range: float = 4.0
    # association / memory
    mem_steps: int = 100  # a filter gate is trusted for 2 s after its last match
    tau0: float = 0.3
    tau_range: float = 0.08
    tau_place: float = 0.9  # + tau_place_range * range for gates predicted by the placement (list + fit)
    tau_place_range: float = 0.0  # (0.15 was tried: more wrong associations, next-pose wrong 5.1 -> 7.9 %)
    place_gain: float = 0.5
    kappa_yaw: float = 1.0  # m^2, weight of the gate yaws in the placement fit
    # weight (x quality-free 1.0) of REMEMBERED filter gates (matched within mem_steps, not this frame) in the
    # placement refit: with one gate in view the list placement's yaw then comes from the baseline to the gates
    # seen a moment ago instead of from the one gate's own yaw (randomised +-0.2 rad). 0 = v4.
    place_mem_w: float = 0.0
    # next / after-next rescue (v4): nearest unmatched detection within rescue_tol0 + rescue_tol_range * range
    rescue_tol0: float = 1.5
    rescue_tol_range: float = 0.2
    rescue_after_next: bool = True
    # lock / relocalisation
    reloc_margin: float = 1.0
    reloc_steps: int = 10
    neg_weight: float = 0.4
    neg_range: float = 6.0
    # next gate
    pass_window: float = 0.5
    geo_margin: float = 1.0
    geo_steps: int = 15
    # Acro tracks (a low gate almost beneath a high one): in the next-gate consistency rule the plane of a gate
    # more than geo_past_dz BELOW the drone does not count as "passed", so diving past a low gate's plane metres
    # above it does not skip that gate (gates above keep the plain test). inf = off (standard estimator, unchanged).
    geo_past_dz: float = math.inf


@struct.dataclass
class EstState:
    up: jax.Array  # (N, 3) world up in the body frame
    bias: jax.Array  # (N, 3) gyro bias estimate
    rate: jax.Array  # (N, 3) last bias-corrected rate
    x: jax.Array  # (N, 21) filter state [v, p_0..p_5] in H
    P: jax.Array  # (N, 21, 21)
    gy: jax.Array  # (N, 6) gate yaws (travel direction) in H
    gage: jax.Array  # (N, 6) int steps since the gate was matched
    pth: jax.Array  # (N,) placement yaw (list -> H)
    pt: jax.Array  # (N, 3) placement translation
    page: jax.Array  # (N,) int steps since the placement was fitted
    locked: jax.Array  # (N,) bool
    nxt: jax.Array  # (N,) int
    s_prev: jax.Array  # (N, 6) signed distance of the drone along each gate's travel direction
    geo_cnt: jax.Array  # (N,) int
    rel_cnt: jax.Array  # (N,) int
    rej_cnt: jax.Array  # (N, 6) int consecutive rejected measurements of a gate
    first: jax.Array  # (N,) bool: no frame processed yet


def est_init(n: int, cfg: "EstConfig" = None, up0=None) -> EstState:
    """Fresh estimator. ``up0`` (N, 3): initial up vector in the body frame, i.e. the attitude the flight
    controller carries into the episode (its IMU attitude since arming; episodes spawn in the air, where a
    hovering quadrotor's accelerometer cannot observe tilt). Default level."""
    cfg = cfg or EstConfig()
    z3, zi = jnp.zeros((n, 3)), jnp.zeros(n, jnp.int32)
    P0 = jnp.diag(jnp.concatenate([jnp.full(3, cfg.sig_v0**2), jnp.ones(3 * N_GATES), jnp.full(2, cfg.sig_tilt0**2)]))
    up = jnp.zeros((n, 3)).at[:, 2].set(1.0) if up0 is None else up0
    return EstState(
        up=up, bias=z3, rate=z3, x=jnp.zeros((n, NX)),
        P=jnp.broadcast_to(P0, (n, NX, NX)), gy=jnp.zeros((n, N_GATES)), gage=jnp.full((n, N_GATES), BIG_AGE, jnp.int32),
        pth=jnp.zeros(n), pt=z3, page=jnp.full(n, BIG_AGE, jnp.int32), locked=jnp.zeros(n, bool), nxt=zi,
        s_prev=jnp.zeros((n, N_GATES)), geo_cnt=zi, rel_cnt=zi, rej_cnt=jnp.zeros((n, N_GATES), jnp.int32),
        first=jnp.ones(n, bool),
    )  # fmt: skip


def est_reset_masked(state: EstState, mask, cfg: "EstConfig" = None, up0=None) -> EstState:
    new = est_init(mask.shape[0], cfg, up0)
    return jax.tree.map(lambda a, b: jnp.where(mask.reshape(-1, *[1] * (a.ndim - 1)), a, b), new, state)


# region small helpers


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotz(v, th):
    """Rotate vectors (..., 3) about z by th (...)."""
    c, s = jnp.cos(th), jnp.sin(th)
    x = c * v[..., 0] - s * v[..., 1]
    return jnp.stack([x, s * v[..., 0] + c * v[..., 1], jnp.broadcast_to(v[..., 2], x.shape)], -1)


def rotz_mat(th):
    c, s, z, o = jnp.cos(th), jnp.sin(th), jnp.zeros_like(th), jnp.ones_like(th)
    return jnp.stack([jnp.stack([c, -s, z], -1), jnp.stack([s, c, z], -1), jnp.stack([z, z, o], -1)], -2)


def tilt_matrix(up):
    """Tilt-only body -> H rotation (..., 3, 3) whose last row is ``up`` (= Ry(theta) Rx(phi), ZYX, zero yaw)."""
    u = up / jnp.linalg.norm(up, axis=-1, keepdims=True)
    st = -u[..., 0]
    ct = jnp.sqrt(jnp.maximum(1 - st**2, 1e-9))
    sp, cp = u[..., 1] / ct, u[..., 2] / ct
    z = jnp.zeros_like(st)
    return jnp.stack([jnp.stack([ct, st * sp, st * cp], -1), jnp.stack([z, cp, -sp], -1),
                      jnp.stack([-st, ct * sp, ct * cp], -1)], -2)  # fmt: skip


def tilt_only_from_quat(quat):
    """Tilt-only version of a true attitude (the teacher's R without its world yaw)."""
    from fpv.env import quat_to_mat

    return tilt_matrix(quat_to_mat(quat)[..., 2, :])


def _apply(R, v):
    return jnp.einsum("...ij,...j->...i", R, v)


def _apply_T(R, v):
    return jnp.einsum("...ji,...j->...i", R, v)


def _seg_dist(a, b):
    """Distance from the origin to segments a -> b (..., 3)."""
    d = b - a
    t = jnp.clip(-(a * d).sum(-1) / jnp.maximum((d * d).sum(-1), 1e-9), 0.0, 1.0)
    return jnp.linalg.norm(a + t[..., None] * d, axis=-1)


def _yaw_dist_pi(a, b):
    """|a - b| modulo pi, in [0, pi / 2]."""
    return jnp.abs(wrap(2 * (a - b))) / 2


def _resolve(yaw_mod_pi, ref):
    """The representative of yaw (defined modulo pi) nearest to ref."""
    return ref + wrap(2 * (yaw_mod_pi - ref)) / 2


def place(list_pos, list_yaw, th, t):
    """List gates under placement (th (...), t (..., 3)) -> centres (..., 6, 3), yaws (..., 6) in H."""
    return rotz(list_pos, th[..., None]) + t[..., None, :], list_yaw + th[..., None]


def _dir(yaw):
    return jnp.stack([jnp.cos(yaw), jnp.sin(yaw), jnp.zeros_like(yaw)], -1)


def _left(yaw):
    return jnp.stack([-jnp.sin(yaw), jnp.cos(yaw), jnp.zeros_like(yaw)], -1)


def signed_dist(c, yaw):
    """Drone (H origin) along the travel direction of a gate at c, yaw: < 0 before its plane."""
    return -(c * _dir(yaw)).sum(-1)


def in_camera(c_body, margin: float = 0.85):
    tan_h, tan_v = track.camera_half_fov_tan()
    f = c_body @ jnp.asarray(CAM_FWD)
    lft, up = c_body @ jnp.asarray(CAM_LEFT), c_body @ jnp.asarray(CAM_UP)
    return (f > 0.3) & (jnp.abs(lft) < margin * tan_h * f) & (jnp.abs(up) < margin * tan_v * f)


def _gate_xyz(x):
    return x[:, 3 : 3 + 3 * N_GATES].reshape(-1, N_GATES, 3)


def _skew(v):
    z = jnp.zeros_like(v[..., 0])
    return jnp.stack([jnp.stack([z, -v[..., 2], v[..., 1]], -1), jnp.stack([v[..., 2], z, -v[..., 0]], -1),
                      jnp.stack([-v[..., 1], v[..., 0], z], -1)], -2)  # fmt: skip


def _block_mask(gmask):
    """(N, 6) gate mask -> (N, NX) state-element mask."""
    zf = jnp.zeros((gmask.shape[0], 3), bool)
    return jnp.concatenate([zf, jnp.repeat(gmask, 3, axis=1), zf[:, :2]], 1)


def _reset_gates(x, P, gmask, z, Rz):
    """Set gates ``gmask`` (N, 6) to z (N, 6, 3) with covariance Rz (N, 6, 3, 3), uncorrelated."""
    m = _block_mask(gmask)
    x = jnp.where(m, jnp.concatenate([x[:, :3], z.reshape(-1, 3 * N_GATES), x[:, TL]], 1), x)
    keep = ~m
    P = P * (keep[:, :, None] & keep[:, None, :])
    eye6 = jnp.eye(N_GATES, dtype=bool)
    blk = jnp.where(gmask[:, :, None, None, None] & eye6[None, :, None, :, None], Rz[:, :, :, None, :], 0.0)
    blk = blk.reshape(-1, 3 * N_GATES, 3 * N_GATES)  # block-diagonal (N, 18, 18)
    P = P.at[:, 3 : 3 + 3 * N_GATES, 3 : 3 + 3 * N_GATES].add(blk)
    return x, P


def _meas_cov(z, dist, q, cfg):
    """Measurement covariance (..., 3, 3) of PnP centres z (..., 3) [H] at distance dist, quality q."""
    f = geometry.F_PX
    d = jnp.maximum(dist, 0.2)
    s_lat = jnp.maximum(cfg.sig_px * d / f, cfg.r_floor)
    # along the line of sight: PnP range sensitivity near, size-based range (~10 % relative) far
    s_rng = jnp.maximum(jnp.minimum(cfg.sig_px * d**2 / (f * track.GATE_OPENING), 0.1 * d), cfg.r_floor)
    los = z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-6)
    inflate = 1.0 / jnp.maximum(q, 0.2)
    R = (s_lat**2)[..., None, None] * jnp.eye(3) + (s_rng**2 - s_lat**2)[..., None, None] * los[..., :, None] * los[..., None, :]
    return R * inflate[..., None, None]


# region update


def _score(cfg, dc, dyaw, dval, anchor, th, t, list_pos, list_yaw, T, tau_a=0.5, tau_b=0.15, poses=None, wy_det=None):
    """Score placements th (N, M), t (N, M, 3) (or explicit gate ``poses`` = (pc (N, M, 6, 3), py (N, M, 6)))
    against the detections (dc (N, K, 3) H, dyaw (N, K) mod pi, dval (N, K)); inlier gate tau_a + tau_b *
    (lever arm from ``anchor`` (N, M, 3)). A gate predicted in the camera, 1-``neg_range`` m away, facing the
    camera and not behind a nearer detection, that no detection explains, costs ``neg_weight``.
    Returns score, inliers, hit (N, M, K, 6) nearest-gate inliers, predicted yaws (N, M, 6)."""
    pc, py = place(list_pos[:, None], list_yaw[:, None], th, t) if poses is None else poses  # (N, M, 6, 3)
    # height counts half: early in an episode the tilt (hence the height of far gates) is uncertain
    wyd = jnp.ones_like(dyaw) if wy_det is None else wy_det  # trust in each detection's yaw
    D = jnp.linalg.norm((dc[:, None, :, None] - pc[:, :, None]) * jnp.asarray([1.0, 1.0, 0.5]), axis=-1) + 0.5 * (
        wyd[:, None, :, None] * _yaw_dist_pi(dyaw[:, None, :, None], py[:, :, None]))  # (N, M, K, 6)  # fmt: skip
    lever = jnp.linalg.norm(dc[:, None] - anchor[:, :, None], axis=-1)  # (N, M, K)
    tau = tau_a + tau_b * lever
    nearest = D <= D.min(-1, keepdims=True)
    hit = (D < tau[..., None]) & dval[:, None, :, None] & nearest
    inl = hit.any(-1).sum(-1)
    explained = hit.any(-2)  # (N, M, 6)
    pb = _apply_T(T[:, None, None], pc)  # body frame
    pd = jnp.linalg.norm(pc, axis=-1)
    facing = jnp.abs((_dir(py) * pc).sum(-1)) / jnp.maximum(pd, 1e-6) > 0.3
    # a gate predicted behind a nearer detected gate (within its angular radius + 3 deg) may be occluded
    dd = jnp.linalg.norm(dc, axis=-1)  # (N, K)
    cosang = (pc[:, :, :, None] * dc[:, None, None]).sum(-1) / jnp.maximum(pd[..., None] * dd[:, None, None], 1e-6)
    ang_r = track.GATE_OUTER / 2 / jnp.maximum(dd, 0.1) + 0.05
    occl = (dval[:, None, None] & (dd[:, None, None] < pd[..., None] - 0.3)
            & (cosang > jnp.cos(ang_r)[:, None, None])).any(-1)  # fmt: skip
    expect = in_camera(pb) & (pd < cfg.neg_range) & (pd > 1.0) & facing & ~occl
    neg = (expect & ~explained).sum(-1)
    return inl - cfg.neg_weight * neg, inl, hit, py


def _kabsch(w, pL, yL, pH, yH, wy, kappa, th0, prior=0.05):
    """Weighted 2D rigid fit list -> H over pairs. w (..., P), pL/pH (..., P, 3), yL/yH (..., P) yaws,
    wy (..., P) trust in the measured yaws; th0 (...) prior yaw with weight ``prior`` (m^2), which decides
    when the pairs do not (one pair with an untrusted yaw). Returns th (...), t (..., 3)."""
    ws = jnp.maximum(w.sum(-1), 1e-9)
    mu_L = (w[..., None] * pL).sum(-2) / ws[..., None]
    mu_H = (w[..., None] * pH).sum(-2) / ws[..., None]
    xL, xH = pL - mu_L[..., None, :], pH - mu_H[..., None, :]
    dy = wrap(yH - yL)
    s_sin = (w * (xL[..., 0] * xH[..., 1] - xL[..., 1] * xH[..., 0] + kappa * wy * jnp.sin(dy))).sum(-1)
    s_cos = (w * (xL[..., 0] * xH[..., 0] + xL[..., 1] * xH[..., 1] + kappa * wy * jnp.cos(dy))).sum(-1)
    th = jnp.arctan2(s_sin + prior * jnp.sin(th0), s_cos + prior * jnp.cos(th0))
    return th, mu_H - rotz(mu_L, th)


def _geo_next(cfg, pc, py, T=None):
    """Which gate the drone is approaching: argmin cost (N,), cost (N, 6), with
    cost_j = 0.5 * distance + 3 [past gate j's plane] + 2 [gate j not in the camera] + 1 [not past gate j-1's plane].
    (Right after a pass the passed gate is behind and past: + 5; at a spawn in front of a gate that gate is
    near, in view and on its entry side.) ``T`` (N, 3, 3) body -> H for the camera test (None: heading only).
    Extra leading axes after N are allowed (pc (N, ..., 6, 3)); the result then has them too."""
    prev_c, prev_y = jnp.roll(pc, 1, axis=-2), jnp.roll(py, 1, axis=-1)
    if T is None:
        seen = pc[..., 0] > 0
    else:
        Tb = T.reshape(T.shape[0], *[1] * (pc.ndim - 2), 3, 3)
        seen = in_camera(_apply_T(Tb, pc), margin=1.0)
    if math.isinf(cfg.geo_past_dz):
        cost = (0.5 * jnp.linalg.norm(pc, axis=-1) + 3.0 * (signed_dist(pc, py) > 0.0) + 2.0 * ~seen
                + 1.0 * (signed_dist(prev_c, prev_y) < 0.0))  # fmt: skip
    else:
        past = (signed_dist(pc, py) > 0.0) & (pc[..., 2] > -cfg.geo_past_dz)
        prev_past = (signed_dist(prev_c, prev_y) >= 0.0) & (prev_c[..., 2] > -cfg.geo_past_dz)
        cost = 0.5 * jnp.linalg.norm(pc, axis=-1) + 3.0 * past + 2.0 * ~seen + 1.0 * ~prev_past
    return jnp.argmin(cost, axis=-1), cost


def _outputs(cfg, st: EstState, list_pos, list_yaw):
    """Current pose of every list gate in H (filter gate if matched recently, else placement) and whether
    it comes from the filter."""
    pc, py = place(list_pos, list_yaw, st.pth, st.pt)
    own = st.gage < cfg.mem_steps
    return jnp.where(own[..., None], _gate_xyz(st.x), pc), jnp.where(own, st.gy, py), own


def _hypotheses(cfg, dc, dyaw, dval, list_pos, list_yaw, T, anchor_ok, wpnp, ext=None, start_idx=None):
    """Best placement hypothesis: (th (N,), t (N, 3), score (N,), inliers (N,)). Each hypothesis (anchor
    detection k = list gate j, travel direction f) is scored loosely, refitted (Kabsch) to its inliers and
    rescored with a tight gate: the true identity explains every visible gate within the per-gate
    randomisation, a wrong one rarely does. ``ext`` = (centres (N, E, 3), yaws (N, E), valid (N, E)):
    remembered gates (filter tracks not seen in this frame) that are scored like detections, so gates
    seen a moment ago (e.g. the one just passed) add evidence; anchors are this frame's detections only."""
    n, kk = dval.shape
    th_h = wrap(dyaw[:, :, None, None] + math.pi * jnp.arange(2)[None, None, None] - list_yaw[:, None, :, None])
    th_h = th_h.reshape(n, -1)  # (N, K*6*2), order k, j, f
    lp = jnp.broadcast_to(list_pos[:, None, :, None, :], (n, kk, N_GATES, 2, 3)).reshape(n, -1, 3)
    anchor = jnp.broadcast_to(dc[:, :, None, None, :], (n, kk, N_GATES, 2, 3)).reshape(n, -1, 3)
    t_h = anchor - rotz(lp, th_h)
    if ext is not None:
        dc_s = jnp.concatenate([dc, ext[0]], 1)
        dyaw_s = jnp.concatenate([dyaw, ext[1]], 1)
        dval_s = jnp.concatenate([dval, ext[2]], 1)
        wy_s = jnp.concatenate([wpnp, 0.5 * ext[2]], 1)
    else:
        dc_s, dyaw_s, dval_s, wy_s = dc, dyaw, dval, wpnp
    ks = dval_s.shape[1]
    # two refits (Kabsch on the inlier pairs) with a gate that tightens: the anchor's PnP yaw can be off
    # by 0.2-0.3 rad with detector noise, which moves far gates by metres
    m = th_h.shape[1]
    pL = jnp.broadcast_to(list_pos[:, None, None], (n, m, ks, N_GATES, 3)).reshape(n, m, -1, 3)
    yL = jnp.broadcast_to(list_yaw[:, None, None], (n, m, ks, N_GATES)).reshape(n, m, -1)
    pH = jnp.broadcast_to(dc_s[:, None, :, None], (n, m, ks, N_GATES, 3)).reshape(n, m, -1, 3)
    wy = jnp.broadcast_to(wy_s[:, None, :, None], (n, m, ks, N_GATES)).reshape(n, m, -1)
    th_r, t_r = th_h, t_h
    for tau_b in (0.35, 0.2):
        _, _, hit, py = _score(cfg, dc_s, dyaw_s, dval_s, anchor, th_r, t_r, list_pos, list_yaw, T, 0.5, tau_b, wy_det=wy_s)
        w = hit.astype(jnp.float32).reshape(n, m, -1)
        yH = _resolve(dyaw_s[:, None, :, None], py[:, :, None, :]).reshape(n, m, -1)
        th_r, t_r = _kabsch(w, pL, yL, pH, yH, wy, cfg.kappa_yaw, th_h)
    # tight gate after the refit: per-gate randomisation (+-0.3 m, +-0.2 rad) still leaves ~0.1 rad of yaw
    s_h, inl, _, _ = _score(cfg, dc_s, dyaw_s, dval_s, anchor, th_r, t_r, list_pos, list_yaw, T, 0.6, 0.12, wy_det=wy_s)
    hval = jnp.broadcast_to((dval & anchor_ok)[:, :, None, None], (n, kk, N_GATES, 2)).reshape(n, -1)
    bonus = jnp.broadcast_to((jnp.arange(2) == 0)[None, None, None], (n, kk, N_GATES, 2)).reshape(n, -1)
    s_h = s_h + 0.2 * bonus
    if start_idx is not None:
        # known start (race start box): hypotheses whose approach rule names the announced first gate win
        pc_h, py_h = place(list_pos[:, None], list_yaw[:, None], th_r, t_r)
        n_h, _ = _geo_next(cfg, pc_h, py_h, T)
        s_h = s_h + jnp.where(start_idx[:, None] >= 0, 5.0 * (n_h == start_idx[:, None]), 0.0)
    s_h = jnp.where(hval, s_h, -1e3)
    hb = jnp.argmax(s_h, axis=1)

    def take(a):
        return jnp.take_along_axis(a, hb.reshape(n, 1, *[1] * (a.ndim - 2)), 1)[:, 0]

    return take(th_r), take(t_r), take(s_h), take(inl)


def _apply_lock(cfg, st: EstState, mask, th, t, list_pos, list_yaw, T):
    """Jump to placement (th, t) where mask: filter gates reset to it, nxt by the approach rule."""
    pc, py = place(list_pos, list_yaw, th, t)
    n_geo, _ = _geo_next(cfg, pc, py, T)
    x, P = _reset_gates(st.x, st.P, mask[:, None] & jnp.ones((1, N_GATES), bool), pc, jnp.broadcast_to(jnp.eye(3), pc.shape + (3,)))
    return st.replace(
        x=x, P=P, gy=jnp.where(mask[:, None], py, st.gy), pth=jnp.where(mask, th, st.pth),
        pt=jnp.where(mask[:, None], t, st.pt), page=jnp.where(mask, 0, st.page),
        gage=jnp.where(mask[:, None], BIG_AGE, st.gage), locked=st.locked | mask, rel_cnt=jnp.where(mask, 0, st.rel_cnt),
        nxt=jnp.where(mask, n_geo, st.nxt), s_prev=jnp.where(mask[:, None], signed_dist(pc, py), st.s_prev),
        geo_cnt=jnp.where(mask, 0, st.geo_cnt),
    )  # fmt: skip


def est_update(st: EstState, corners, conf, gyro, acc, list_pos, list_yaw, dt, cfg: EstConfig = EstConfig(),
               start_idx=None):  # fmt: skip
    """One step. dt (N,) seconds since the previous frame (0 on the first frame of an episode).
    Returns (state, info); info for checks: det (geometry outputs), matched (N, K, 6), passed, relock, ...
    Matrix products run at full float32 precision (GPU default may use TF32)."""
    with jax.default_matmul_precision("float32"):
        return _est_update(st, corners, conf, gyro, acc, list_pos, list_yaw, dt, cfg, start_idx)


def _est_update(st: EstState, corners, conf, gyro, acc, list_pos, list_yaw, dt, cfg: EstConfig, start_idx=None):
    """One step. dt (N,) seconds since the previous frame (0 on the first frame of an episode).
    Returns (state, info); info for checks: det (geometry outputs), matched (N, K, 6), passed, relock, ..."""
    n, kk = conf.shape
    dt = jnp.asarray(dt, jnp.float32) * jnp.ones(n)
    first = st.first
    # 1. attitude propagation
    w = gyro - st.bias
    Rb = geometry._rodrigues(w * dt[:, None])
    up0 = st.up
    up_p = _apply_T(Rb, up0)
    up_p = up_p / jnp.linalg.norm(up_p, axis=-1, keepdims=True)
    M = tilt_matrix(up0) @ Rb @ jnp.swapaxes(tilt_matrix(up_p), -1, -2)
    dpsi = jnp.arctan2(M[:, 1, 0], M[:, 0, 0])
    # 2. filter prediction (velocity + ego-motion of the gates) and placement ego-motion
    # f_H: specific force in H; its tilt error enters the velocity as -[f_H]x * delta (delta = tilt error,
    # rotation about H x / y: true H vectors = (I + [delta]x) estimated ones)
    f_H = _apply(tilt_matrix(up0 + up_p), acc)
    f_H = jnp.where((dt > 0)[:, None], f_H, jnp.asarray([0.0, 0.0, G]))
    acc_H = f_H - jnp.asarray([0.0, 0.0, G])
    Rz = rotz_mat(-dpsi)  # (N, 3, 3)
    v = st.x[:, :3]
    disp = (v + 0.5 * acc_H * dt[:, None]) * dt[:, None]
    gates = _gate_xyz(st.x)
    x = jnp.concatenate([_apply(Rz, v + acc_H * dt[:, None]),
                         _apply(Rz[:, None], gates - disp[:, None]).reshape(n, -1), jnp.zeros((n, 2))], 1)  # fmt: skip
    eye6 = jnp.eye(N_GATES)
    G3 = 3 * N_GATES
    Fx = _skew(f_H)[:, :, :2]  # (N, 3, 2)
    dt2 = dt[:, None, None]
    F = jnp.zeros((n, NX, NX))
    F = F.at[:, :3, :3].set(Rz)
    F = F.at[:, 3 : 3 + G3, 3 : 3 + G3].set(jnp.einsum("ab,nij->naibj", eye6, Rz).reshape(n, G3, G3))
    F = F.at[:, 3 : 3 + G3, :3].set(jnp.tile(-Rz * dt2, (1, N_GATES, 1)))
    F = F.at[:, :3, TL].set(-(Rz @ Fx) * dt2)
    F = F.at[:, 3 : 3 + G3, TL].set(jnp.tile(0.5 * (Rz @ Fx) * dt2**2, (1, N_GATES, 1)))
    F = F.at[:, TL, TL].set(Rz[:, :2, :2])
    gd2 = (gates**2).sum(-1)  # squared distances
    q_g = jnp.repeat((cfg.sig_rot * dt[:, None]) ** 2 * gd2 + 1e-6 * dt[:, None], 3, axis=1)
    q_v = jnp.broadcast_to(((cfg.sig_acc * dt) ** 2)[:, None], (n, 3))
    q_t = jnp.broadcast_to(((cfg.sig_tilt_rw * dt) ** 2)[:, None], (n, 2))
    P = F @ st.P @ jnp.swapaxes(F, -1, -2) + jax.vmap(jnp.diag)(jnp.concatenate([q_v, q_g, q_t], 1))
    gy = st.gy - dpsi[:, None]
    pt = rotz(st.pt - disp, -dpsi)
    pth = st.pth - dpsi
    # 3. detections
    det = geometry.gate_pose(corners, cfg.refine_iters)
    q = det["quality"] * (conf > cfg.conf_min)
    dval = q > cfg.q_min
    ddist = det["dist"]
    # tilt correction from gate up edges (in-plane axis closest to the current up), in-plane component only
    cand = jnp.stack([det["axis_r"], -det["axis_r"], det["axis_d"], -det["axis_d"]], -2)  # (N, K, 4, 3)
    best = jnp.argmax((cand * up_p[:, None, None]).sum(-1), -1)
    up_m = jnp.take_along_axis(cand, best[..., None, None], -2)[..., 0, :]
    nb = det["normal"]
    delta = up_m - up_p[:, None]
    delta = delta - (delta * nb).sum(-1, keepdims=True) * nb
    wpnp = det["w_pnp"]
    wt = q * dval * (ddist < cfg.tilt_range) * wpnp
    wsum = wt.sum(1)
    dm = (wt[..., None] * delta).sum(1) / jnp.maximum(wsum, 1e-9)[:, None]
    um_full = (wt[..., None] * up_m).sum(1) / jnp.maximum(wsum, 1e-9)[:, None]
    has_t = wsum > 0
    up = up_p + cfg.kp_tilt * jnp.minimum(wsum, 1.0)[:, None] * dm
    up = up / jnp.linalg.norm(up, axis=-1, keepdims=True)
    um = up_p + dm
    um = um / jnp.linalg.norm(um, axis=-1, keepdims=True)
    bias = st.bias - jnp.where((has_t & ~first)[:, None], cfg.ki_bias * jnp.minimum(wsum, 1.0)[:, None]
                               * jnp.cross(um, up_p), 0.0)  # fmt: skip
    T = tilt_matrix(up)
    dc = _apply(T[:, None], det["centre"])  # (N, K, 3) in H
    nH = _apply(T[:, None], nb)
    dyaw = jnp.arctan2(nH[..., 1], nH[..., 0])  # away from the camera; travel direction is this or + pi
    st = st.replace(up=up, bias=bias, rate=w, x=x, P=P, gy=gy, pt=pt, pth=pth)

    # 4. first lock (before association, so the first frame already has an estimate)
    # remembered gates not seen in the last frame join the scoring as extra evidence
    ext = (_gate_xyz(st.x), st.gy, (st.gage < cfg.mem_steps) & (st.gage >= 1) & st.locked[:, None])
    s_idx = None if start_idx is None else jnp.where(st.locked, -1, start_idx)
    th_b, t_b, s_best, inl_best = _hypotheses(cfg, dc, dyaw, dval, list_pos, list_yaw, T, wpnp > 0.3, wpnp,
                                              ext=ext, start_idx=s_idx)  # fmt: skip
    lock_now = ~st.locked & (s_best >= 0.9)
    st = _apply_lock(cfg, st, lock_now, th_b, t_b, list_pos, list_yaw, T)

    # 5. association
    jj = jnp.arange(N_GATES)
    Pc, Py, own = _outputs(cfg, st, list_pos, list_yaw)
    D = jnp.linalg.norm(dc[:, :, None] - Pc[:, None], axis=-1) + 0.5 * _yaw_dist_pi(dyaw[:, :, None], Py[:, None])
    Pjj_tr = jnp.stack([jnp.trace(st.P[:, 3 + 3 * j : 6 + 3 * j, 3 + 3 * j : 6 + 3 * j], axis1=-2, axis2=-1)
                        for j in range(N_GATES)], 1)  # fmt: skip
    sig_g = jnp.sqrt(jnp.maximum(Pjj_tr, 0) / 3)
    tau = cfg.tau0 + cfg.tau_range * ddist[:, :, None] + jnp.where(
        own[:, None], 3 * sig_g[:, None], cfg.tau_place + cfg.tau_place_range * ddist[:, :, None])  # fmt: skip
    ok = (D < tau) & dval[:, :, None] & st.locked[:, None, None]
    Dm = jnp.where(ok, D, 1e9)
    j_of_k = jnp.argmin(Dm, axis=2)  # (N, K)
    k_of_j = jnp.argmin(Dm, axis=1)  # (N, 6)
    matched = ok & (j_of_k[:, :, None] == jj) & (k_of_j[:, None, :] == jnp.arange(kk)[None, :, None])
    # rescue for the gates that matter most (next and after-next): if one is not matched and not seen for
    # > 2 frames, the nearest unmatched good detection within 1.5 m + 0.2 * range of its prediction is taken
    # as it (re-initialising its track). Right after a pass the new next gate is usually only predicted by the
    # placement (list geometry, off by the gate randomisation and the placement yaw error times distance).
    unm = dval & ~matched.any(2)
    rescued = jnp.zeros_like(own)
    for k_off in (0, 1) if cfg.rescue_after_next else (0,):
        j = (st.nxt + k_off) % N_GATES
        oh_j = jax.nn.one_hot(j, N_GATES, dtype=bool)  # (N, 6)
        need = st.locked & ~(matched.any(1) & oh_j).any(1) & (jnp.take_along_axis(st.gage, j[:, None], 1)[:, 0] > 2)
        Dj = jnp.take_along_axis(D, j[:, None, None], 2)[..., 0]  # (N, K)
        cand = unm & (Dj < cfg.rescue_tol0 + cfg.rescue_tol_range * ddist)
        kb = jnp.argmin(jnp.where(cand, Dj, 1e9), axis=1)
        do = need & cand.any(1)
        oh_k = jax.nn.one_hot(kb, kk, dtype=bool) & do[:, None]
        matched = matched | (oh_k[:, :, None] & oh_j[:, None, :])
        unm = unm & ~oh_k
        rescued = rescued | (do[:, None] & oh_j)
    mg = matched.any(1)  # (N, 6)
    mf = matched.astype(jnp.float32)
    c_m = jnp.einsum("nkj,nki->nji", mf, dc)
    yaw_m = _resolve(jnp.einsum("nkj,nk->nj", mf, dyaw), Py)
    q_m = jnp.einsum("nkj,nk->nj", mf, q)
    d_m = jnp.einsum("nkj,nk->nj", mf, ddist)
    w_m = jnp.einsum("nkj,nk->nj", mf, wpnp)  # trust in the measured yaw
    Rm = _meas_cov(c_m, d_m, q_m, cfg)  # (N, 6, 3, 3)
    # 6. filter update: new gates initialised at the measurement, tracked gates sequential KF updates
    init = mg & (~own | rescued)
    x, P = _reset_gates(st.x, st.P, init, c_m, Rm)
    upd = mg & own & ~rescued
    rejected = jnp.zeros_like(mg)
    for j in range(N_GATES):
        sl = slice(3 + 3 * j, 6 + 3 * j)
        # measured (estimated-H) centre = (I - [delta]x) p = p + [p]x delta
        Hm = jnp.zeros((n, 3, NX)).at[:, :, sl].set(jnp.eye(3)).at[:, :, TL].set(_skew(x[:, sl])[:, :, :2])
        PHt = P @ jnp.swapaxes(Hm, -1, -2)  # (N, NX, 3)
        S = Hm @ PHt + Rm[:, j]
        K = jnp.swapaxes(jnp.linalg.solve(S, jnp.swapaxes(PHt, -1, -2)), -1, -2)  # (N, NX, 3), S symmetric
        r = c_m[:, j] - x[:, sl]
        maha = (r * jnp.linalg.solve(S, r[..., None])[..., 0]).sum(-1)
        u = upd[:, j] & (maha < cfg.chi2_gate)  # outlier fits (e.g. a corner off by 10+ px) are not used
        rejected = rejected.at[:, j].set(upd[:, j] & ~u)
        x = jnp.where(u[:, None], x + jnp.einsum("nij,nj->ni", K, r), x)
        P = jnp.where(u[:, None, None], P - K @ (Hm @ P), P)
    P = 0.5 * (P + jnp.swapaxes(P, -1, -2))
    # a gate rejected 3 frames in a row is re-initialised at the measurement (the track was wrong)
    rej_cnt = jnp.where(rejected, st.rej_cnt + 1, jnp.where(mg, 0, st.rej_cnt))
    reinit = rejected & (rej_cnt >= 3)
    x, P = _reset_gates(x, P, reinit, c_m, Rm)
    rej_cnt = jnp.where(reinit, 0, rej_cnt)
    # fold the estimated tilt error into the attitude: T_true = (I + [delta]x) T  ->  u = T^T (e_z - delta x e_z)
    ez = jnp.asarray([0.0, 0.0, 1.0])
    dl = jnp.concatenate([x[:, TL], jnp.zeros((n, 1))], 1)
    up_f = _apply_T(tilt_matrix(st.up), ez - jnp.cross(dl, ez))
    up_f = up_f / jnp.linalg.norm(up_f, axis=-1, keepdims=True)
    x = x.at[:, TL].set(0.0)
    good = mg & (~rejected | reinit)
    new_y = jnp.where(own & ~reinit, Py + cfg.alpha_yaw * w_m * wrap(yaw_m - Py),
                      jnp.where(w_m > 0.3, yaw_m, Py))  # fmt: skip
    gy = jnp.where(good, new_y, jnp.where(own, st.gy, Py))
    gage = jnp.where(good, 0, jnp.minimum(st.gage + 1, BIG_AGE))
    # placement refit (weighted 2D Kabsch on centres + gate yaws), on the filtered gate positions
    gx = _gate_xyz(x)
    rem = (gage < cfg.mem_steps) & ~good & st.locked[:, None]
    pw = good * q_m + cfg.place_mem_w * rem
    psum = pw.sum(1)
    has_p = psum > 0
    mu_L = (pw[..., None] * list_pos).sum(1) / jnp.maximum(psum, 1e-9)[:, None]
    mu_H = (pw[..., None] * gx).sum(1) / jnp.maximum(psum, 1e-9)[:, None]
    xL, xH = list_pos - mu_L[:, None], gx - mu_H[:, None]
    dy = wrap(gy - list_yaw)
    s_sin = (pw * (xL[..., 0] * xH[..., 1] - xL[..., 1] * xH[..., 0] + cfg.kappa_yaw * w_m * jnp.sin(dy))).sum(1)
    s_cos = (pw * (xL[..., 0] * xH[..., 0] + xL[..., 1] * xH[..., 1] + cfg.kappa_yaw * w_m * jnp.cos(dy))).sum(1)
    th_fit = jnp.arctan2(s_sin, s_cos)
    t_fit = mu_H - rotz(mu_L, th_fit)
    pth = jnp.where(has_p, st.pth + cfg.place_gain * wrap(th_fit - st.pth), st.pth)
    pt = jnp.where(has_p[:, None], st.pt + cfg.place_gain * (t_fit - st.pt), st.pt)
    page = jnp.where((good * q_m).sum(1) > 0, 0, jnp.minimum(st.page + 1, BIG_AGE))  # age of measured evidence
    # gates no longer trusted: decorrelate them from the velocity (keeps the filter well conditioned)
    stale = (gage == cfg.mem_steps)
    pc_place, _ = place(list_pos, list_yaw, pth, pt)
    x, P = _reset_gates(x, P, stale, pc_place, jnp.broadcast_to(jnp.eye(3), pc_place.shape + (3,)))
    st = st.replace(up=up_f, x=x, P=P, gy=gy, gage=gage, pth=pth, pt=pt, page=page, rej_cnt=rej_cnt)

    # 7. relocalisation (after the update; the jump takes effect for this frame's outputs)
    near_good = (dval & (ddist < cfg.neg_range) & (q > 0.2)).any(1)
    n_matched = mg.sum(1)
    # current belief scored like a hypothesis, on its own gate poses (filter gates where tracked)
    Pc0, Py0, _ = _outputs(cfg, st, list_pos, list_yaw)
    s_cur, _, _, _ = _score(cfg, jnp.concatenate([dc, ext[0]], 1), jnp.concatenate([dyaw, ext[1]], 1),
                            jnp.concatenate([dval, ext[2]], 1), jnp.zeros((n, 1, 3)), None, None, list_pos, list_yaw,
                            T, 0.6, 0.12, poses=(Pc0[:, None], Py0[:, None]),
                            wy_det=jnp.concatenate([wpnp, 0.5 * ext[2]], 1))  # fmt: skip
    s_cur = s_cur[:, 0]
    bad = st.locked & ~lock_now & ((near_good & (n_matched == 0)) | (s_best >= s_cur + cfg.reloc_margin))
    rel_cnt = jnp.where(bad, st.rel_cnt + 1, 0)
    relock = (rel_cnt >= cfg.reloc_steps) & (s_best >= 0.9)
    st = st.replace(rel_cnt=rel_cnt)
    st = _apply_lock(cfg, st, relock, th_b, t_b, list_pos, list_yaw, T)

    # 8. next gate
    Pc, Py, own = _outputs(cfg, st, list_pos, list_yaw)
    n_geo, cost = _geo_next(cfg, Pc, Py, T)
    # a) flown through a TRACKED gate (matched within mem_steps): plane crossed inside the pass window ->
    #    the next gate is the one after it (any tracked gate, so a missed pass or a wrong belief is fixed by
    #    the next gate actually flown through; untracked gates, i.e. placement predictions, never count)
    s_all = signed_dist(Pc, Py)  # (N, 6)
    lat = jnp.abs((Pc * _left(Py)).sum(-1))
    fresh_lock = lock_now | relock
    thru = (own & (st.s_prev < 0) & (s_all >= 0) & (lat < cfg.pass_window) & (jnp.abs(Pc[..., 2]) < cfg.pass_window)
            & (st.locked & ~fresh_lock)[:, None])  # fmt: skip
    passed = thru.any(1)
    j_thru = jnp.argmax(thru, axis=1)
    nxt = jnp.where(passed, (j_thru + 1) % N_GATES, st.nxt)
    c_nxt = jnp.take_along_axis(cost, nxt[:, None], 1)[:, 0]
    c_geo = jnp.take_along_axis(cost, n_geo[:, None], 1)[:, 0]
    disagree = st.locked & (n_geo != nxt) & (c_nxt - c_geo > cfg.geo_margin)
    geo_cnt = jnp.where(disagree, st.geo_cnt + 1, 0)
    jump = geo_cnt >= cfg.geo_steps
    nxt = jnp.where(jump, n_geo, nxt)
    geo_cnt = jnp.where(jump, 0, geo_cnt)
    st = st.replace(nxt=nxt, s_prev=s_all, geo_cnt=geo_cnt, first=jnp.zeros_like(first))
    info = dict(det=det, q=q, valid=dval, matched=matched, passed=passed, relock=relock, lock=lock_now,
                n_geo=n_geo, geo_jump=jump, s_best=s_best)  # fmt: skip
    return st, info


# region observation


def gate_estimates(st: EstState, list_pos, list_yaw, cfg: EstConfig = EstConfig()):
    """Estimated centres (N, 6, 3) and yaws (N, 6) of every list gate in H, own (N, 6) bool, T (N, 3, 3)."""
    Pc, Py, own = _outputs(cfg, st, list_pos, list_yaw)
    return Pc, Py, own, tilt_matrix(st.up)


def velocity_body(st: EstState):
    return _apply_T(tilt_matrix(st.up), st.x[:, :3])


def observation(st: EstState, list_pos, list_yaw, prev_action, cfg: EstConfig = EstConfig()):
    """Policy observation (N, 46): teacher layout with estimates + [age next, age after-next, locked]."""
    Pc, Py, own, T = gate_estimates(st, list_pos, list_yaw, cfg)
    v_b = _apply_T(T, st.x[:, :3])
    parts = [v_b / VEL_SCALE, T.reshape(-1, 9), st.rate / jnp.asarray([MAX_RATE_XY, MAX_RATE_XY, MAX_RATE_Z])]
    ages = []
    for k in range(2):
        idx = (st.nxt + k) % N_GATES
        c = jnp.take_along_axis(Pc, idx[:, None, None], 1)[:, 0]
        y = jnp.take_along_axis(Py, idx[:, None], 1)[:, 0]
        corners_b = _apply_T(T[:, None], track.gate_corners(c, y))  # drone at the H origin
        parts.append(jnp.where(st.locked[:, None], corners_b.reshape(-1, 12) / POS_SCALE, 0.0))
        o = jnp.take_along_axis(own, idx[:, None], 1)[:, 0]
        a = jnp.where(o, jnp.take_along_axis(st.gage, idx[:, None], 1)[:, 0], cfg.mem_steps + st.page)
        ages.append(jnp.where(st.locked, jnp.clip(a / (2.0 * cfg.mem_steps), 0, 1), 1.0))
    parts.append(prev_action)
    parts.append(jnp.stack([*ages, st.locked.astype(jnp.float32)], -1))
    return jnp.nan_to_num(jnp.concatenate(parts, -1))
