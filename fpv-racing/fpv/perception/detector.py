"""Gate detector: one 320x180 RGB frame -> the opening corners of up to 6 gates + a confidence each.

    det = load_detector("runs/percep/detector/<run>/ckpt.pkl")
    out = det(frames_uint8)          # frames (N, 180, 320, 3) uint8 RGB, jax or numpy
    out["corners"]                   # (N, 6, 4, 2) pixels, contract image order (fpv.perception.labels)
    out["conf"]                      # (N, 6) in [0, 1]; slots unordered, keep conf >= threshold

Formulation (CenterNet-style, anchor-free, one stage): a small residual CNN (strides 2..32) with a
top-down feature pyramid down to stride 4 (48 x 80 cells; the 180-row frame is padded to 192 rows).
Heads at stride 4:
  * gate-centre heatmap (1): Gaussian at the projected 3D gate centre (clamped into the image for
    partly visible gates), focal loss;
  * corner regression (8): the 4 opening corners relative to the centre cell, loss = min over the 4
    cyclic shifts (so the corner the image-order rule starts at may flip without penalty);
  * corner heatmap (1) + sub-cell offset (2): every opening corner inside the image, class-agnostic.
Decoding: the 6 highest local maxima (3x3) of the centre heatmap give the slots and confidences;
each regressed corner is replaced by the nearest corner-heatmap peak within a radius ("snapping",
``snap=True``); the quad is then put into the contract image order.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fpv.perception import labels as L

K_SLOTS = 6
PAD_TOP = 6  # 180 rows + 6 + 6 = 192, divisible by 32


@dataclasses.dataclass(frozen=True)
class DetectorConfig:
    widths: tuple = (24, 48, 96, 160, 224)  # channels at strides 2, 4, 8, 16, 32
    blocks: tuple = (1, 1, 2, 2, 2)  # residual blocks per stage
    fpn: int = 64
    stride: int = 4  # output stride (4 or 2)
    reg_scale: float = 32.0  # px per regression unit
    snap: bool = True
    snap_frac: float = 0.25  # snapping radius as a fraction of the quad side (px), at least snap_min px
    snap_min: float = 3.0
    snap_thresh: float = 0.2  # corner heatmap score needed to snap
    corner_sigma: float = 0.6  # cells
    centre_sigma_frac: float = 0.08  # centre Gaussian sigma = this * side / stride, clipped to [0.7, 4] cells
    w_reg: float = 1.0
    w_corner: float = 1.0
    w_off: float = 1.0


def grid_shape(cfg: DetectorConfig, height: int = L.H, width: int = L.W):
    return (height + 2 * PAD_TOP) // cfg.stride, width // cfg.stride


def cell_centres(cfg: DetectorConfig, height: int = L.H, width: int = L.W):
    """Pixel coordinates (u, v) of cell centres, (Hc, Wc, 2)."""
    hc, wc = grid_shape(cfg, height, width)
    s = cfg.stride
    u = jnp.arange(wc) * s + (s - 1) / 2
    v = jnp.arange(hc) * s + (s - 1) / 2 - PAD_TOP
    return jnp.stack(jnp.meshgrid(u, v), -1)


def to_grid(cfg: DetectorConfig, uv):
    """Pixels (..., 2) -> continuous cell coordinates (..., 2) (x = column, y = row)."""
    s = cfg.stride
    return jnp.stack([(uv[..., 0] - (s - 1) / 2) / s, (uv[..., 1] + PAD_TOP - (s - 1) / 2) / s], -1)


class _Res(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x):
        y = nn.silu(nn.GroupNorm(8)(nn.Conv(self.ch, (3, 3))(x)))
        y = nn.GroupNorm(8)(nn.Conv(self.ch, (3, 3))(y))
        return nn.silu(x + y)


class GateDetector(nn.Module):
    cfg: DetectorConfig

    @nn.compact
    def __call__(self, frames):
        c = self.cfg
        x = frames.astype(jnp.float32) / 255.0 - 0.5
        x = jnp.pad(x, ((0, 0), (PAD_TOP, PAD_TOP), (0, 0), (0, 0)))
        feats = []
        for ch, nb in zip(c.widths, c.blocks):
            x = nn.silu(nn.GroupNorm(8)(nn.Conv(ch, (3, 3), strides=2)(x)))
            for _ in range(nb):
                x = _Res(ch)(x)
            feats.append(x)
        n_up = {4: 3, 2: 4}[c.stride]  # from stride 32 down to the output stride
        y = nn.Conv(c.fpn, (1, 1))(feats[-1])
        for f in feats[-2 : -2 - n_up : -1]:
            y = jax.image.resize(y, (y.shape[0], f.shape[1], f.shape[2], y.shape[3]), "nearest")
            y = nn.silu(nn.GroupNorm(8)(nn.Conv(c.fpn, (3, 3))(y + nn.Conv(c.fpn, (1, 1))(f))))
        y = nn.silu(nn.Conv(c.fpn, (3, 3))(y))
        prior = float(np.log(0.01 / 0.99))
        hm_c = nn.Conv(1, (1, 1), bias_init=nn.initializers.constant(prior))(y)[..., 0]
        reg = nn.Conv(8, (1, 1))(nn.silu(nn.Conv(c.fpn, (3, 3))(y)))
        hm_k = nn.Conv(1, (1, 1), bias_init=nn.initializers.constant(prior))(y)[..., 0]
        off = nn.Conv(2, (1, 1))(y)
        return dict(hm_c=hm_c, reg=reg, hm_k=hm_k, off=off)


def n_params(params) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree.leaves(params)))


# region targets and loss


def _gauss(grid_xy, pts, sigma):
    """grid (Hc, Wc, 2), pts (N, P, 2), sigma (N, P) -> (N, P, Hc, Wc)."""
    d2 = ((grid_xy[None, None] - pts[:, :, None, None, :]) ** 2).sum(-1)
    return jnp.exp(-d2 / (2 * sigma[..., None, None] ** 2))


def make_targets(cfg: DetectorConfig, lab: dict, height: int = L.H, width: int = L.W) -> dict:
    """Training targets from `labels.gate_labels` output (jax arrays, leading N)."""
    hc, wc = grid_shape(cfg, height, width)
    gy, gx = jnp.meshgrid(jnp.arange(hc), jnp.arange(wc), indexing="ij")
    grid = jnp.stack([gx, gy], -1).astype(jnp.float32)
    vis, dc = lab["visible"], lab["dontcare"]
    ctr = jnp.stack([jnp.clip(lab["centre"][..., 0], 0, width - 1), jnp.clip(lab["centre"][..., 1], 0, height - 1)], -1)
    # for visible gates whose 3D centre projects badly (behind / far outside) use the corners' mean
    ctr = jnp.where((lab["centre_cam"][..., 2:3] > 0.05), ctr, lab["corners"].mean(-2))
    ctr = jnp.stack([jnp.clip(ctr[..., 0], 0, width - 1), jnp.clip(ctr[..., 1], 0, height - 1)], -1)
    cg = jnp.round(to_grid(cfg, ctr))
    cg = jnp.stack([jnp.clip(cg[..., 0], 0, wc - 1), jnp.clip(cg[..., 1], 0, hc - 1)], -1)
    sig = jnp.clip(cfg.centre_sigma_frac * lab["side_px"] / cfg.stride, 0.7, 4.0)
    g = _gauss(grid, cg, sig)  # (N, G, Hc, Wc)
    hm_c = jnp.where(vis[..., None, None], g, 0.0).max(1)
    ignore = jnp.where(dc[..., None, None], g, 0.0).max(1) > 0.1
    w_c = jnp.where(ignore & (hm_c < 0.5), 0.0, 1.0)
    cell = (cg[..., 1] * wc + cg[..., 0]).astype(jnp.int32)  # (N, G)
    cc = cell_centres(cfg, height, width).reshape(-1, 2)[cell]  # (N, G, 2)
    reg = (lab["corners"] - cc[..., None, :]) / cfg.reg_scale  # (N, G, 4, 2)
    # corners: every opening corner inside the image of a visible or don't-care gate in front
    k_uv = lab["corners"].reshape(lab["corners"].shape[0], -1, 2)
    k_ok = (L.in_image(lab["corners"], width, height) & (vis | dc)[..., None]).reshape(k_uv.shape[0], -1)
    k_g = to_grid(cfg, k_uv)
    k_cell = jnp.round(k_g)
    gk = _gauss(grid, k_cell, jnp.full(k_ok.shape, cfg.corner_sigma))
    hm_k = jnp.where(k_ok[..., None, None], gk, 0.0).max(1)
    k_idx = (jnp.clip(k_cell[..., 1], 0, hc - 1) * wc + jnp.clip(k_cell[..., 0], 0, wc - 1)).astype(jnp.int32)
    return dict(hm_c=hm_c, w_c=w_c, cell=cell, reg=reg, reg_mask=vis, side=lab["side_px"], hm_k=hm_k,
                k_idx=k_idx, k_off=k_g - k_cell, k_mask=k_ok)  # fmt: skip


def _focal(logit, target, weight):
    p = jax.nn.sigmoid(logit)
    pos = target >= 0.999
    lp, ln = jax.nn.log_sigmoid(logit), jax.nn.log_sigmoid(-logit)
    loss = jnp.where(pos, -((1 - p) ** 2) * lp, -((1 - target) ** 4) * p**2 * ln) * weight
    return loss.sum() / jnp.maximum(pos.sum(), 1.0)


def _gather(x, idx):
    """x (N, Hc, Wc, C), idx (N, P) flat cell -> (N, P, C)."""
    n, hc, wc, c = x.shape
    return jnp.take_along_axis(x.reshape(n, hc * wc, c), idx[..., None], 1)


def loss_fn(cfg: DetectorConfig, out: dict, tgt: dict):
    l_c = _focal(out["hm_c"], tgt["hm_c"], tgt["w_c"])
    l_k = _focal(out["hm_k"], tgt["hm_k"], 1.0)
    pr = _gather(out["reg"], tgt["cell"]).reshape(*tgt["cell"].shape, 4, 2)  # (N, G, 4, 2)
    err = jnp.stack([jnp.abs(jnp.roll(tgt["reg"], s, -2) - pr).sum((-1, -2)) for s in range(4)], -1).min(-1)
    err_px = err * cfg.reg_scale / 4  # mean L1 per corner, pixels
    rel = err_px / (8.0 + 0.1 * tgt["side"])
    m = tgt["reg_mask"]
    l_r = jnp.where(m, rel, 0.0).sum() / jnp.maximum(m.sum(), 1.0)
    po = _gather(out["off"], tgt["k_idx"])
    km = tgt["k_mask"]
    l_o = jnp.where(km, jnp.abs(po - tgt["k_off"]).sum(-1), 0.0).sum() / jnp.maximum(km.sum(), 1.0)
    total = l_c + cfg.w_corner * l_k + cfg.w_reg * l_r + cfg.w_off * l_o
    return total, dict(loss=total, centre=l_c, corner_hm=l_k, reg=l_r, reg_px=jnp.where(m, err_px, 0).sum()
                       / jnp.maximum(m.sum(), 1.0), off=l_o)  # fmt: skip


# region decoding


def _nms(h):
    mx = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 3, 3), (1, 1, 1), "SAME")
    return jnp.where(h >= mx, h, 0.0)


def decode(cfg: DetectorConfig, out: dict, k: int = K_SLOTS, n_corner_peaks: int = 48, snap: bool | None = None):
    """Network outputs -> dict(corners (N, k, 4, 2) px in contract order, conf (N, k),
    corners_reg (N, k, 4, 2) before snapping, snapped (N, k, 4) bool)."""
    snap = cfg.snap if snap is None else snap
    n, hc, wc = out["hm_c"].shape
    cc = cell_centres(cfg).reshape(-1, 2)
    hc_ = _nms(jax.nn.sigmoid(out["hm_c"])).reshape(n, -1)
    conf, idx = jax.lax.top_k(hc_, k)
    reg = _gather(out["reg"], idx).reshape(n, k, 4, 2) * cfg.reg_scale + cc[idx][:, :, None, :]
    corners = reg
    snapped = jnp.zeros((n, k, 4), bool)
    if snap:
        hk = _nms(jax.nn.sigmoid(out["hm_k"])).reshape(n, -1)
        ks, kidx = jax.lax.top_k(hk, n_corner_peaks)
        kxy = cc[kidx] + _gather(out["off"], kidx) * cfg.stride  # (N, M, 2)
        d = jnp.sqrt(((reg[:, :, :, None, :] - kxy[:, None, None]) ** 2).sum(-1))  # (N, k, 4, M)
        d = jnp.where(ks[:, None, None, :] >= cfg.snap_thresh, d, jnp.inf)
        j = jnp.argmin(d, -1)
        dmin = jnp.take_along_axis(d, j[..., None], -1)[..., 0]
        x, y = reg[..., 0], reg[..., 1]
        side = jnp.sqrt(0.5 * jnp.abs((x * jnp.roll(y, -1, -1) - jnp.roll(x, -1, -1) * y).sum(-1)))  # (N, k)
        rad = jnp.maximum(cfg.snap_min, cfg.snap_frac * side)[..., None]
        snapped = dmin < rad
        tgt = jnp.take_along_axis(kxy, j.reshape(n, k * 4)[..., None], 1).reshape(n, k, 4, 2)
        corners = jnp.where(snapped[..., None], tgt, reg)
    return dict(corners=L.canonical_order(corners), conf=conf, corners_reg=L.canonical_order(reg), snapped=snapped)

def match(corners, conf, lab: dict, thresh: float, gate_px: float = 5.0, gate_frac: float = 0.5) -> dict:
    """Greedy one-to-one matching of detections to labelled gates (numpy).

    corners (N, K, 4, 2), conf (N, K); lab = `labels.gate_labels` output as numpy (N, G, ...).
    Detections with conf >= thresh are matched to visible gates by the mean corner distance under
    the best cyclic shift, accepted below max(gate_px, gate_frac * side_px); left-over detections
    matching a don't-care gate the same way are neutral, the rest are false detections.
    Returns: det_of_gate (N, G) detection index or -1, err (N, G) mean corner error in contract order
    [px], err_inv (N, G) the same under the best cyclic shift, err_vec (N, G, 4, 2) signed corner
    errors (prediction - truth, contract order), fp (N, K) bool, kept (N, K) bool.
    """
    n, k = conf.shape
    g = lab["corners"].shape[1]
    gt = lab["corners"]
    kept = conf >= thresh
    d_rot = np.stack([np.linalg.norm(np.roll(corners, s, -2)[:, None] - gt[:, :, None], axis=-1).mean(-1)
                      for s in range(4)], -1)  # (N, G, K, 4)  # fmt: skip
    cost = d_rot.min(-1)
    lim = np.maximum(gate_px, gate_frac * lab["side_px"])[:, :, None]
    ok = (cost < lim) & kept[:, None, :]
    det_of_gate = np.full((n, g), -1)
    used = np.zeros((n, k), bool)
    for which, mask in ((0, lab["visible"]), (1, lab["dontcare"] & ~lab["visible"])):
        c = np.where(ok & mask[:, :, None] & ~used[:, None, :], cost, np.inf)
        taken_g = np.zeros((n, g), bool)
        for _ in range(min(g, k)):
            flat = c.reshape(n, -1)
            j = flat.argmin(-1)
            best = flat[np.arange(n), j]
            gi, ki = j // k, j % k
            hit = np.isfinite(best)
            r = np.flatnonzero(hit)
            if which == 0:
                det_of_gate[r, gi[r]] = ki[r]
            used[r, ki[r]] = True
            taken_g[r, gi[r]] = True
            c[r, gi[r], :] = np.inf
            c[r, :, ki[r]] = np.inf
            if not hit.any():
                break
    sel = np.clip(det_of_gate, 0, k - 1)
    pred = np.take_along_axis(corners, sel[:, :, None, None], 1)  # (N, G, 4, 2)
    err_vec = pred - gt
    err = np.linalg.norm(err_vec, axis=-1).mean(-1)
    err_inv = np.take_along_axis(d_rot, sel[:, :, None, None], 2)[:, :, 0].min(-1)
    m = det_of_gate >= 0
    return dict(det_of_gate=det_of_gate, err=np.where(m, err, np.nan), err_inv=np.where(m, err_inv, np.nan),
                err_vec=np.where(m[..., None, None], err_vec, np.nan), fp=kept & ~used, kept=kept)  # fmt: skip



# region training / checkpoint


def init_params(cfg: DetectorConfig, seed: int = 0):
    net = GateDetector(cfg)
    return net, net.init(jax.random.key(seed), jnp.zeros((1, L.H, L.W, 3), jnp.uint8))


def make_optimizer(total_steps: int, lr: float = 2e-3, warmup: int = 500, wd: float = 1e-4):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(total_steps, warmup + 1), lr * 0.02)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=wd))


def save_checkpoint(path, cfg: DetectorConfig, params, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(dict(config=dataclasses.asdict(cfg), params=jax.tree.map(np.asarray, params),
                                      extra=extra or {})))  # fmt: skip
    tmp.rename(path)


class Detector:
    """Loaded detector: ``det(frames)`` -> dict(corners (N, 6, 4, 2), conf (N, 6)) as numpy (jitted)."""

    def __init__(self, path, snap: bool | None = None):
        ck = pickle.loads(Path(path).read_bytes())
        c = dict(ck["config"])
        c["widths"], c["blocks"] = tuple(c["widths"]), tuple(c["blocks"])
        self.cfg = DetectorConfig(**c)
        self.net = GateDetector(self.cfg)
        self.params = jax.tree.map(jnp.asarray, ck["params"])
        self.extra = ck.get("extra", {})
        self.snap = self.cfg.snap if snap is None else snap
        cfg, s = self.cfg, self.snap
        self.forward = jax.jit(lambda p, f: decode(cfg, self.net.apply(p, f), snap=s))

    def __call__(self, frames, numpy: bool = True):
        out = self.forward(self.params, jnp.asarray(frames))
        return {k: np.asarray(v) for k, v in out.items()} if numpy else out


def load_detector(path, snap: bool | None = None) -> Detector:
    return Detector(path, snap)
