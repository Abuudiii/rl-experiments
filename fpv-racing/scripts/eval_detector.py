"""Gate detector: label check against the renderer, and the evaluation.

    python scripts/eval_detector.py check-labels --out runs/percep/detector/label_check
    python scripts/eval_detector.py eval runs/percep/detector/<run>/ckpt_best.pkl --out runs/percep/detector/<run>/eval

check-labels: projects the opening corners (`fpv.perception.labels.gate_labels`) of flight poses and
compares them with the ray caster's own hit map rendered at 4x the resolution (1280x720, one ray per
pixel): along every edge of every fully-in-image visible gate, it scans outward from the projected
edge and records where the gate's frame begins (signed px at 320x180; 0 = the label is exactly on
the rendered opening edge). Also writes a contact sheet with the labels drawn on rendered frames.

eval: see `evaluate`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fpv import track  # noqa: E402
from fpv.perception import detector as D  # noqa: E402
from fpv.perception import labels as L  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_detector import TEACHER, bank_path, load_bank, log, render_set  # noqa: E402


def check_labels(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(bank_path(args.bank_dir, "random", 4242, 16, 300), teacher_ckpt=args.teacher, n_envs=16,
                     steps=300, seed=4242, wide_spawn_p=0.5, teacher_noise=0.3)  # fmt: skip
    rng = np.random.default_rng(0)
    idx = rng.choice(len(bank["pos"]), args.n, replace=False)
    scenes = L.scene_pool(256, 11)
    frames, lab = render_set(L.make_renderer(), bank, idx, scenes, 5)
    S = 4
    fine = L.make_renderer(L.W * S, L.H * S, 1)
    ids = []
    for i in range(0, len(idx), 16):
        j = idx[i : i + 16]
        jp = np.r_[j, np.full(16 - len(j), j[-1])]
        si = np.arange(i, i + 16) % len(scenes["tgt"])
        _, hid = fine(jnp.asarray(bank["pos"][jp]), jnp.asarray(bank["quat"][jp]), jnp.asarray(bank["gates_pos"][jp]),
                      jnp.asarray(bank["gates_yaw"][jp]), {a: jnp.asarray(v[si]) for a, v in scenes.items()},
                      jax.random.key(0))  # fmt: skip
        ids.append(np.asarray(hid[: len(j)]))
    ids = np.concatenate(ids)  # (n, 720, 1280)
    # edge scan
    ts = np.linspace(0.2, 0.8, 7)
    ss = np.arange(-3.0, 4.0 + 1e-9, 0.125)
    trans, dist_of, occl, resid = [], [], 0, []
    # the frame bars are GATE_DEPTH thick: the opening's edge in the image is the nearer of the
    # front-face and back-face opening edges (projected corners of the opening shifted +-depth/2)
    from fpv.render_synth import GATE_DEPTH

    sel = lambda a: a[idx].astype(np.float64)  # noqa: E731
    nrm_w, _ = track.gate_axes(sel(bank["gates_yaw"]))
    faces = [L.gate_labels(sel(bank["pos"]), sel(bank["quat"]), sel(bank["gates_pos"]) + s * GATE_DEPTH / 2 * nrm_w,
                           sel(bank["gates_yaw"]))["corners"] for s in (-1, 1)]  # fmt: skip
    full = lab["visible"] & (lab["n_inside"] == 4) & (lab["side_px"] >= 8)
    for f, g in zip(*np.nonzero(full)):
        q = lab["corners"][f, g]
        c = q.mean(0)
        for e in range(4):
            a, b = q[e], q[(e + 1) % 4]
            d = (b - a) / np.linalg.norm(b - a)
            nrm = np.array([d[1], -d[0]])
            if np.dot(nrm, (a + b) / 2 - c) < 0:
                nrm = -nrm
            pf = []
            for fq in faces:  # signed offset (along nrm) of this edge of each face from p, at each t
                qq = fq[f, g]
                a2, b2 = qq[np.argmin(np.linalg.norm(qq - a, axis=1))], qq[np.argmin(np.linalg.norm(qq - b, axis=1))]
                d2 = (b2 - a2) / max(np.linalg.norm(b2 - a2), 1e-9)
                n2 = np.array([d2[1], -d2[0]])
                pf.append((a2, n2))
            for t in ts:
                p = a + t * (b - a)
                pts = p[None] + ss[:, None] * nrm[None]
                fx = np.round((pts[:, 0] + 0.5) * S - 0.5).astype(int)
                fy = np.round((pts[:, 1] + 0.5) * S - 0.5).astype(int)
                okp = (fx >= 0) & (fx < L.W * S) & (fy >= 0) & (fy < L.H * S)
                hit = np.zeros(len(ss), bool)
                hit[okp] = ids[f, fy[okp], fx[okp]] == g
                if not hit.any():
                    occl += 1
                    continue
                k = int(np.argmax(hit))
                trans.append(ss[k] - 0.0625)
                dist_of.append(lab["dist"][f, g])
                exp = min(np.dot(a2 - p, n2) / np.dot(nrm, n2) for a2, n2 in pf)
                resid.append(trans[-1] - exp)
    trans, dist_of, resid = np.asarray(trans), np.asarray(dist_of), np.asarray(resid)
    res = dict(frames=int(args.n), gates_checked=int(full.sum()), edge_samples=int(len(trans)),
               no_frame_found=int(occl), median=float(np.median(trans)), mean=float(trans.mean()),
               p10=float(np.percentile(trans, 10)), p90=float(np.percentile(trans, 90)),
               frac_abs_le_0p25=float((np.abs(trans) <= 0.25).mean()),
               frac_abs_le_0p5=float((np.abs(trans) <= 0.5).mean()), frac_abs_le_1=float((np.abs(trans) <= 1).mean()))  # fmt: skip
    for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 12)):
        b = (dist_of >= lo) & (dist_of < hi)
        if b.any():
            res[f"median_{lo}_{hi}m"] = float(np.median(trans[b]))
            res[f"frac_abs_le_0p5_{lo}_{hi}m"] = float((np.abs(trans[b]) <= 0.5).mean())
    res["visible_per_frame"] = float(lab["visible"].sum() / args.n)
    res["dontcare_per_frame"] = float(lab["dontcare"].sum() / args.n)
    res["face_aware_residual"] = dict(
        note="transition minus the nearer of the projected front/back face opening edges (px at 320x180, 1/8 px scan)",
        median=float(np.median(resid)), mean=float(resid.mean()), p1=float(np.percentile(resid, 1)),
        p99=float(np.percentile(resid, 99)), frac_abs_le_0p25=float((np.abs(resid) <= 0.25).mean()),
        frac_abs_le_0p5=float((np.abs(resid) <= 0.5).mean()))  # fmt: skip
    res["note"] = "transition = signed px (320x180) from the projected opening edge (gate mid-plane) to the first frame pixel, outward"
    (out / "label_check.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res))
    # contact sheet: 8 frames with several visible gates
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-(lab["visible"].sum(1) + 0.1 * rng.random(len(idx))))
    pick = np.r_[order[:6], np.flatnonzero((lab["visible"] & (lab["n_inside"] < 4)).any(1))[:2]][:8]
    fig, axs = plt.subplots(2, 4, figsize=(20, 5.9))
    cols = ["yellow", "cyan", "lime", "orange"]
    for ax, f in zip(axs.ravel(), pick):
        ax.imshow(frames[f], extent=(-0.5, L.W - 0.5, L.H - 0.5, -0.5))
        for g in range(6):
            if not (lab["visible"][f, g] or lab["dontcare"][f, g]):
                continue
            q = lab["corners"][f, g]
            ls = "-" if lab["visible"][f, g] else ":"
            ax.plot(*np.r_[q, q[:1]].T, ls, color="white", lw=0.6)
            for c in range(4):
                ax.plot(*q[c], "o", ms=3, color=cols[c])
            ax.text(*q.mean(0), f"G{g} {lab['dist'][f, g]:.1f}m", color="white", fontsize=6)
        ax.set_xlim(-0.5 - 20, L.W - 0.5 + 20)
        ax.set_ylim(L.H - 0.5 + 20, -0.5 - 20)
        ax.set_title(f"frame {f}: {int(lab['visible'][f].sum())} visible", fontsize=8)
        ax.axis("off")
    fig.suptitle("labels on synth frames: corner 1 yellow, 2 cyan, 3 lime, 4 orange (clockwise from image top-left); "
                 "solid = visible, dotted = don't-care", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "label_sheet.png", dpi=110)
    log(f"wrote {out}/label_check.json, label_sheet.png")

# region evaluation

DIST_BINS = ((0, 2), (2, 4), (4, 8), (8, 12))
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def square_pnp(uv, side: float = track.GATE_OPENING):
    """Gate centre in camera coordinates (M, 3) from its 4 opening corners (M, 4, 2) in contract order,
    by the DLT homography of the known square and its decomposition (no refinement)."""
    f, cx, cy = L.intrinsics()
    h = side / 2
    X = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])  # clockwise with x right, y down, like the image
    x = (uv - np.array([cx, cy])) / f
    m = uv.shape[0]
    A = np.zeros((m, 8, 9))
    for i in range(4):
        Xi, Yi = X[i]
        xi, yi = x[:, i, 0], x[:, i, 1]
        A[:, 2 * i, 0:3] = [Xi, Yi, 1]
        A[:, 2 * i, 6:9] = np.stack([-xi * Xi, -xi * Yi, -xi], -1)
        A[:, 2 * i + 1, 3:6] = [Xi, Yi, 1]
        A[:, 2 * i + 1, 6:9] = np.stack([-yi * Xi, -yi * Yi, -yi], -1)
    Hm = np.linalg.svd(A)[2][:, -1].reshape(m, 3, 3)
    lam = 2 / (np.linalg.norm(Hm[:, :, 0], axis=-1) + np.linalg.norm(Hm[:, :, 1], axis=-1))
    t = lam[:, None] * Hm[:, :, 2]
    return np.where(t[:, 2:3] < 0, -t, t)


def size_range(uv, side: float = track.GATE_OPENING):
    """Gate centre (M, 3) from the apparent size alone: depth = f * side / (longest edge in px), along the
    ray through the intersection of the quad's diagonals (the perspective-correct centre)."""
    f, cx, cy = L.intrinsics()
    edge = np.linalg.norm(uv - np.roll(uv, -1, -2), axis=-1).max(-1)
    p0, p1, p2, p3 = uv[:, 0], uv[:, 1], uv[:, 2], uv[:, 3]
    d1, d2 = p2 - p0, p3 - p1
    den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    s = ((p1 - p0)[:, 0] * d2[:, 1] - (p1 - p0)[:, 1] * d2[:, 0]) / np.where(np.abs(den) > 1e-9, den, 1e-9)
    c = np.where(np.abs(den)[:, None] > 1e-9, p0 + s[:, None] * d1, uv.mean(1))
    z = f * side / np.maximum(edge, 1e-3)
    return np.stack([(c[:, 0] - cx) / f * z, (c[:, 1] - cy) / f * z, z], -1)


def pose_errors(t_est, t_true):
    """Range error [m] (signed), relative range error, bearing error [deg] (total, azimuth, elevation)."""
    r_e, r_t = np.linalg.norm(t_est, axis=-1), np.linalg.norm(t_true, axis=-1)
    cosb = (t_est * t_true).sum(-1) / (r_e * r_t)
    az = lambda t: np.degrees(np.arctan2(t[..., 0], t[..., 2]))  # noqa: E731
    el = lambda t: np.degrees(np.arctan2(-t[..., 1], np.hypot(t[..., 0], t[..., 2])))  # noqa: E731
    return dict(range=r_e - r_t, rel=(r_e - r_t) / r_t, bearing=np.degrees(np.arccos(np.clip(cosb, -1, 1))),
                az=(az(t_est) - az(t_true) + 180) % 360 - 180, el=el(t_est) - el(t_true))  # fmt: skip


def stats(x) -> dict:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)))  # fmt: skip


def signed_stats(x) -> dict:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), bias=float(x.mean()), std=float(x.std()), mad=float(np.median(np.abs(x - np.median(x)))),
                abs_median=float(np.median(np.abs(x))), abs_p90=float(np.percentile(np.abs(x), 90)))  # fmt: skip


def run_detector(det, render, bank, scenes, seed, chunk=64):
    """Render every bank pose and detect on the device. Each episode keeps one look for its whole flight
    (scene = episode % pool; pixel noise is fresh per frame), so errors along a flight are not decorrelated by
    look changes. Returns numpy dict:
    corners, conf, corners_reg (M, 6, 4, 2 / 6), hidden (M, 6) `labels.hidden_fraction` of each gate."""
    key = jax.random.key(seed)
    ns = len(scenes["tgt"])
    sc_dev = {a: jnp.asarray(v) for a, v in scenes.items()}

    @jax.jit
    def step(pos, quat, gp, gy, si, k):
        img, ids = render(pos, quat, gp, gy, jax.tree.map(lambda a: a[si], sc_dev), k)
        out = D.decode(det.cfg, det.net.apply(det.params, img), snap=det.snap)
        hid = L.hidden_fraction(pos, quat, gp, gy, ids, L.gate_labels(pos, quat, gp, gy)["side_px"])
        return out["corners"], out["conf"], out["corners_reg"], hid

    res = {k: [] for k in ("corners", "conf", "corners_reg", "hidden")}
    m = len(bank["pos"])
    for i in range(0, m, chunk):
        j = np.arange(i, min(i + chunk, m))
        jp = np.r_[j, np.full(chunk - len(j), j[-1])]
        key, k = jax.random.split(key)
        o = step(jnp.asarray(bank["pos"][jp]), jnp.asarray(bank["quat"][jp]), jnp.asarray(bank["gates_pos"][jp]),
                 jnp.asarray(bank["gates_yaw"][jp]), jnp.asarray(bank["episode"][jp] % ns), k)  # fmt: skip
        for name, v in zip(res, o):
            res[name].append(np.asarray(v)[: len(j)])
    return {k: np.concatenate(v) for k, v in res.items()}


def view_angle(bank):
    """Angle [deg] between each gate's normal and the line of sight to its centre (90 = edge-on)."""
    gp, gy = bank["gates_pos"].astype(np.float64), bank["gates_yaw"].astype(np.float64)
    normal, _ = track.gate_axes(gy)
    los = gp - bank["pos"].astype(np.float64)[:, None]
    cosang = np.abs((normal * los).sum(-1)) / np.linalg.norm(los, axis=-1)
    return np.degrees(np.arccos(np.clip(cosang, 0, 1)))


def evaluate_set(name, bank, res, thresh, out_dir) -> dict:
    lab = L.gate_labels(bank["pos"].astype(np.float64), bank["quat"].astype(np.float64),
                        bank["gates_pos"].astype(np.float64), bank["gates_yaw"].astype(np.float64))  # fmt: skip
    n = len(bank["pos"])
    vis = lab["visible"]
    view_deg = view_angle(bank)
    occ = res["hidden"]  # fraction of the frame hidden by nearer gates (labels.hidden_fraction)
    R = dict(set=name, frames=int(n), episodes=int(len(np.unique(bank["episode"]))), threshold=thresh,
             visible_gates=int(vis.sum()), visible_per_frame=float(vis.sum() / n),
             dontcare_per_frame=float(lab["dontcare"].sum() / n))  # fmt: skip
    # precision / recall over thresholds
    pr = []
    for t in THRESHOLDS:
        mt = D.match(res["corners"], res["conf"], lab, t)
        tp = int((mt["det_of_gate"][vis] >= 0).sum())
        fp = int(mt["fp"].sum())
        pr.append(dict(thresh=t, recall=tp / max(vis.sum(), 1), precision=tp / max(tp + fp, 1),
                       fp_per_frame=fp / n, miss=1 - tp / max(vis.sum(), 1)))  # fmt: skip
    R["pr"] = pr
    mt = D.match(res["corners"], res["conf"], lab, thresh)
    err, found = mt["err"], mt["det_of_gate"] >= 0
    R["miss_rate"] = float(1 - found[vis].mean())
    unocc = vis & ~(occ >= 0.7)  # contract labels ignore occlusion; >= 70% of the frame hidden by a nearer gate
    R["miss_rate_excl_hidden_70pct"] = float(1 - found[unocc].mean())
    R["hidden_70pct_gates"] = int((vis & (occ >= 0.7)).sum())
    R["fp_per_frame"] = float(mt["fp"].sum() / n)
    R["frames_with_fp"] = float(mt["fp"].any(1).mean())
    R["corner_err_px"] = stats(err[vis])
    R["corner_err_px_order_free"] = stats(mt["err_inv"][vis])
    R["order_flips"] = float(((err - mt["err_inv"]) > 1.0)[vis & found].mean())  # start corner differs from the label's
    bins = {}
    for lo, hi in DIST_BINS:
        b = vis & (lab["dist"] >= lo) & (lab["dist"] < hi)
        bins[f"{lo}-{hi} m"] = dict(gates=int(b.sum()), miss=float(1 - found[b].mean()) if b.any() else None,
                                    err=stats(err[b]), side_px_median=float(np.median(lab["side_px"][b])) if b.any() else None)  # fmt: skip
    R["by_distance"] = bins
    R["by_corners_inside"] = {f"{k} inside": dict(gates=int((vis & (lab["n_inside"] == k)).sum()),
                                                  miss=float(1 - found[vis & (lab["n_inside"] == k)].mean()) if (vis & (lab["n_inside"] == k)).any() else None,
                                                  err=stats(err[vis & (lab["n_inside"] == k)])) for k in (2, 3, 4)}  # fmt: skip
    ob = {}
    for lo, hi in ((0, 0.05), (0.05, 0.3), (0.3, 0.7), (0.7, 1.01)):
        b = vis & (occ >= lo) & (occ < hi)
        ob[f"{lo}-{min(hi, 1)}"] = dict(gates=int(b.sum()), miss=float(1 - found[b].mean()) if b.any() else None, err=stats(err[b]))
    R["by_frame_occluded_frac"] = ob
    vb = {}
    for lo, hi in ((0, 45), (45, 70), (70, 80), (80, 90.1)):
        b = vis & (view_deg >= lo) & (view_deg < hi)
        vb[f"{lo}-{min(hi, 90)} deg"] = dict(gates=int(b.sum()), miss=float(1 - found[b].mean()) if b.any() else None, err=stats(err[b]))
    R["by_view_angle_from_normal"] = vb
    # next gate along the flight
    tg = bank["target"].astype(int)
    rows = np.arange(n)
    nv = vis[rows, tg]
    nf = found[rows, tg]
    ne = err[rows, tg]
    R["next_gate"] = dict(visible_frac=float(nv.mean()), detection_rate=float(nf[nv].mean()), corner_err_px=stats(ne[nv]),
                          by_distance={f"{lo}-{hi} m": dict(n=int((nv & (lab["dist"][rows, tg] >= lo) & (lab["dist"][rows, tg] < hi)).sum()),
                                                             det=float(nf[nv & (lab["dist"][rows, tg] >= lo) & (lab["dist"][rows, tg] < hi)].mean())
                                                             if (nv & (lab["dist"][rows, tg] >= lo) & (lab["dist"][rows, tg] < hi)).any() else None,
                                                             err=stats(ne[nv & (lab["dist"][rows, tg] >= lo) & (lab["dist"][rows, tg] < hi)]))
                                       for lo, hi in DIST_BINS})  # fmt: skip
    # PnP range / bearing, all matched gates and the next gate; and from the true corners (method floor)
    sel = np.clip(mt["det_of_gate"], 0, 5)
    pred = np.take_along_axis(res["corners"], sel[:, :, None, None], 1)
    fi, gi = np.nonzero(vis & found)
    pe = pose_errors(square_pnp(pred[fi, gi]), lab["centre_cam"][fi, gi])
    ps = pose_errors(size_range(pred[fi, gi]), lab["centre_cam"][fi, gi])
    pt = pose_errors(square_pnp(lab["corners"][fi, gi]), lab["centre_cam"][fi, gi])
    pts = pose_errors(size_range(lab["corners"][fi, gi]), lab["centre_cam"][fi, gi])
    d = lab["dist"][fi, gi]
    pnp, size = {}, {}
    for lo, hi in DIST_BINS:
        b = (d >= lo) & (d < hi)
        pnp[f"{lo}-{hi} m"] = dict(range_m=signed_stats(pe["range"][b]), range_rel=signed_stats(pe["rel"][b]),
                                   bearing_deg=stats(pe["bearing"][b]), az_deg=signed_stats(pe["az"][b]),
                                   el_deg=signed_stats(pe["el"][b]),
                                   true_corners_range_abs_max=float(np.abs(pt["range"][b]).max()) if b.any() else None)  # fmt: skip
        size[f"{lo}-{hi} m"] = dict(range_m=signed_stats(ps["range"][b]), range_rel=signed_stats(ps["rel"][b]),
                                    bearing_deg=stats(ps["bearing"][b]),
                                    true_corners_range_rel=signed_stats(pts["rel"][b]))  # fmt: skip
    R["pnp_by_distance"] = pnp
    R["size_range_by_distance"] = size
    R["pnp_all"] = dict(range_rel=signed_stats(pe["rel"]), bearing_deg=stats(pe["bearing"]))
    R["size_range_all"] = dict(range_rel=signed_stats(ps["rel"]), bearing_deg=stats(ps["bearing"]))
    # time correlation of the next gate's errors along each episode (consecutive frames, same target)
    ep, t = bank["episode"], bank["t"]
    o = np.lexsort((t, ep))
    nfo = (nv & nf)[o]
    ev = mt["err_vec"][rows, tg][o].reshape(n, 8)
    has = nfo
    tmp = np.full((n, 4), np.nan)
    k_idx = np.flatnonzero((nv & nf)[o])
    if len(k_idx):
        oo = o[k_idx]
        pr_ = pose_errors(square_pnp(pred[oo, tg[oo]]), lab["centre_cam"][oo, tg[oo]])
        sr_ = pose_errors(size_range(pred[oo, tg[oo]]), lab["centre_cam"][oo, tg[oo]])
        tmp[k_idx] = np.stack([pr_["rel"], pr_["az"], pr_["el"], sr_["rel"]], -1)
    corr = {}
    ep_o, t_o, tg_o = ep[o], t[o], tg[o]
    for lag in (1, 2, 3, 5, 10, 25):
        a, b = np.arange(n - lag), np.arange(lag, n)
        ok = (ep_o[a] == ep_o[b]) & (t_o[b] - t_o[a] == lag) & (tg_o[a] == tg_o[b]) & has[a] & has[b]
        if ok.sum() < 30:
            continue
        def cc(x, y):
            x, y = x - x.mean(0), y - y.mean(0)
            return float(((x * y).sum(0) / np.sqrt((x**2).sum(0) * (y**2).sum(0) + 1e-12)).mean())
        corr[f"lag {lag} ({lag * 20} ms)"] = dict(pairs=int(ok.sum()), corner_xy=cc(ev[a][ok], ev[b][ok]),
                                                  range_rel=cc(tmp[a][ok][:, :1], tmp[b][ok][:, :1]),
                                                  size_range_rel=cc(tmp[a][ok][:, 3:4], tmp[b][ok][:, 3:4]),
                                                  azimuth=cc(tmp[a][ok][:, 1:2], tmp[b][ok][:, 1:2]),
                                                  elevation=cc(tmp[a][ok][:, 2:3], tmp[b][ok][:, 2:3]))  # fmt: skip
    R["next_gate_error_autocorr"] = corr
    # miss bursts of the next gate (visible but not detected), consecutive frames
    miss_o = (nv & ~nf)[o]
    vis_o = nv[o]
    a, b = np.arange(n - 1), np.arange(1, n)
    cont = (ep_o[a] == ep_o[b]) & (t_o[b] - t_o[a] == 1) & (tg_o[a] == tg_o[b]) & vis_o[a] & vis_o[b]
    runs, cur = [], 0
    for i in range(n):
        if miss_o[i]:
            cur = cur + 1 if (i > 0 and cont[i - 1] and miss_o[i - 1]) else 1
            if i == n - 1 or not (cont[i] if i < n - 1 else False) or not miss_o[i + 1]:
                runs.append(cur)
    R["next_gate_miss_bursts"] = dict(p_miss=float(miss_o[vis_o].mean()) if vis_o.any() else None,
                                      p_miss_after_miss=float(miss_o[b][cont & miss_o[a]].mean()) if (cont & miss_o[a]).any() else None,
                                      p_miss_after_hit=float(miss_o[b][cont & ~miss_o[a]].mean()) if (cont & ~miss_o[a]).any() else None,
                                      run_length=stats(runs) if runs else None)  # fmt: skip
    # false detections: what are they near?
    fpk = mt["fp"]
    fpc = res["corners"][fpk].mean(1)  # (F, 2)
    fr = np.nonzero(fpk)[0]
    allc = lab["corners"].mean(2)  # (N, G, 2)
    front = lab["centre_cam"][..., 2] > 0
    dd = np.linalg.norm(allc[fr] - fpc[:, None], axis=-1)
    near = np.where(front[fr], dd, np.inf).argmin(1)
    dmin = np.take_along_axis(dd, near[:, None], 1)[:, 0]
    on_gate = dmin < np.maximum(8, lab["side_px"][fr, near])
    kinds = dict(
        near_gate_beyond_12m=int((on_gate & (lab["dist"][fr, near] > 12)).sum()),
        duplicate_or_bad_corners_of_visible_gate=int((on_gate & vis[fr, near]).sum()),
        near_other_gate=int((on_gate & ~vis[fr, near] & (lab["dist"][fr, near] <= 12)).sum()),
        background=int((~on_gate).sum()),
    )
    R["false_detections"] = dict(total=int(fpk.sum()), kinds=kinds, conf=stats(res["conf"][fpk]))
    # misses: breakdown
    miss = vis & ~found
    R["misses"] = dict(total=int(miss.sum()), dist=stats(lab["dist"][miss]), side_px=stats(lab["side_px"][miss]),
                       partial=int((miss & (lab["n_inside"] < 4)).sum()),
                       edge_on_gt70deg=int((miss & (view_deg > 70)).sum()),
                       occluded_gt30pct=int((miss & (occ > 0.3)).sum()),
                       beyond_8m=int((miss & (lab["dist"] >= 8)).sum()))  # fmt: skip
    np.savez_compressed(out_dir / f"per_gate_{name}.npz", err=err, found=found, visible=vis, dist=lab["dist"],
                        n_inside=lab["n_inside"], occ=occ, view_deg=view_deg, target=tg, episode=ep, t=t,
                        conf=res["conf"], fp=fpk)  # fmt: skip
    return R


def timing(det):
    out = {}
    for b in (1, 64):
        x = jnp.asarray(np.random.default_rng(0).integers(0, 255, (b, L.H, L.W, 3), np.uint8))
        f = jax.jit(lambda p, x: D.decode(det.cfg, det.net.apply(p, x), snap=det.snap))
        g = jax.jit(lambda p, x: det.net.apply(p, x))
        for name, fn in (("net+decode", f), ("net", g)):
            jax.block_until_ready(fn(det.params, x))
            ts = []
            for _ in range(300):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(det.params, x))
                ts.append(time.perf_counter() - t0)
            ts = np.array(ts[50:]) * 1e3
            out[f"batch {b} {name}"] = dict(median_ms=float(np.median(ts)), p99_ms=float(np.percentile(ts, 99)),
                                            per_frame_ms=float(np.median(ts) / b))  # fmt: skip
        # host round trip at batch 1 (numpy frame in, numpy detections out)
    xh = np.random.default_rng(0).integers(0, 255, (1, L.H, L.W, 3), np.uint8)
    det(xh)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        det(xh)
        ts.append(time.perf_counter() - t0)
    out["batch 1 host numpy round trip"] = dict(median_ms=float(np.median(ts) * 1e3), p99_ms=float(np.percentile(ts, 99) * 1e3))
    out["device"] = str(jax.devices()[0])
    return out

def failure_sheet(det, render, bank, res, scenes, thresh, path, seed: int = 0):
    """2x4 sheet: 4 frames with a missed gate that is not hidden behind another (>= 70%), 4 with a false
    detection. The frames are re-rendered with the same pose and look (fresh pixel noise) and detected again;
    labels white (dotted: don't-care), detections above threshold red with their confidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = L.gate_labels(*(bank[k].astype(np.float64) for k in ("pos", "quat", "gates_pos", "gates_yaw")))
    mt = D.match(res["corners"], res["conf"], lab, thresh)
    rng = np.random.default_rng(seed)
    miss = np.flatnonzero((lab["visible"] & (mt["det_of_gate"] < 0) & (res["hidden"] < 0.7)).any(1))
    fp = np.flatnonzero(mt["fp"].any(1))
    n_miss = min(4, len(miss))
    pick = np.r_[rng.permutation(miss)[:4], rng.permutation(fp)[:4]].astype(int)
    if not len(pick):
        return
    ns = len(scenes["tgt"])
    frames = []
    for j in pick:  # one frame at a time, padded to the compiled chunk of 64
        jp = np.full(64, j)
        img, _ = render(jnp.asarray(bank["pos"][jp]), jnp.asarray(bank["quat"][jp]), jnp.asarray(bank["gates_pos"][jp]),
                        jnp.asarray(bank["gates_yaw"][jp]), {a: jnp.asarray(v[bank["episode"][jp] % ns]) for a, v in scenes.items()},
                        jax.random.key(int(j)))  # fmt: skip
        frames.append(np.asarray(img[0]))
    frames = np.stack(frames)
    dd = det(frames)
    fig, axs = plt.subplots(2, 4, figsize=(20, 5.9))
    for ax in axs.ravel():
        ax.axis("off")
    for k, (f, fr) in enumerate(zip(pick, frames)):
        ax = axs.ravel()[k if k < n_miss else 4 + k - n_miss]
        ax.imshow(fr, extent=(-0.5, L.W - 0.5, L.H - 0.5, -0.5))
        for g in range(6):
            if lab["visible"][f, g] or lab["dontcare"][f, g]:
                q = lab["corners"][f, g]
                ax.plot(*np.r_[q, q[:1]].T, "-" if lab["visible"][f, g] else ":", color="white", lw=0.8)
                ax.text(*q.max(0), f"G{g} {lab['dist'][f, g]:.1f}m h{res['hidden'][f, g]:.1f}", color="white", fontsize=6)
        for s in range(dd["conf"].shape[1]):
            if dd["conf"][k, s] >= thresh:
                q = dd["corners"][k, s]
                ax.plot(*np.r_[q, q[:1]].T, "-", color="red", lw=0.8)
                ax.text(*q.min(0), f"{dd['conf'][k, s]:.2f}", color="red", fontsize=6)
        ax.set_xlim(-20.5, L.W + 19.5)
        ax.set_ylim(L.H + 19.5, -20.5)
        ax.set_title(("miss" if k < n_miss else "false detection") + f" (frame {f})", fontsize=8)
    fig.suptitle("failures (top: missed visible gate, bottom: false detection). white = labels "
                 "(h = hidden fraction), red = detections >= threshold", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)



def evaluate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = D.load_detector(args.ckpt, snap=False if args.no_snap else None)
    render = L.make_renderer()
    log(f"detector {args.ckpt}: {D.n_params(det.params):,} parameters, snap {det.snap}")
    sets = [("random_heldout", None, 2000, args.random_envs, args.random_steps)]
    sets += [(s, s, 3000 + i, args.layout_envs, args.layout_steps) for i, s in enumerate(track.SCENARIOS)]
    banks = {name: load_bank(bank_path(args.bank_dir, name, seed, ne, ns), teacher_ckpt=args.teacher, n_envs=ne,
                             steps=ns, seed=seed, scenario=sc, wide_spawn_p=0.3, teacher_noise=0.2)
             for name, sc, seed, ne, ns in sets}  # fmt: skip
    thresh = args.thresh
    if thresh is None:  # best F1 on the validation bank of training (random tracks, seed 1500)
        vb = load_bank(bank_path(args.bank_dir, "random", 1500, 32, 300), teacher_ckpt=args.teacher, n_envs=32,
                       steps=300, seed=1500, wide_spawn_p=0.3, teacher_noise=0.2)  # fmt: skip
        vr = run_detector(det, render, vb, L.scene_pool(512, 1507), 1500)
        vl = L.gate_labels(*(vb[k].astype(np.float64) for k in ("pos", "quat", "gates_pos", "gates_yaw")))
        f1 = []
        for t in THRESHOLDS:
            mt = D.match(vr["corners"], vr["conf"], vl, t)
            tp = (mt["det_of_gate"][vl["visible"]] >= 0).sum()
            fp = mt["fp"].sum()
            f1.append(2 * tp / (2 * tp + fp + (vl["visible"].sum() - tp)))
        thresh = THRESHOLDS[int(np.argmax(f1))]
        log(f"threshold {thresh} (best F1 {max(f1):.4f} on the validation bank)")
    report = dict(ckpt=str(args.ckpt), params=D.n_params(det.params), config=det.cfg.__dict__, threshold=thresh, sets={})
    pooled_b, pooled_r = [], []
    for i, (name, *_rest) in enumerate(sets):
        t0 = time.time()
        scenes = L.scene_pool(1024, 9001 + i)
        res = run_detector(det, render, banks[name], scenes, 9001 + i)
        report["sets"][name] = evaluate_set(name, banks[name], res, thresh, out)
        log(f"{name}: {len(banks[name]['pos'])} frames, miss {report['sets'][name]['miss_rate']:.4f}, "
            f"err median {report['sets'][name]['corner_err_px'].get('median', float('nan')):.3f} px ({time.time() - t0:.0f} s)")
        if name == "random_heldout":
            failure_sheet(det, render, banks[name], res, scenes, thresh, out / "failures_random_heldout.png")
        else:
            b = dict(banks[name])
            b["episode"] = b["episode"] + 10_000_000 * i  # keep episodes of different layouts apart
            pooled_b.append(b)
            pooled_r.append(res)
    cat = lambda ds: {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    report["sets"]["six_layouts_pooled"] = evaluate_set("six_layouts_pooled", cat(pooled_b), cat(pooled_r), thresh, out)
    log(f"six layouts pooled: {report['sets']['six_layouts_pooled']['frames']} frames")
    report["timing"] = timing(det)
    (out / "eval.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"wrote {out}/eval.json")



def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-labels")
    c.add_argument("--out", default="runs/percep/detector/label_check")
    c.add_argument("--n", type=int, default=512)
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--out", required=True)
    e.add_argument("--random-envs", type=int, default=64)
    e.add_argument("--random-steps", type=int, default=500)
    e.add_argument("--layout-envs", type=int, default=16)
    e.add_argument("--layout-steps", type=int, default=300)
    e.add_argument("--thresh", type=float, default=None, help="default: best F1 on the validation bank")
    e.add_argument("--no-snap", action="store_true")
    for p in (c, e):
        p.add_argument("--teacher", default=TEACHER)
        p.add_argument("--bank-dir", default="/tmp/fpv_percep/PerceptDetector")
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.cmd == "check-labels":
        check_labels(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
