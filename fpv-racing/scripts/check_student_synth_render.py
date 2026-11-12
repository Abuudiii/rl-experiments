"""Geometry and speed checks of `fpv.render_synth.SynthRenderer`.

    CUDA_VISIBLE_DEVICES=2 .venv-sim/bin/python scripts/check_student_synth_render.py [--sheet out.png]

(i)   poses from runs/teacher/cruise_s1/rollouts.npz: the magenta mask of a clean render against an
      independent rasterisation (project the gate corners with `project_points`, fill polygons with
      PIL): IoU per frame; and corner pixels (just inside each outer corner magenta, opening centre not).
(ii)  occlusion: where two gate frames overlap in the image the hit map names the nearer gate.
(iii) horizon: pixels whose ray points above / below the world horizontal are sky / ground.
(iv)  throughput of the randomised renderer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from fpv import track  # noqa: E402
from fpv.render_synth import SynthRenderer, _pixel_dirs, project_points  # noqa: E402
from fpv.env import quat_to_mat  # noqa: E402


def reference_mask(pos, quat, gp, gy, w, h, ss=4):
    """Union of the six gate frames by polygon filling (frames with a corner behind the camera: None)."""
    outer = track.gate_corners(gp, gy, track.GATE_OUTER)  # (6, 4, 3)
    inner = track.gate_corners(gp, gy, track.GATE_OPENING)
    uo, do = project_points(pos, quat, outer.reshape(-1, 3), w, h)
    ui, _ = project_points(pos, quat, inner.reshape(-1, 3), w, h)
    do = do.reshape(6, 4)
    if ((do > -0.3) & (do < 0.3)).any() or ((do.min(1) < 0) != (do.max(1) < 0)).any():
        return None  # a gate straddles the camera plane
    mask = np.zeros((h, w), np.float32)
    for g in range(6):
        if do[g].max() < 0:
            continue
        ring = Image.new("L", (w * ss, h * ss), 0)
        dr = ImageDraw.Draw(ring)
        dr.polygon([tuple((p + 0.5) * ss) for p in uo.reshape(6, 4, 2)[g]], fill=255)
        dr.polygon([tuple((p + 0.5) * ss) for p in ui.reshape(6, 4, 2)[g]], fill=0)
        mask = np.maximum(mask, np.asarray(ring, np.float32).reshape(h, ss, w, ss).mean((1, 3)) / 255)
    return mask


def is_magenta(img):
    r, g, b = (img[..., i].astype(np.float32) for i in range(3))
    return (r > 120) & (b > 120) & (g < 0.6 * np.minimum(r, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="runs/teacher/cruise_s1/rollouts.npz")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=90)
    args = ap.parse_args()
    w, h = args.width, args.height
    d = np.load(args.rollouts)
    rng = np.random.default_rng(0)
    n = 512
    ep = rng.integers(0, d["pos"].shape[0], n)
    st = np.array([rng.integers(0, d["length"][e]) for e in ep])
    pos, quat, gp, gy = d["pos"][ep, st], d["quat"][ep, st], d["gates_pos"][ep], d["gates_yaw"][ep]
    res = {}

    # (i) clean render against polygon rasterisation
    r = SynthRenderer(n, w, h, randomize=False)
    r.set_gates(np.arange(n), gp, gy)
    img, ids = r.render_ids(pos, quat)
    mag = is_magenta(img)
    ious, used, diff_px = [], 0, []
    for i in range(n):
        ref = reference_mask(pos[i], quat[i], gp[i], gy[i], w, h)
        if ref is None or (ref > 0.5).sum() < 20:
            continue
        a, b = mag[i], ref > 0.5
        ious.append((a & b).sum() / max((a | b).sum(), 1))
        diff_px.append((a ^ b).sum())
        used += 1
    res["mask_iou"] = dict(frames=used, mean=float(np.mean(ious)), p05=float(np.percentile(ious, 5)),
                           min=float(np.min(ious)), mean_differing_pixels=float(np.mean(diff_px)),
                           mean_gate_pixels=float(mag.sum() / n))  # fmt: skip
    # corner test on the target gate: a point 15% inside each outer corner (on the frame) must be
    # magenta or hidden by a nearer gate, the opening centre must not show the target gate itself
    tgt = d["target"][ep, st].astype(int)
    c_out = track.gate_corners(gp[np.arange(n), tgt], gy[np.arange(n), tgt], track.GATE_OUTER)
    centre = gp[np.arange(n), tgt]
    probe = centre[:, None] + 0.85 * (c_out - centre[:, None])
    uv, depth = project_points(pos, quat, probe, w, h)
    px = np.rint(uv).astype(int)
    ok = (depth > 0.5) & (px[..., 0] >= 1) & (px[..., 0] < w - 1) & (px[..., 1] >= 1) & (px[..., 1] < h - 1)
    ii = np.repeat(np.arange(n), 4).reshape(n, 4)
    hit = ids[ii[ok], px[..., 1][ok], px[..., 0][ok]]
    res["corner_probe"] = dict(n=int(ok.sum()), on_target_gate=float((hit == tgt[ii[ok]]).mean()),
                               on_any_gate=float(((hit >= 0) & (hit < 6)).mean()))  # fmt: skip
    uvc, dc = project_points(pos, quat, centre[:, None], w, h)
    pc = np.rint(uvc[:, 0]).astype(int)
    okc = (dc[:, 0] > 0.5) & (pc[:, 0] >= 0) & (pc[:, 0] < w) & (pc[:, 1] >= 0) & (pc[:, 1] < h)
    hc = ids[np.flatnonzero(okc), pc[okc, 1], pc[okc, 0]]
    res["opening_centre"] = dict(n=int(okc.sum()), shows_target_gate=float((hc == tgt[okc]).mean()))

    # (ii) occlusion: the hit map must name the nearest of all gates covering a pixel
    dirs = _pixel_dirs(w, h, 2).reshape(h, 2, w, 2, 3)[:, 0, :, 0]  # the sub-ray the hit map uses
    rot = np.asarray(quat_to_mat(quat))
    wrong = total = overlap = 0
    for i in range(64):
        dw = dirs @ rot[i].T
        best, bid = np.full((h, w), np.inf), np.full((h, w), -1)
        cover = np.zeros((h, w), int)
        for g in range(6):
            nrm, left = track.gate_axes(gy[i, g])
            dn = dw @ nrm
            t = ((gp[i, g] - pos[i]) @ nrm) / np.where(np.abs(dn) > 1e-6, dn, 1e-6)
            rel = pos[i] + t[..., None] * dw - gp[i, g]
            rr = np.maximum(np.abs(rel @ left), np.abs(rel[..., 2]))
            m = (t > 0.03) & (rr >= 0.2) & (rr < 0.36)
            cover += m
            upd = m & (t < best)
            best, bid = np.where(upd, t, best), np.where(upd, g, bid)
        sel = bid >= 0
        total += sel.sum()
        overlap += (cover > 1).sum()
        wrong += (ids[i][sel] != bid[sel]).sum()
    res["occlusion"] = dict(gate_pixels=int(total), pixels_with_two_gates=int(overlap), wrong_gate_id=int(wrong))

    # (iii) horizon
    dw = np.einsum("nij,hwj->nhwi", rot[:64], dirs)
    elev = dw[..., 2] / np.linalg.norm(dw, axis=-1)
    bg = ids[:64] < 0
    res["horizon"] = dict(
        sky_where_ray_up=float((ids[:64][bg & (elev > 0.01)] == -2).mean()),
        ground_where_ray_down=float((ids[:64][bg & (elev < -0.01)] == -1).mean()),
    )

    # (iv) throughput (randomised look, including the copy to the host)
    for nn in (256, 1024):
        rr = SynthRenderer(nn, w, h, seed=1)
        e2 = rng.integers(0, d["pos"].shape[0], nn)
        p2, q2 = d["pos"][e2, 100], d["quat"][e2, 100]
        rr.set_gates(np.arange(nn), d["gates_pos"][e2], d["gates_yaw"][e2])
        rr.render(p2, q2)
        t0 = time.perf_counter()
        for _ in range(20):
            out = rr.render(p2, q2)
        dt = time.perf_counter() - t0
        res[f"speed_n{nn}"] = dict(frames_per_s=float(20 * nn / dt), shape=list(out.shape))
        if nn == 256 and args.sheet:
            tiles = [out[i] for i in range(12)] + [img[i] for i in range(4)]
            rows = [np.concatenate(tiles[k : k + 4], 1) for k in range(0, 16, 4)]
            Image.fromarray(np.concatenate(rows, 0)).resize((w * 8, h * 8), Image.NEAREST).save(args.sheet)
    print("RESULT " + json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
