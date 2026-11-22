"""Video of one recorded flight on an acrobatic layout, with the synthetic renderer (no Unreal).

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 .venv-rocm/bin/python scripts/acro_video.py \
        runs/teacher/acro/<run>/eval_acro/record_ep0.npz --out runs/teacher/acro/<run>/video

Input: a flight record (`eval_teacher.write_record` / `scripts/eval_acro.py` format). Output in --out:
  <stem>_combined.mp4  1280x720, 2x2: FPV (synthetic ray caster, next gate green, stick bars) | CHASE
                       (ray caster from `render_airsim.chase_camera`'s path, looking at the drone; the
                       drone is drawn on top as a glyph: arms, red nose, thrust axis green when it points
                       up and red when it points down) | SIDE view (orthographic x-z, from -y: gates,
                       path, drone glyph) | TOP-DOWN map (`flight_video.TopDownMap`).
  <stem>_{fpv,chase,side}.mp4  the panels on their own.
  <stem>_dive_sheet.jpg  combined frames at 8 evenly spaced steps of the first dive (G1 pass -> G2 pass).
  <stem>_video.json    frame count, dive steps on the sheet, tilt at each of them.
The glyph is drawn about 3.5x the true size (arm 0.12 m) so that it is visible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import flight_video as fv  # noqa: E402
from fpv import render_airsim as ra  # noqa: E402
from fpv import render_synth as rs  # noqa: E402
from fpv import track  # noqa: E402

W, H = fv.PANEL
ARM, AXIS = 0.12, 0.16  # glyph sizes [m] in the chase view
GREEN, RED, NOSE = (60, 230, 90), (255, 60, 40), (255, 120, 0)


def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)  # fmt: skip


def glyph_points(pos, R, arm=ARM, axis=AXIS):
    """World points of the drone glyph: centre, 4 arm tips, nose, thrust-axis tip."""
    bx, by, bz = R[:, 0], R[:, 1], R[:, 2]
    tips = [pos + arm * (sx * bx + sy * by) / math.sqrt(2) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return np.stack([pos, *tips, pos + arm * 1.3 * bx, pos + axis * bz])


def draw_glyph(d, uv, depth, cz, scale=1.0):
    if (depth <= 0.02).any():
        return
    c = tuple(uv[0])
    for i in range(1, 5):
        d.line([c, tuple(uv[i])], fill=(235, 235, 235, 255), width=max(2, int(3 * scale)))
        r = 6 * scale
        d.ellipse([uv[i][0] - r, uv[i][1] - r, uv[i][0] + r, uv[i][1] + r], outline=(200, 200, 200, 255), width=2)
    d.line([c, tuple(uv[5])], fill=NOSE + (255,), width=max(3, int(4 * scale)))
    d.line([c, tuple(uv[6])], fill=(GREEN if cz > 0 else RED) + (255,), width=max(3, int(4 * scale)))


class SideView:
    """Orthographic x-z view seen from -y (x to the right, z up)."""

    def __init__(self, rec, z_max):
        self.rec = rec
        gp, gy = rec["gates_pos"].astype(np.float64), rec["gates_yaw"].astype(np.float64)
        self.outer = track.gate_corners(gp, gy, track.GATE_OUTER)  # (G, 4, 3)
        self.inner = track.gate_corners(gp, gy, track.GATE_OPENING)
        xs = np.concatenate([self.outer[..., 0].ravel(), rec["pos"][:, 0]])
        lo, hi = np.array([xs.min() - 0.5, 0.0]), np.array([xs.max() + 0.5, max(z_max, rec["pos"][:, 2].max()) + 0.2])
        top = 60
        self.s = min((W - 30) / (hi[0] - lo[0]), (H - top - 15) / (hi[1] - lo[1]))
        self.ox = 15 + ((W - 30) - self.s * (hi[0] - lo[0])) / 2 - self.s * lo[0]
        self.oz = H - 10
        self.z_max = z_max
        self.font, self.small = fv._font(18), fv._font(14)

    def px(self, p):
        p = np.asarray(p)
        return np.stack([self.ox + self.s * p[..., 0], self.oz - self.s * p[..., 2]], -1)

    def draw(self, t, R, title="SIDE VIEW (from -y)"):
        rec = self.rec
        im = Image.new("RGB", (W, H), (24, 26, 30))
        d = ImageDraw.Draw(im, "RGBA")
        g0 = self.px(np.array([[-100, 0, 0], [100, 0, 0]]))
        d.line([tuple(g0[0]), tuple(g0[1])], fill=(120, 100, 70, 255), width=2)
        zc = self.px(np.array([[-100, 0, self.z_max], [100, 0, self.z_max]]))
        d.line([tuple(zc[0]), tuple(zc[1])], fill=(120, 120, 120, 120), width=1)
        nxt = int(rec["target"][t]) if rec.get("target") is not None else -1
        for i in range(len(self.outer)):
            col = (0, 230, 0, 255) if i == nxt else (255, 0, 204, 255)
            d.polygon([tuple(p) for p in self.px(self.outer[i])], outline=col, width=3)
            d.polygon([tuple(p) for p in self.px(self.inner[i])], outline=col, width=1)
            c = self.px(rec["gates_pos"][i])
            d.text((c[0] + 14, c[1] - 26), f"G{i + 1}", fill=(255, 140, 230, 255), font=self.small)
        path = self.px(rec["pos"][: t + 1])
        if len(path) > 1:
            d.line([tuple(p) for p in path], fill=(150, 150, 150, 110), width=1)
            k0 = max(0, t - int(1.5 * rec["hz"]))
            d.line([tuple(p) for p in path[k0:]], fill=(255, 220, 0, 230), width=2)
        p = rec["pos"][t]
        c = self.px(p)
        bx, bz = R[:, 0], R[:, 2]
        L = 0.45
        for v, col, w in ((bx, NOSE, 4), (bz, GREEN if bz[2] > 0 else RED, 4)):
            e = self.px(p + L * v)
            d.line([tuple(c), tuple(e)], fill=col + (255,), width=w)
        e1, e2 = self.px(p + 0.25 * R[:, 0]), self.px(p - 0.25 * R[:, 0])
        d.line([tuple(e1), tuple(e2)], fill=(235, 235, 235, 255), width=3)
        d.ellipse([c[0] - 4, c[1] - 4, c[0] + 4, c[1] + 4], fill=(255, 255, 255, 255))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, bz[2]))))
        d.text((14, 10), title, fill=(255, 255, 255, 255), font=self.font)
        d.text((14, 34), f"tilt {tilt:5.1f} deg   z {p[2]:.2f} m   orange = nose, green/red = thrust axis up/down",
               fill=(210, 210, 210, 255), font=self.small)
        if tilt > 90:
            tag = "UPSIDE DOWN" if tilt > 150 else "PAST 90 DEG"
            d.rectangle([W - 190, 8, W - 10, 36], fill=(200, 30, 30, 230) if tilt > 150 else (200, 120, 0, 230))
            d.text((W - 180, 12), tag, fill=(255, 255, 255, 255), font=self.font)
        return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-max", type=float, default=6.0)
    ap.add_argument("--seconds", type=float, default=None, help="cut the video after this many seconds")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = fv.load_record(args.record)
    stem = Path(args.record).stem
    T = len(rec["pos"]) if args.seconds is None else min(len(rec["pos"]), int(args.seconds * rec["hz"]))
    pos, quat = rec["pos"][:T].astype(np.float64), rec["quat"][:T].astype(np.float64)
    target = rec["target"][:T] if rec.get("target") is not None else np.full(T, -1)
    Rm = ra.quat_to_mat(quat)

    r = rs.SynthRenderer(1, W, H, seed=0, randomize=False, highlight_target=True)
    r.set_gates([0], rec["gates_pos"][None], rec["gates_yaw"][None])
    fpv_frames = r.render_trajectories([0], pos[None], quat[None], target=target[None])[0]
    cp, cq, cinfo = ra.chase_camera(pos, quat, rec["gates_pos"], rec["gates_yaw"], hz=rec["hz"], ahead=0.3)
    s = math.sin(math.radians(track.CAMERA["pitch_up_deg"]) / 2)
    q_pitch = np.array([0.0, s, 0.0, math.cos(math.radians(track.CAMERA["pitch_up_deg"]) / 2)])
    cq_body = qmul(cq, np.broadcast_to(q_pitch, cq.shape))  # synth optical axis (pitched up) = chase axis
    chase_frames = r.render_trajectories([0], cp[None], cq_body[None], target=target[None])[0]

    side = SideView(rec, args.z_max)
    tmap = fv.TopDownMap(rec, fv.PANEL)
    writers = {v: fv.Writer(out / f"{stem}_{v}.mp4", rec["hz"]) for v in ("fpv", "chase", "side")}
    combo = fv.Writer(out / f"{stem}_combined.mp4", rec["hz"])
    # first dive: G1 (index 0) passed -> G2 (index 1) passed
    k1 = np.flatnonzero(rec["passed"][:T] & (target == 0))
    dive = None
    if len(k1):
        k2 = np.flatnonzero(rec["passed"][:T] & (target == 1))
        k2 = k2[k2 > k1[0]]
        if len(k2):
            dive = (int(k1[0]) + 1, int(k2[0]) + 1)
    sheet_steps = list(np.linspace(dive[0] - 5, dive[1] + 3, 8).round().astype(int)) if dive else []
    sheet = []
    for t in range(T):
        cz = Rm[t, 2, 2]
        f = fv.draw_bars(fpv_frames[t], rec["action"][t], None)
        f = fv.annotate(f, "PILOT VIEW (FPV, synthetic)", fv.caption(rec, t))
        im = Image.fromarray(chase_frames[t])
        d = ImageDraw.Draw(im, "RGBA")
        pts = glyph_points(pos[t], Rm[t])
        uv, depth = rs.project_points(cp[t], cq_body[t], pts, W, H)
        draw_glyph(d, uv, depth, cz)
        c = fv.annotate(np.asarray(im), "CHASE (synthetic, drone glyph 3.5x)", fv.caption(rec, t))
        sd = side.draw(t, Rm[t])
        panels = [f, c, sd, tmap.draw(t)]
        frame = np.zeros((fv.OUT_SIZE[1], fv.OUT_SIZE[0], 3), np.uint8)
        for i, img in enumerate(panels):
            r_, c_ = divmod(i, 2)
            frame[r_ * H:(r_ + 1) * H, c_ * W:(c_ + 1) * W] = img
        writers["fpv"].add(f), writers["chase"].add(c), writers["side"].add(sd)
        combo.add(frame)
        if t in sheet_steps:
            sheet.append((t, frame))
    for w in (*writers.values(), combo):
        w.close()
    info = dict(record=args.record, steps=T, hz=rec["hz"], chase=cinfo, first_dive_steps=dive,
                sheet=[dict(step=int(t), t_s=round(t / rec["hz"], 2),
                            tilt_deg=round(math.degrees(math.acos(float(np.clip(Rm[t, 2, 2], -1, 1)))), 1),
                            z=round(float(pos[t, 2]), 2)) for t, _ in sheet])  # fmt: skip
    if sheet:
        tw, th = 640, 360
        img = Image.new("RGB", (2 * tw, 4 * th))
        for i, (t, fr) in enumerate(sheet):
            r_, c_ = divmod(i, 2)
            img.paste(Image.fromarray(fr).resize((tw, th), Image.BILINEAR), (c_ * tw, r_ * th))
        img.save(out / f"{stem}_dive_sheet.jpg", quality=85)
    (out / f"{stem}_video.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, default=float)[:2000])


if __name__ == "__main__":
    main()
