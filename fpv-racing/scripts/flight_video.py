"""Videos of recorded flights in Unreal: the pilot's FPV view, a chase camera and a fixed overview.

    .venv-sim/bin/python scripts/flight_video.py <record.npz> [<record2.npz> ...] --out <dir> \
        [--views fpv,chase,overview] [--layout grid|pip] [--gpus 4 5] [--ipg 4]

Flight record (one episode per .npz; `save_record` writes one):
    pos [T,3] world, quat [T,4] xyzw body->world, action [T,4] applied action in [-1, 1],
    label [T,4] teacher action for the same state (optional), gates_pos [6,3], gates_yaw [6],
    passed [T] bool, step carried the drone through its target gate (optional, else recomputed),
    target [T] int, the next gate at each step (optional, else derived from passed),
    hz (50), name (scenario / caption).

Per record (<stem> = file name without .npz) in --out:
    <stem>_combined.mp4   1280x720. grid: FPV with stick bars | chase / overview | top-down map with
                          caption. pip: chase full frame, FPV and overview as insets.
    <stem>_<view>.mp4     each requested view on its own (fpv with stick bars, overview with a
                          marker on the drone).
    <stem>_sheet.jpg      contact sheet, 12 chase-camera moments (--sheet-name to rename).
    <stem>_video.json     camera placement, render timing, sanity numbers.
--subdirs puts them in <out>/<stem>/, --caption replaces the caption, --summary-sheet adds one
sheet with a mid-flight chase and overview frame per record.
All videos: h264, yuv420p, frame rate = the record's `hz`.

The FPV view comes from the default FPV renderer (no drone model in the scene); chase and overview
come from a renderer with the quadrotor model (`fpv.render_airsim.drone_parts`, `--drone-scale`
times true size) and an 80 degree camera. Each view's frames are split over all slots of its
renderer (frames do not depend on each other).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpv import track  # noqa: E402
from fpv import render_airsim as ra  # noqa: E402

NAMES = ("THR", "ROLL", "PITCH", "YAW")
VIEWS = ("fpv", "chase", "overview")
PANEL = (640, 360)
OUT_SIZE = (1280, 720)


# ----------------------------------------------------------------------------- flight record
def save_record(path, *, pos, quat, action, gates_pos, gates_yaw, name, label=None, passed=None, target=None, hz=50):
    rec = dict(pos=np.asarray(pos, np.float32), quat=np.asarray(quat, np.float32), action=np.asarray(action, np.float32),
               gates_pos=np.asarray(gates_pos, np.float32).reshape(6, 3), gates_yaw=np.asarray(gates_yaw, np.float32).reshape(6),
               hz=np.float32(hz), name=np.str_(name))
    if label is not None:
        rec["label"] = np.asarray(label, np.float32)
    if passed is not None:
        rec["passed"] = np.asarray(passed, bool)
    if target is not None:
        rec["target"] = np.asarray(target, np.int8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **rec)
    return Path(path)


def gate_passes(pos, gates_pos, gates_yaw):
    """Replay the gate logic on a path -> (passed [T], target [T]): step t -> t+1 crossed the current
    target gate through its opening; target = next gate at step t (the first gate the drone flies
    through counts as the target from the start; -1 before it is known, i.e. never passed)."""
    T = len(pos)
    passed, target = np.zeros(T, bool), np.full(T, -1, np.int64)
    cur, first = None, None
    for t in range(T - 1):
        ok, _, _ = track.gate_crossing(pos[t][None], pos[t + 1][None], gates_pos, gates_yaw)
        hit = np.flatnonzero(ok)
        if cur is None and len(hit):
            cur = first = int(hit[0])
        if cur is not None:
            target[t] = cur
            if cur in hit:
                passed[t] = True
                cur = (cur + 1) % len(gates_yaw)
    if cur is not None:
        target[T - 1] = cur
        target[target < 0] = first  # before the first pass the drone was heading for that gate
    return passed, target


def load_record(path) -> dict:
    with np.load(path) as z:
        rec = {k: z[k] for k in z.files}
    for k in ("pos", "quat", "action", "gates_pos", "gates_yaw"):
        if k not in rec:
            raise ValueError(f"{path}: flight record lacks '{k}'")
    rec["hz"] = float(rec.get("hz", 50.0))
    rec["name"] = str(rec.get("name", Path(path).stem))
    rec["label"] = rec.get("label")
    if "passed" not in rec or "target" not in rec:
        passed, target = gate_passes(rec["pos"].astype(np.float64), rec["gates_pos"].astype(np.float64),
                                     rec["gates_yaw"].astype(np.float64))
        rec.setdefault("passed", passed)
        rec.setdefault("target", target if (target >= 0).any() else None)
    rec["gates_cum"] = np.concatenate([[0], np.cumsum(rec["passed"])[:-1]])  # gates passed BEFORE frame t
    if rec["target"] is not None:
        rec["target"] = np.asarray(rec["target"], np.int64)
    return rec


# ----------------------------------------------------------------------------- cameras
def camera_paths(rec, views, size):
    """{view: (cam_pos [T,3], cam_quat [T,4])} for chase / overview, plus info."""
    T, out, info = len(rec["pos"]), {}, {}
    if "chase" in views:
        cp, cq, ci = ra.chase_camera(rec["pos"], rec["quat"], rec["gates_pos"], rec["gates_yaw"], hz=rec["hz"])
        yaw = np.unwrap(np.arctan2(ra.quat_to_mat(cq)[:, 1, 0], ra.quat_to_mat(cq)[:, 0, 0]))
        ci["max_camera_yaw_rate_deg_s"] = round(float(np.degrees(np.abs(np.diff(yaw)).max()) * rec["hz"]), 1)
        out["chase"], info["chase"] = (cp, cq), ci
    if "overview" in views:
        pts = np.concatenate([track.gate_corners(rec["gates_pos"].astype(np.float64), rec["gates_yaw"].astype(np.float64),
                                                 track.GATE_OUTER).reshape(-1, 3), rec["pos"][:: max(1, T // 300)]])
        op, oq = ra.overview_camera(pts, *size)
        out["overview"] = (np.repeat(op[None], T, 0), np.repeat(oq[None], T, 0))
        info["overview"] = {"pos": op.round(3).tolist(), "quat_xyzw": oq.round(4).tolist()}
    return out, info


def render_split(r, cam_pos, cam_quat, drone_pos=None, drone_quat=None, target=None):
    """Render one long pose sequence on all slots of ``r`` (contiguous segments, one per slot)."""
    T, k = len(cam_pos), r.n
    seg = -(-T // k)
    pad = lambda x: None if x is None else np.concatenate([x, np.repeat(x[-1:], k * seg - T, 0)]).reshape(k, seg, -1)  # noqa: E731
    tg = None if target is None else pad(np.asarray(target)[:, None])[..., 0]
    f = r.render_trajectories(np.arange(k), pad(cam_pos), pad(cam_quat), pad(drone_pos), pad(drone_quat), target=tg)
    return f.reshape(k * seg, r.height, r.width, 3)[:T]


def render_records(records, views, size, gpus, ipg, look, drone_scale, log=print, highlight=("chase", "overview")):
    """[{view: frames [T,H,W,3]}] per record, plus info per record. ``highlight``: views that show the
    record's target gate in `fpv.render_airsim.HIGHLIGHT_RGB` (records without a target: none)."""
    frames = [dict() for _ in records]
    infos = [dict(render_s={}) for _ in records]
    paths = []
    for rec, info in zip(records, infos):
        p, i = camera_paths(rec, views, size)
        paths.append(p)
        info["cameras"] = i
    renderers = {}
    hl = any(r["target"] is not None for r in records)
    if "fpv" in views:
        renderers["fpv"] = dict(hfov_deg=None, mount_pitch_deg=None, drone_scale=0.0,
                                highlight_target=hl and "fpv" in highlight)
    if {"chase", "overview"} & set(views):
        renderers["third_person"] = dict(hfov_deg=ra.TP_HFOV_DEG, mount_pitch_deg=0.0, drone_scale=drone_scale,
                                         highlight_target=hl and bool({"chase", "overview"} & set(highlight)))
    for kind, kw in renderers.items():
        t0 = time.time()
        with ra.AirsimRenderer(gpus=gpus, instances_per_gpu=ipg, width=size[0], height=size[1], look=look, **kw) as r:
            started = round(time.time() - t0, 1)
            log(f"{kind} renderer: {r.n} slots up in {started} s, launches {sum(r.launches.values())}")
            for rec, fr, info, path in zip(records, frames, infos, paths):
                info.setdefault("renderers", {})[kind] = {"slots": r.n, "startup_s": started,
                                                         "failed_launches": [x["error"] for x in r.failed_launches]}
                r.set_gates(np.arange(r.n), np.repeat(rec["gates_pos"][None], r.n, 0), np.repeat(rec["gates_yaw"][None], r.n, 0))
                T = len(rec["pos"])
                t1 = time.time()
                tgt = rec["target"] if kw["highlight_target"] and rec["target"] is not None else None
                if kw["highlight_target"] and tgt is None:
                    tgt = np.full(T, -1)
                if kind == "fpv":
                    fr["fpv"] = render_split(r, rec["pos"], rec["quat"], target=tgt)
                    info["render_s"]["fpv"] = round(time.time() - t1, 1)
                else:
                    tp = [v for v in ("chase", "overview") if v in views]
                    cp = np.concatenate([path[v][0] for v in tp])
                    cq = np.concatenate([path[v][1] for v in tp])
                    dp = np.concatenate([rec["pos"]] * len(tp))
                    dq = np.concatenate([rec["quat"]] * len(tp))
                    tt = None if tgt is None else np.concatenate([tgt if v in highlight else np.full(T, -1) for v in tp])
                    allf = render_split(r, cp, cq, dp, dq, tt)
                    for j, v in enumerate(tp):
                        fr[v] = allf[j * T:(j + 1) * T]
                    info["render_s"]["+".join(tp)] = round(time.time() - t1, 1)
                log(f"  {rec['name']}: {kind} {T} frames in {time.time() - t1:.1f} s")
    return frames, infos


# ----------------------------------------------------------------------------- drawing
def _font(size):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_bars(img, action, label):
    """Four vertical stick bars in the lower-left corner (thrust, roll, pitch, yaw rate in [-1, 1];
    green = commanded, white tick = teacher's label when given). Returns a new array."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    h = img.shape[0]
    x0, y0, bw, bh, gap = 16, h - 26, 22, 90, 14
    d.rectangle([x0 - 8, y0 - bh - 10, x0 + 4 * (bw + gap), y0 + 18], fill=(0, 0, 0, 120))
    for i in range(4):
        x = x0 + i * (bw + gap)
        mid = y0 - bh / 2
        d.rectangle([x, y0 - bh, x + bw, y0], outline=(255, 255, 255, 220))
        d.line([x, mid, x + bw, mid], fill=(255, 255, 255, 120))
        v = float(np.clip(action[i], -1, 1))
        d.rectangle([x + 2, min(mid, mid - v * bh / 2), x + bw - 2, max(mid, mid - v * bh / 2)], fill=(60, 220, 90, 255))
        if label is not None:
            t = mid - float(np.clip(label[i], -1, 1)) * bh / 2
            d.line([x - 3, t, x + bw + 3, t], fill=(255, 255, 255, 255), width=2)
        d.text((x, y0 + 3), NAMES[i], fill=(255, 255, 255, 255))
    return np.asarray(im)


def _tag(d, xy, text, font):
    x, y = xy
    l, t, r, b = d.textbbox((x, y), text, font=font)
    d.rectangle([l - 5, t - 3, r + 5, b + 3], fill=(0, 0, 0, 140))
    d.text((x, y), text, fill=(255, 255, 255, 255), font=font)


def caption(rec, t):
    return f"{rec['name']}   t {t / rec['hz']:5.2f} s   gates {int(rec['gates_cum'][t])}"


def annotate(img, title, text=None, font_size=15):
    """Panel title top-left, optional caption top-right."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    f = _font(font_size)
    _tag(d, (10, 8), title, f)
    if text:
        w = d.textlength(text, font=f)
        _tag(d, (img.shape[1] - w - 12, 8), text, f)
    return np.asarray(im)


def mark_drone(img, rec, t, cam_pos, cam_quat, trail_s=1.0):
    """Overview: yellow ring on the drone and its path over the last ``trail_s`` seconds."""
    from PIL import Image, ImageDraw

    H, W = img.shape[:2]
    t0 = max(0, t - int(trail_s * rec["hz"]))
    uv, z = ra.project_points(rec["pos"][t0 : t + 1], cam_pos, cam_quat, W, H, ra.TP_HFOV_DEG)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    if len(uv) > 1:
        d.line([tuple(p) for p in uv.tolist()], fill=(255, 220, 0, 160), width=2)
    if z[-1] > 0:
        u, v = uv[-1]
        d.ellipse([u - 11, v - 11, u + 11, v + 11], outline=(255, 220, 0, 255), width=2)
    return np.asarray(im)


class TopDownMap:
    """Top-down map panel: gates, flown path so far, drone position and heading, caption."""

    def __init__(self, rec, size):
        self.rec, (self.W, self.H) = rec, size
        gp, gy = rec["gates_pos"].astype(np.float64), rec["gates_yaw"].astype(np.float64)
        pts = np.concatenate([gp[:, :2], rec["pos"][:, :2]])
        lo, hi = pts.min(0) - 0.6, pts.max(0) + 0.6
        top = 92  # caption area
        self.s = min((self.W - 40) / (hi[0] - lo[0]), (self.H - top - 20) / (hi[1] - lo[1]))
        self.c = (lo + hi) / 2
        self.cx, self.cy = self.W / 2, top + (self.H - top) / 2
        self.gates = []
        for g, y in zip(gp, gy):
            _, left = track.gate_axes(np.float64(y))
            a, b = g[:2] + track.GATE_OUTER / 2 * left[:2], g[:2] - track.GATE_OUTER / 2 * left[:2]
            self.gates.append((self.px(a), self.px(b), self.px(g[:2])))
        self.font, self.small = _font(20), _font(14)

    def px(self, xy):  # x forward -> right, y left -> up
        return (self.cx + self.s * (xy[0] - self.c[0]), self.cy - self.s * (xy[1] - self.c[1]))

    def draw(self, t):
        from PIL import Image, ImageDraw

        rec = self.rec
        im = Image.new("RGB", (self.W, self.H), (24, 26, 30))
        d = ImageDraw.Draw(im, "RGBA")
        target = int(rec["gates_cum"][t])
        nxt = int(rec["target"][t]) if rec.get("target") is not None else -1
        for i, (a, b, c) in enumerate(self.gates):
            d.line([a, b], fill=(0, 230, 0, 255) if i == nxt else (255, 0, 204, 255), width=5)
            d.text((c[0] + 7, c[1] + 5), f"G{i + 1}", fill=(255, 140, 230, 255), font=self.small)
        path = [self.px(p) for p in rec["pos"][: t + 1, :2]]
        if len(path) > 1:
            d.line(path, fill=(255, 220, 0, 200), width=2)
        x, y = path[-1]
        fx = ra.quat_to_mat(rec["quat"][t])[:, 0]
        h = np.array([fx[0], fx[1]]) / max(1e-6, math.hypot(fx[0], fx[1]))
        d.line([(x, y), (x + 16 * h[0], y - 16 * h[1])], fill=(255, 60, 30, 255), width=3)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255, 255))
        sp = np.linalg.norm(rec["pos"][min(t + 1, len(rec["pos"]) - 1)] - rec["pos"][max(t - 1, 0)]) * rec["hz"] / 2
        d.text((16, 12), rec["name"], fill=(255, 255, 255, 255), font=self.font)
        d.text((16, 42), f"t {t / rec['hz']:5.2f} s    gates {target}    {sp:3.1f} m/s    z {rec['pos'][t, 2]:.2f} m",
               fill=(220, 220, 220, 255), font=self.small)
        legend = "bars: commanded sticks" + ("   white tick: teacher" if rec["label"] is not None else "")
        d.text((16, 64), legend, fill=(160, 160, 160, 255), font=self.small)
        return np.asarray(im)


def _resize(img, size):
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize(size, Image.BILINEAR))


def compose(layout, panels, t, rec, tmap):
    """One 1280x720 frame from the annotated per-view frames of step t."""
    out = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), np.uint8)
    if layout == "grid":
        order = [v for v in VIEWS if v in panels]
        cells = [panels[v] for v in order] + [tmap.draw(t)]
        for i, img in enumerate(cells[:4]):
            r_, c_ = divmod(i, 2)
            out[r_ * 360:(r_ + 1) * 360, c_ * 640:(c_ + 1) * 640] = _resize(img, PANEL) if img.shape[:2] != (360, 640) else img
        return out
    # pip: chase (or the first third-person view) full frame, the others as insets
    main = "chase" if "chase" in panels else next(iter(panels))
    out[:] = _resize(panels[main], OUT_SIZE)
    inset = (448, 252)
    spots = [(12, 12), (OUT_SIZE[0] - inset[0] - 12, OUT_SIZE[1] - inset[1] - 12)]
    for (x, y), v in zip(spots, [v for v in VIEWS if v in panels and v != main]):
        out[y - 2:y + inset[1] + 2, x - 2:x + inset[0] + 2] = 255
        out[y:y + inset[1], x:x + inset[0]] = _resize(panels[v], inset)
    return out


class Writer:
    def __init__(self, path, fps):
        import imageio.v2 as imageio

        self.w = imageio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=1,
                                    pixelformat="yuv420p", ffmpeg_params=["-crf", "20", "-preset", "medium"])

    def add(self, img):
        self.w.append_data(np.ascontiguousarray(img))

    def close(self):
        self.w.close()


def contact_sheet(frames, rec, path, n=12, cols=4, tile=(480, 270), max_bytes=1_450_000):
    from PIL import Image

    T = len(frames)
    ts = np.linspace(min(T - 1, int(0.5 * rec["hz"])), T - 1, n).round().astype(int)
    rows = -(-n // cols)
    sheet = np.zeros((rows * tile[1], cols * tile[0], 3), np.uint8)
    for i, t in enumerate(ts):
        img = annotate(_resize(frames[t], tile), f"t {t / rec['hz']:.2f} s   gates {int(rec['gates_cum'][t])}", font_size=14)
        r_, c_ = divmod(i, cols)
        sheet[r_ * tile[1]:(r_ + 1) * tile[1], c_ * tile[0]:(c_ + 1) * tile[0]] = img
    for q in (88, 80, 72, 64, 55):
        Image.fromarray(sheet).save(path, quality=q)
        if Path(path).stat().st_size <= max_bytes:
            break
    return [int(t) for t in ts]


def write_outputs(rec, fr, info, out, stem, views, layout, sheet_name=None, log=print):
    """Annotate, compose and encode one record's videos, sheet and json."""
    T, fps = len(rec["pos"]), rec["hz"]
    paths = {}
    cams = {}
    if "overview" in fr:
        o = info["cameras"]["overview"]
        cams["overview"] = (np.array(o["pos"]), np.array(o["quat_xyzw"]))
    tmap = TopDownMap(rec, PANEL)
    writers = {v: Writer(out / f"{stem}_{v}.mp4", fps) for v in views if v in fr}
    combo = Writer(out / f"{stem}_combined.mp4", fps)
    titles = {"fpv": "PILOT VIEW (FPV)", "chase": "CHASE", "overview": "OVERVIEW"}
    t0 = time.time()
    for t in range(T):
        panels = {}
        for v in views:
            if v not in fr:
                continue
            img = fr[v][t]
            if v == "fpv":
                img = draw_bars(img, rec["action"][t], None if rec["label"] is None else rec["label"][t])
            elif v == "overview":
                img = mark_drone(img, rec, t, *cams["overview"])
            img = annotate(img, titles[v], caption(rec, t))
            panels[v] = img
            writers[v].add(img)
        combo.add(compose(layout, panels, t, rec, tmap))
    for w in writers.values():
        w.close()
    combo.close()
    paths.update({v: str(out / f"{stem}_{v}.mp4") for v in writers}, combined=str(out / f"{stem}_combined.mp4"))
    log(f"  encoded {T} frames x {len(writers) + 1} videos in {time.time() - t0:.1f} s")
    if "chase" in fr:
        sheet = out / (sheet_name or f"{stem}_sheet.jpg")
        info["sheet"] = {"path": str(sheet), "steps": contact_sheet(fr["chase"], rec, sheet), "bytes": sheet.stat().st_size}
    if "chase" in fr:  # sanity: the drone's centre stays inside the chase image

        cp, cq, _ = ra.chase_camera(rec["pos"], rec["quat"], rec["gates_pos"], rec["gates_yaw"], hz=rec["hz"])
        H, W = fr["chase"].shape[1:3]
        inside = [bool(((uv := ra.project_points(rec["pos"][t], cp[t], cq[t], W, H, ra.TP_HFOV_DEG)[0][0]) >= 0).all()
                       and uv[0] < W and uv[1] < H) for t in range(T)]
        info["chase_drone_centre_in_image_frac"] = round(float(np.mean(inside)), 4)
    info.update(steps=T, seconds=T / fps, gates=int(rec["passed"].sum()), name=rec["name"], videos=paths, layout=layout)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="fpv,chase,overview")
    ap.add_argument("--layout", default="grid", choices=("grid", "pip"))
    ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--ipg", type=int, default=4, help="Unreal instances per GPU and renderer")
    ap.add_argument("--look", default="stock")
    ap.add_argument("--drone-scale", type=float, default=ra.DRONE_SCALE)
    ap.add_argument("--sheet-name", default=None, help="file name of the contact sheet (one record only)")
    ap.add_argument("--caption", default=None,
                    help="caption instead of the record's name; {stem} = record file name, {name} = record name")
    ap.add_argument("--subdirs", action="store_true", help="outputs of each record in <out>/<stem>/")
    ap.add_argument("--highlight", default="chase,overview",
                    help="views showing the record's target gate in the highlight colour (from `target`, else derived "
                         "from `passed`); 'none' = off. The FPV view only if listed (for students trained with it)")
    ap.add_argument("--summary-sheet", default=None,
                    help="also write one sheet with a chase and an overview frame (mid-flight) per record")
    args = ap.parse_args()
    make_videos([Path(p) for p in args.records], Path(args.out), args.views.split(","), args.layout, args.gpus,
                args.ipg, args.look, args.drone_scale, args.sheet_name, caption_fmt=args.caption, subdirs=args.subdirs,
                summary_sheet=args.summary_sheet, highlight=[v for v in args.highlight.split(",") if v in VIEWS])


def write_summary_sheet(path, rows, max_bytes=1_450_000):
    """rows: [(label, [image, ...])] -> one jpeg, one row per record."""
    from PIL import Image

    tile = (640, 360)
    cols = max(len(imgs) for _, imgs in rows)
    sheet = np.zeros((len(rows) * tile[1], cols * tile[0], 3), np.uint8)
    for r_, (lab, imgs) in enumerate(rows):
        for c_, img in enumerate(imgs):
            sheet[r_ * tile[1]:(r_ + 1) * tile[1], c_ * tile[0]:(c_ + 1) * tile[0]] = _resize(img, tile)
    for q in (88, 80, 72, 64, 55):
        Image.fromarray(sheet).save(path, quality=q)
        if Path(path).stat().st_size <= max_bytes:
            break
    return Path(path).stat().st_size


def make_videos(record_paths, out, views, layout="grid", gpus=(4, 5), ipg=4, look="stock",
                drone_scale=ra.DRONE_SCALE, sheet_name=None, log=print, caption_fmt=None, subdirs=False, summary_sheet=None,
                highlight=("chase", "overview")):
    """Render and encode all outputs for the given flight records; returns the info dicts."""
    views = [v for v in VIEWS if v in views]
    if not views:
        raise ValueError(f"--views must name some of {VIEWS}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = [load_record(p) for p in record_paths]
    if caption_fmt:
        for p, rec in zip(record_paths, records):
            rec["name"] = caption_fmt.format(stem=Path(p).stem, name=rec["name"])
    size = (1280, 720) if layout == "pip" else PANEL
    frames, infos = render_records(records, views, size, list(gpus), ipg, look, drone_scale, log, tuple(highlight))
    results, sheet_rows = [], []
    for p, rec, fr, info in zip(record_paths, records, frames, infos):
        o = out / p.stem if subdirs else out
        o.mkdir(parents=True, exist_ok=True)
        info = write_outputs(rec, fr, info, o, p.stem, views, layout, sheet_name if len(records) == 1 else None, log)
        info.update(record=str(p), drone_scale=drone_scale, render_size=list(size))
        (o / f"{p.stem}_video.json").write_text(json.dumps(info, indent=1))
        results.append(info)
        log(json.dumps({k: info[k] for k in ("name", "steps", "gates", "render_s")}))
        if summary_sheet:
            t = len(rec["pos"]) // 2
            imgs = []
            for v in ("chase", "overview"):
                if v in fr:
                    img = fr[v][t] if v == "chase" else mark_drone(fr[v][t], rec, t, np.array(info["cameras"]["overview"]["pos"]),
                                                                   np.array(info["cameras"]["overview"]["quat_xyzw"]))
                    imgs.append(annotate(img, f"{v.upper()}  {rec['name']}", caption(rec, t).split("   ", 1)[1]))
            sheet_rows.append((rec["name"], imgs))
    if summary_sheet and sheet_rows:
        log(f"summary sheet {summary_sheet}: {write_summary_sheet(summary_sheet, sheet_rows)} bytes")
    return results



if __name__ == "__main__":
    main()
