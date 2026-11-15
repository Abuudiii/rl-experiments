"""Do stored training frames show the target (next) gate in green, and no other gate?

    .venv-sim/bin/python scripts/check_highlight_frames.py --data <dir> [<dir> ...] --shards 8 --per-shard 300 \
        --out runs/student/<run>/highlight_check.json

For sampled frames of DAgger shards (`fpv.dagger` format) every pixel is assigned to the gate it
shows, from the stored pose and gate layout only (not from the frame): ``--labels synth`` (default)
uses the hit map of `fpv.render_synth.SynthRenderer` for the same pose and gates (exact occlusion
between gates; its blocks are removed, since the frame's own blocks are not recorded);
``--labels poly`` rasterises each gate ring with `project_points` + polygon filling, nearest gate centre
wins. A gate counts as visible with >= ``--min-px`` pixels inside the image and as green
when, among its pixels, those classified green (G dominant) outnumber those classified magenta
(R and B dominant, the normal gate colour). Reported per data directory: the share of frames with
a visible target whose target is green, the share of frames with a visible non-target gate where
some non-target gate is green ("strict": >= 10 green pixels and >= half of that gate's pixels), the
share of green pixels outside the target's frame (grown by 2 px) and on another gate's frame, the
mean green / magenta pixel fractions on target and other gates, and all of it by steps since the
target changed (0 = the first frame after a gate pass; ``--near-change k`` adds every frame with
<= k). ``--sheet n``: up to n frames with a green non-target gate, drawn with the projected gate
outlines (target white), to ``<out stem>_sheet<i>.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from fpv import track  # noqa: E402
from fpv.render_synth import project_points  # noqa: E402


def gate_label_map(pos, quat, gp, gy, w, h):
    """int map [h, w]: index of the nearest gate whose frame covers the pixel, -1 none."""
    outer = track.gate_corners(gp, gy, track.GATE_OUTER)  # (6, 4, 3)
    inner = track.gate_corners(gp, gy, track.GATE_OPENING)
    uv_o, d_o = project_points(pos, quat, outer.reshape(-1, 3), w, h)
    uv_i, d_i = project_points(pos, quat, inner.reshape(-1, 3), w, h)
    _, d_c = project_points(pos, quat, gp, w, h)
    lab = np.full((h, w), -1, int)
    depth = np.full((h, w), np.inf)
    for g in range(len(gp)):
        if (d_o[4 * g : 4 * g + 4] <= 0.05).any() or (d_i[4 * g : 4 * g + 4] <= 0.05).any():
            continue  # partly behind the camera: not rasterised (counted as not visible)
        im = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(im)
        dr.polygon([tuple(p + 0.5) for p in uv_o[4 * g : 4 * g + 4]], fill=1)
        dr.polygon([tuple(p + 0.5) for p in uv_i[4 * g : 4 * g + 4]], fill=0)
        m = (np.asarray(im) > 0) & (d_c[g] < depth)
        lab[m], depth[m] = g, d_c[g]
    return lab


def classify(frame):
    f = frame.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    green = (g > 60) & (g > 1.3 * np.maximum(r, b))
    magenta = (np.maximum(r, b) > 60) & (np.minimum(r, b) > 1.2 * g)
    return green, magenta


def steps_since_change(target, t, valid):
    """[T, S] steps since the target gate of the slot last changed (0 = first frame with this target,
    i.e. the state right after a gate pass or an episode start); -1 where unknown (shard starts mid-leg)."""
    since = np.where(t[0] == 0, 0, -1)[None].repeat(target.shape[0], 0)
    for k in range(1, target.shape[0]):
        changed = (t[k] == 0) | (target[k] != target[k - 1]) | ~valid[k - 1]
        since[k] = np.where(changed, 0, np.where(since[k - 1] >= 0, since[k - 1] + 1, -1))
    return since


SINCE_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 5), (6, 10**9))


def check_dir(d: Path, shards: int, per_shard: int, min_px: int, rng, labels: str = "synth", near_change: int = -1,
              sheet_n: int = 0):
    """``near_change >= 0``: also every frame within that many steps of a target change. ``sheet_n``:
    keep up to that many frames where a non-target gate is green (for `draw_sheet`)."""
    files = sorted(p for p in d.glob("r*_c*_*.npz") if ".tmp" not in p.name)
    pick = [files[i] for i in sorted(rng.choice(len(files), min(shards, len(files)), replace=False))]
    c = dict(frames=0, target_visible=0, target_green=0, other_visible=0, other_green=0, frames_no_gate_visible=0)
    frac = dict(target_green=[], target_magenta=[], other_green=[], other_magenta=[])
    green_others = []  # (gate - target) mod 6, pixels, green share, of non-target gates classified green
    by_since = {f"{a}-{b}" if b < 10**9 else f">={a}": dict(target_visible=0, target_green=0, other_visible=0,
                                                             other_green=0, prev_gate_green=0)
                for a, b in SINCE_BUCKETS}  # fmt: skip
    per_file, offenders = [], []
    for p in pick:
        z = np.load(p)
        valid = z["valid"]
        ti, si = np.nonzero(valid)
        pos, quat, gpos, gyaw, tgt = z["pos"], z["quat"], z["gates_pos"], z["gates_yaw"], z["target"]
        since = steps_since_change(tgt.astype(int), z["t"].astype(int), valid)
        sel = rng.choice(len(ti), min(per_shard, len(ti)), replace=False)
        if near_change >= 0:  # every frame within near_change steps of a target change, plus the random sample
            near = np.flatnonzero((since[ti, si] >= 0) & (since[ti, si] <= near_change))
            sel = np.union1d(sel, near)
        frames = z["frames"]
        h, w = frames.shape[2:4]
        fc = dict(frames=0, target_visible=0, target_green=0, other_visible=0, other_green=0)
        ts, ss = ti[sel], si[sel]
        if labels == "synth":  # exact occlusion: hit map of the synthetic renderer for the same poses and gates
            from fpv.render_synth import SynthRenderer

            rr = SynthRenderer(min(len(sel), 512), w, h, seed=0)  # gates only: its random blocks are moved away below
            hit = []
            for a in range(0, len(sel), rr.n):
                k = min(rr.n, len(sel) - a)
                idx = np.arange(rr.n) % k  # pad the last chunk by repeating its slots
                rr.set_gates(np.arange(rr.n), gpos[ts[a + idx], ss[a + idx]], gyaw[ts[a + idx], ss[a + idx]])
                rr.scene["b_pos"][..., 2] = -1e3  # the frames' own blocks are unknown here: label gates only
                hit.append(np.asarray(rr.render_ids(pos[ts[a + idx], ss[a + idx]], quat[ts[a + idx], ss[a + idx]])[1])[:k])
            hit = np.concatenate(hit)
        for j, k in enumerate(sel):
            t, s = ti[k], si[k]
            lab = hit[j] if labels == "synth" else gate_label_map(pos[t, s], quat[t, s], gpos[t, s], gyaw[t, s], w, h)
            lab = np.where(lab < gpos.shape[2], lab, -1)  # blocks / sky / ground: no gate
            green, magenta = classify(frames[t, s])
            target = int(tgt[t, s])
            c["frames"] += 1
            fc["frames"] += 1
            # green pixels outside the target's visible frame (grown by 2 px), share of all green pixels
            own = lab == target
            for _ in range(2):
                own = own | np.roll(own, 1, 0) | np.roll(own, -1, 0) | np.roll(own, 1, 1) | np.roll(own, -1, 1)
            c["green_px"] = c.get("green_px", 0) + int(green.sum())
            c["green_px_outside_target"] = c.get("green_px_outside_target", 0) + int((green & ~own).sum())
            c["green_px_on_other_gate"] = c.get("green_px_on_other_gate", 0) + int((green & ~own & (lab >= 0)).sum())
            tg_before = c["target_green"]
            any_vis = False
            other_vis, other_green, prev_green, strict_green = False, False, False, False
            for g in range(gpos.shape[2]):
                m = lab == g
                npx = int(m.sum())
                if npx < min_px:
                    continue
                any_vis = True
                ng, nm = int(green[m].sum()), int(magenta[m].sum())
                is_green = ng > nm and ng + nm >= 3
                if g == target:
                    c["target_visible"] += 1
                    fc["target_visible"] += 1
                    c["target_green"] += is_green
                    fc["target_green"] += is_green
                    frac["target_green"].append(ng / npx)
                    frac["target_magenta"].append(nm / npx)
                else:
                    other_vis = True
                    other_green |= is_green
                    if is_green:
                        green_others.append(((g - target) % 6, npx, ng / npx, int((lab == target).sum())))
                        prev_green |= (g - target) % 6 == 5
                        strict_green |= ng >= 10 and ng >= 0.5 * npx
                    frac["other_green"].append(ng / npx)
                    frac["other_magenta"].append(nm / npx)
            c["other_visible"] += other_vis
            c["other_green"] += other_green
            c["other_green_strict"] = c.get("other_green_strict", 0) + strict_green
            fc["other_visible"] += other_vis
            fc["other_green"] += other_green
            c["frames_no_gate_visible"] += not any_vis
            if since[t, s] >= 0:
                a, b = next(x for x in SINCE_BUCKETS if x[0] <= since[t, s] <= x[1])
                bs = by_since[f"{a}-{b}" if b < 10**9 else f">={a}"]
                tv = int((lab == target).sum()) >= min_px
                bs["target_visible"] += tv
                bs["target_green"] += tv and c["target_green"] > tg_before
                bs["other_visible"] += other_vis
                bs["other_green"] += other_green
                bs["prev_gate_green"] += prev_green
            if other_green and len(offenders) < sheet_n:
                offenders.append(dict(frame=frames[t, s].copy(), pos=pos[t, s], quat=quat[t, s], gp=gpos[t, s],
                                      gy=gyaw[t, s], target=target, since=int(since[t, s]), file=p.name))  # fmt: skip
        per_file.append(dict(file=p.name, **fc))
        print(p.name, fc, flush=True)
    out = dict(
        dir=str(d), files_sampled=len(pick), **c,
        share_target_green_of_visible_target=c["target_green"] / max(c["target_visible"], 1),
        share_any_other_gate_green_of_frames_with_visible_other=c["other_green"] / max(c["other_visible"], 1),
        share_other_gate_green_strict=c.get("other_green_strict", 0) / max(c["other_visible"], 1),
        share_green_pixels_outside_target_frame=c.get("green_px_outside_target", 0) / max(c.get("green_px", 0), 1),
        share_green_pixels_on_other_gate=c.get("green_px_on_other_gate", 0) / max(c.get("green_px", 0), 1),
        by_steps_since_target_change={k: dict(v, share_target_green=v["target_green"] / max(v["target_visible"], 1),
                                              share_other_green=v["other_green"] / max(v["other_visible"], 1))
                                      for k, v in by_since.items()},
        mean_pixel_fraction={k: float(np.mean(v)) if v else None for k, v in frac.items()},
        green_non_target=dict(
            by_offset_from_target={int(o): sum(x[0] == o for x in green_others) for o in range(1, 6)},
            median_pixels=float(np.median([x[1] for x in green_others])) if green_others else None,
            median_green_share=float(np.median([x[2] for x in green_others])) if green_others else None,
            with_target_visible=float(np.mean([x[3] >= min_px for x in green_others])) if green_others else None),
        per_file=per_file,
    )  # fmt: skip
    return out, offenders


def draw_sheet(offenders, path, cols: int = 4, scale: int = 2):
    """Offending frames with every gate's projected outer / opening outline (target white, others
    yellow, index next to it), scaled by ``scale``."""
    if not offenders:
        return
    h, w = offenders[0]["frame"].shape[:2]
    rows = (len(offenders) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w * scale, rows * (h * scale + 14)), (0, 0, 0))
    for i, o in enumerate(offenders):
        im = Image.fromarray(o["frame"]).resize((w * scale, h * scale), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        for size in (track.GATE_OUTER, track.GATE_OPENING):
            cor = track.gate_corners(o["gp"], o["gy"], size)
            uv, dep = project_points(o["pos"], o["quat"], cor.reshape(-1, 3), w, h)
            for g in range(len(o["gp"])):
                col = (255, 255, 255) if g == o["target"] else (255, 230, 0)
                for a in range(4):
                    p, q = 4 * g + a, 4 * g + (a + 1) % 4
                    if dep[p] > 0.05 and dep[q] > 0.05:
                        dr.line([tuple((uv[p] + 0.5) * scale), tuple((uv[q] + 0.5) * scale)], fill=col, width=1)
                if size is track.GATE_OUTER and (dep[4 * g : 4 * g + 4] > 0.05).all():
                    dr.text(tuple(uv[4 * g : 4 * g + 4].mean(0) * scale), str(g), fill=col)
        x, y = (i % cols) * w * scale, (i // cols) * (h * scale + 14)
        sheet.paste(im, (x, y + 14))
        ImageDraw.Draw(sheet).text((x + 2, y + 1), f"target {o['target']}  since {o['since']}  {o['file']}",
                                   fill=(255, 255, 255))  # fmt: skip
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--per-shard", type=int, default=300)
    ap.add_argument("--min-px", type=int, default=6)
    ap.add_argument("--labels", default="synth", choices=("synth", "poly"))
    ap.add_argument("--near-change", type=int, default=-1,
                    help="also check every frame within this many steps of a target change (gate pass)")
    ap.add_argument("--sheet", type=int, default=0, help="contact sheet of up to this many offending frames per dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = dict(min_px=args.min_px, labels=args.labels, near_change=args.near_change, dirs=[])
    for i, d in enumerate(args.data):
        r, off = check_dir(Path(d), args.shards, args.per_shard, args.min_px, rng, args.labels, args.near_change, args.sheet)
        res["dirs"].append(r)
        if args.sheet:
            draw_sheet(off, Path(args.out).with_name(Path(args.out).stem + f"_sheet{i}.png"))
    for r in res["dirs"]:
        print(json.dumps({k: v for k, v in r.items() if k != "per_file"}))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
