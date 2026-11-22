"""Stage-5 flight videos from the frames kept by `scripts/eval_stage5.py run --video ...` (video_*.npz).

Left: the FPV frame the detector saw (2x), every detection above threshold (cyan quad + confidence), the gate the
estimator believes is next (its estimated corners projected with the camera model; yellow = it is the true next gate
and within the stage-4 "pose right" tolerance, red = wrong gate or wrong pose), true next gate index in the caption.
Right: top-down view: the flight's true (randomised) gates, true next gate in green, the believed next gate (estimate,
placed in the world with the TRUE drone pose, so only the estimator's relative error shows) as a yellow/red marker,
the path so far. The video ends 0.5 s after the flight ends (crash) or at 30 s.

    .venv-rocm/bin/python scripts/stage5_video.py <video_npz> --out <dir> [--names circuit]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fpv import track  # noqa: E402
from fpv.perception import labels  # noqa: E402

OUTCOME = {0: "time limit", 1: "gate crash", 2: "ground", 3: "out of bounds"}


def qmat(q):
    return np.asarray(labels.quat_to_mat(np.asarray(q, np.float64)))


def body_to_px(pb):
    cam = pb @ labels._BODY_TO_CAM.T.astype(np.float64)
    return np.asarray(labels.project(cam)), cam[..., 2]


def make_video(z, s, out_path, title, fps=50):
    T = z["frames"].shape[0]
    alive = z["alive"][:, s]
    end = int(np.flatnonzero(~alive)[0]) if (~alive).any() else T - 1  # first step after the flight ended
    last = min(T, end + 25)
    gp, gy = z["gates_pos"][0, s], z["gates_yaw"][0, s]
    _, left = track.gate_axes(np.asarray(gy, np.float64))
    pos = z["pos"][:, s].astype(np.float64)
    fig = plt.figure(figsize=(12.8, 4.6), dpi=100)
    ax_i = fig.add_axes([0.0, 0.0, 0.5, 0.84])
    ax_t = fig.add_axes([0.55, 0.06, 0.42, 0.78])
    ax_i.axis("off")
    im = ax_i.imshow(np.zeros((360, 640, 3), np.uint8), extent=(-0.5, 319.5, 179.5, -0.5))
    ax_i.set_xlim(-0.5, 319.5)
    ax_i.set_ylim(179.5, -0.5)
    half = track.GATE_OUTER / 2
    gate_lines = []
    for g in range(6):
        a, b = gp[g, :2] - half * left[g, :2], gp[g, :2] + half * left[g, :2]
        (ln,) = ax_t.plot([a[0], b[0]], [a[1], b[1]], "-", color="m", lw=3)
        ax_t.text(gp[g, 0], gp[g, 1] + 0.4, str(g), color="k", fontsize=8, ha="center")
        gate_lines.append(ln)
    lo, hi = gp[:, :2].min(0) - 3, gp[:, :2].max(0) + 3
    ax_t.set_xlim(lo[0], hi[0])
    ax_t.set_ylim(lo[1], hi[1])
    ax_t.set_aspect("equal")
    ax_t.set_title("top-down (m): path, true next gate green, believed next gate x", fontsize=9)
    (path_ln,) = ax_t.plot([], [], "-", color="tab:blue", lw=1)
    (drone_pt,) = ax_t.plot([], [], "o", color="tab:blue", ms=5)
    (belief_pt,) = ax_t.plot([], [], "x", color="gold", ms=10, mew=3)
    det_lines = [ax_i.plot([], [], "-", color="cyan", lw=1.2)[0] for _ in range(z["corners"].shape[2])]
    det_txt = [ax_i.text(0, 0, "", color="cyan", fontsize=7) for _ in range(z["corners"].shape[2])]
    (bel_ln,) = ax_i.plot([], [], "--", color="gold", lw=1.5)
    cap = fig.text(0.01, 0.875, "", fontsize=10, family="monospace")
    fig.text(0.01, 0.955, title, fontsize=10)
    w = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    gates_passed = 0
    for t in range(last):
        im.set_data(np.repeat(np.repeat(z["frames"][t, s], 2, 0), 2, 1))
        for k in range(len(det_lines)):
            c, cf = z["corners"][t, s, k], z["conf"][t, s, k]
            if cf > 0:
                q = np.concatenate([c, c[:1]])
                det_lines[k].set_data(q[:, 0], q[:, 1])
                det_txt[k].set_position((c[:, 0].min(), c[:, 1].min() - 2))
                det_txt[k].set_text(f"{cf:.2f}")
            else:
                det_lines[k].set_data([], [])
                det_txt[k].set_text("")
        tgt, nxt, lk = int(z["target"][t, s]), int(z["est_nxt"][t, s]), bool(z["locked"][t, s])
        cb = z["belief_corners_b"][t, s].astype(np.float64)
        R = qmat(z["quat"][t, s])
        bw = pos[t] + cb @ R.T  # believed corners in the world (true drone pose)
        bc = bw.mean(0)
        true_c = gp[tgt]
        ok = lk and nxt == tgt and np.linalg.norm(bc - true_c) <= max(0.5, 0.15 * np.linalg.norm(true_c - pos[t]))
        col = "gold" if ok else "red"
        px, zc = body_to_px(cb)
        if lk and (zc > 0.05).all():
            q = np.concatenate([px, px[:1]])
            bel_ln.set_data(q[:, 0], q[:, 1])
        else:
            bel_ln.set_data([], [])
        bel_ln.set_color(col)
        belief_pt.set_data([bc[0]], [bc[1]] if lk else [])
        if not lk:
            belief_pt.set_data([], [])
        belief_pt.set_color(col)
        for g, ln in enumerate(gate_lines):
            ln.set_color("tab:green" if g == tgt else "m")
        path_ln.set_data(pos[: t + 1, 0], pos[: t + 1, 1])
        drone_pt.set_data([pos[t, 0]], [pos[t, 1]])
        gates_passed += int(z["passed"][t, s])
        oc = int(z["outcome"][:, s].max()) if "outcome" in z else -1
        status = "flying" if t < end else f"ENDED: {OUTCOME.get(oc, 'flight over')}"
        cap.set_text(f"t {t * 0.02:5.2f} s  gates {gates_passed:2d}  next: true {tgt} believed {nxt if lk else '-'}"
                     f"{'' if ok or not lk else ' (WRONG)'}  dets {int((z['conf'][t, s] > 0).sum())}  {status}")  # fmt: skip
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    w.close()
    plt.close(fig)
    return dict(path=str(out_path), steps=last, flight_end_step=end, gates=gates_passed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in args.npz:
        z = dict(np.load(f, allow_pickle=True))
        meta = json.loads(str(z["meta"]))
        for j, s in enumerate(range(z["frames"].shape[1])):
            name = f"{meta['block']}_flight{meta['idx'][j]}"
            title = f"stage 5, {meta['block']} flight {meta['idx'][j]} (first flights, not picked): rendered frame -> detector E thr {meta['thr']} -> estimator -> policy"
            r = make_video(z, s, out / f"{name}.mp4", title)
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
