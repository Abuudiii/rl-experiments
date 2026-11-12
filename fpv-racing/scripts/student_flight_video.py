"""FPV video of a student-flown flight on the nominal circuit + top-down student / teacher figure.

    CUDA_VISIBLE_DEVICES=2 .venv-sim/bin/python scripts/student_flight_video.py runs/student/<run>/ckpt_latest.pkl \
        --renderer synth --out runs/student/<run>

The student flies alone from the camera on the nominal circuit (one episode per renderer slot, 30 s);
the slot with the most gates passed is shown (``--pick worst`` for the fewest). The frames it flew
on are low resolution, so its trajectory is rendered again at 640x360 with ``render_trajectories``
(same poses, same layout) for ``fpv_flight.mp4``; the four stick values the student commanded are
drawn as bars (thrust, roll, pitch, yaw rate in [-1, 1]; white tick = the teacher's label).
``trajectory_student_vs_teacher.png``: top-down paths of the student and of the teacher flown from
the same spawn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

from fpv import dagger, student as st, track  # noqa: E402
from fpv.closed_loop import FlightBatch  # noqa: E402

NAMES = ("THR", "ROLL", "PITCH", "YAW")


def draw_bars(img, action, label):
    """Four vertical stick bars in the lower-left corner (in place)."""
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
        t = mid - float(np.clip(label[i], -1, 1)) * bh / 2
        d.line([x - 3, t, x + bw + 3, t], fill=(255, 255, 255, 255), width=2)
        d.text((x, y0 + 3), NAMES[i], fill=(255, 255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--teacher", default="runs/teacher/cruise_wide_noise2_s1/ckpt_latest.pkl")
    ap.add_argument("--renderer", default="synth")
    ap.add_argument("--renderer-args", default="{}")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sim-device", default=None, help="Crazyflow device: cpu (faster for few slots) or gpu")
    ap.add_argument("--pick", default="best", choices=("best", "worst"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-size", default="640x360")
    args = ap.parse_args()
    magenta = lambda f: float(((f[..., 0] > 60) & (f[..., 0] > 1.6 * f[..., 1].astype(float)) & (f[..., 2] > 1.3 * f[..., 1].astype(float))).mean())  # noqa: E731
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vw, vh = (int(v) for v in args.video_size.split("x"))
    kw = json.loads(args.renderer_args)

    cfg, params, _ = st.load_student(args.checkpoint)
    renderer = dagger.make_renderer(args.renderer, args.n_envs, cfg.width, cfg.height, seed=args.seed, **kw)
    n = renderer.n
    runs = {}
    for who in ("student", "teacher"):
        fb = FlightBatch(n, args.teacher, seed=args.seed, randomize=False, device=args.sim_device)
        runner = st.StudentRunner(cfg, params, n) if who == "student" else None
        m, eps, info = dagger.evaluate(fb, renderer, runner, rng=np.random.default_rng(0), trace=True)
        runs[who] = (m, {e["slot"]: e for e in eps}, info["trace"])
        print(who, {k: m[k] for k in ("frac_1_lap", "frac_3_laps", "gates_per_episode", "lap_time_s", "speed_m_s")})
    m, eps, tr = runs["student"]
    order = sorted(eps, key=lambda s: eps[s]["gates"])
    slot = order[-1] if args.pick == "best" else order[0]
    ep = eps[slot]
    T = ep["length"]
    pos = np.stack([t["pos"][slot] for t in tr[:T]])
    quat = np.stack([t["quat"][slot] for t in tr[:T]])
    act = np.stack([t["action"][slot] for t in tr[:T]])
    lab = np.stack([t["label"][slot] for t in tr[:T]])
    look = {k: v[slot : slot + 1].copy() for k, v in renderer.scene.items()} if hasattr(renderer, "scene") else None
    gp, gy = track.circuit_arrays()

    # the same flight at video resolution
    if args.renderer == "synth":
        renderer.close()
        hi = dagger.make_renderer("synth", 1, vw, vh, seed=args.seed, chunk=64, **kw)
        hi.set_gates([0], gp[None], gy[None])
        hi.scene = look
        frames = hi.render_trajectories([0], pos[None], quat[None])[0]
    else:
        renderer.close()
        kw1 = dict(kw, gpus=list(kw["gpus"])[:1], instances_per_gpu=1)
        hi = dagger.make_renderer(args.renderer, 1, vw, vh, **kw1)
        hi.set_gates([0], gp[None], gy[None])
        frames = np.array(hi.render_trajectories([0], pos[None], quat[None])[0])
    hi.close()
    import imageio.v2 as imageio

    with imageio.get_writer(out / "fpv_flight.mp4", fps=50, codec="libx264", quality=None, macro_block_size=1,
                            ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p"]) as w:
        for i in range(T):
            w.append_data(draw_bars(frames[i], act[i], lab[i]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    tpos = np.stack([t["pos"][slot] for t in runs["teacher"][2][: runs["teacher"][1][slot]["length"]]])
    ax.plot(tpos[:, 0], tpos[:, 1], color="0.35", lw=1.0, ls="--", label=f"teacher (state based), {runs['teacher'][1][slot]['gates']} gates")
    ax.plot(pos[:, 0], pos[:, 1], color="tab:blue", lw=1.0, alpha=0.9, label=f"student (camera only), {ep['gates']} gates in {T / 50:.1f} s")
    ax.plot(*pos[0, :2], "o", color="tab:green", label="start")
    if ep["outcome"] > 0:
        ax.plot(*pos[-1, :2], "x", color="tab:red", ms=10, mew=2, label=f"crash ({dagger.OUTCOMES[ep['outcome']]})")
    for g in range(6):
        _, left = track.gate_axes(gy[g])
        for a, b in ((track.GATE_OPENING / 2, track.GATE_OUTER / 2), (-track.GATE_OUTER / 2, -track.GATE_OPENING / 2)):
            ax.plot([gp[g, 0] + a * left[0], gp[g, 0] + b * left[0]], [gp[g, 1] + a * left[1], gp[g, 1] + b * left[1]], color="m", lw=4)
        ax.annotate(f"G{g + 1}", gp[g, :2], textcoords="offset points", xytext=(8, 8), color="m")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Nominal circuit, top-down: camera-only student vs teacher")
    ax.legend(loc="center")
    ax.grid(alpha=0.3)
    fig.savefig(out / "trajectory_student_vs_teacher.png", dpi=130, bbox_inches="tight")
    info = dict(slot=int(slot), steps=int(T), seconds=T / 50, gates=ep["gates"], outcome=dagger.OUTCOMES[ep["outcome"]],
                video_frames=int(len(frames)), video_size=[vw, vh], nominal_all_slots=m,
                frame_mean=float(frames.mean()), frame_std=float(frames.std()),
                magenta_frac=magenta(frames[::10]))  # fmt: skip
    (out / "fpv_flight.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
