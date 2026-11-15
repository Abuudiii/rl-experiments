"""The camera-only student flies alone on one gate layout; its flight is recorded and rendered as video.

    .venv-sim/bin/python scripts/student_flight_video.py runs/student/<run>/ckpt_latest.pkl \
        --renderer airsim --renderer-args '{"gpus":[4,5],"instances_per_gpu":4}' --sim-device cpu \
        --scenario circuit --out runs/student/<run>/third_person

The student flies from its own camera frames (``--renderer``, at the checkpoint's resolution) on the
layout ``--scenario`` (a name of `fpv.track.SCENARIOS` or a track.json path; default the nominal
circuit), nominal physics, one 30 s episode per renderer slot; the slot with the most gates passed
is kept (``--pick worst`` for the fewest). Outputs in --out:

    flight.npz                         flight record (format: scripts/flight_video.py), teacher labels included
    flight_combined.mp4, flight_{fpv,chase,overview}.mp4, sheet.jpg, flight_video.json
                                       rendered by scripts/flight_video.py (Unreal, --video-gpus)
    trajectory_student_vs_teacher.png  top-down paths of the student and of the teacher from the same spawn
    flight.json                        summary of the flight and of all slots
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

import flight_video  # noqa: E402
from fpv import dagger, student as st, track  # noqa: E402
from fpv.closed_loop import FlightBatch  # noqa: E402


def resolve_gates(scenario):
    if scenario.endswith(".json"):
        return track.load_track_json(scenario)
    if hasattr(track, "resolve"):
        return track.resolve(scenario)
    if scenario != "circuit":
        raise ValueError(f"unknown scenario {scenario!r} (this fpv.track has only the circuit)")
    return track.CIRCUIT


def flight_batch(n, teacher, seed, device, gates):
    """FlightBatch with nominal physics on the given layout."""
    params = inspect.signature(FlightBatch.__init__).parameters
    if "scenario" in params:
        return FlightBatch(n, teacher, seed=seed, randomize=False, device=device, scenario=gates)
    if gates != track.CIRCUIT:
        raise ValueError("this fpv.closed_loop.FlightBatch only flies the nominal circuit")
    return FlightBatch(n, teacher, seed=seed, randomize=False, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--teacher", default="runs/teacher/cruise_wide_noise2_s1/ckpt_latest.pkl")
    ap.add_argument("--renderer", default="airsim", help="renderer the student flies with: airsim or synth")
    ap.add_argument("--renderer-args", default="{}")
    ap.add_argument("--n-envs", type=int, default=16, help="slots for the synth renderer")
    ap.add_argument("--scenario", default="circuit")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sim-device", default=None, help="Crazyflow device: cpu (faster for few slots) or gpu")
    ap.add_argument("--pick", default="best", choices=("best", "worst", "first"),
                    help="slot whose flight is rendered: most / fewest gates, or slot 0 (fixed before the flight)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="fpv,chase,overview")
    ap.add_argument("--layout", default="grid", choices=("grid", "pip"))
    ap.add_argument("--video-gpus", type=int, nargs="+", default=None, help="default: the renderer's gpus, else 4 5")
    ap.add_argument("--video-ipg", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(args.renderer_args)
    gates = resolve_gates(args.scenario)
    gp, gy = track.circuit_arrays(gates)
    name = Path(args.scenario).stem if args.scenario.endswith(".json") else args.scenario

    cfg, params, _ = st.load_student(args.checkpoint)
    renderer = dagger.make_renderer(args.renderer, args.n_envs, cfg.width, cfg.height, seed=args.seed, **kw)
    n = renderer.n
    runs = {}
    try:
        for who in ("student", "teacher"):
            fb = flight_batch(n, args.teacher, args.seed, args.sim_device, gates)
            runner = st.StudentRunner(cfg, params, n) if who == "student" else None
            m, eps, info = dagger.evaluate(fb, renderer, runner, rng=np.random.default_rng(0), trace=True)
            runs[who] = (m, {e["slot"]: e for e in eps}, info["trace"])
            print(who, {k: m[k] for k in ("frac_1_lap", "frac_3_laps", "gates_per_episode", "lap_time_s", "speed_m_s")})
    finally:
        renderer.close()
    m, eps, tr = runs["student"]
    order = sorted(eps, key=lambda s: eps[s]["gates"])
    slot = {"best": order[-1], "worst": order[0], "first": 0}[args.pick]
    ep = eps[slot]
    T = ep["length"]
    pos = np.stack([t["pos"][slot] for t in tr[:T]])
    quat = np.stack([t["quat"][slot] for t in tr[:T]])
    passed = np.zeros(T, bool)
    passed[[k - 1 for k in ep["pass_steps"] if 0 < k <= T]] = True  # pass_steps: episode step count after the passing step
    # what the student saw: share of its frames with a green (highlighted) gate on screen
    fr = np.stack([t["frames"][slot] for t in tr[:T]]).astype(np.int16) if tr[0]["frames"] is not None else None
    green_px = None if fr is None else ((fr[..., 1] > 100) & (fr[..., 1] > 1.5 * np.maximum(fr[..., 0], fr[..., 2]))).sum((1, 2))
    rec_path = flight_video.save_record(
        out / "flight.npz", pos=pos, quat=quat, action=np.stack([t["action"][slot] for t in tr[:T]]),
        label=np.stack([t["label"][slot] for t in tr[:T]]), gates_pos=gp, gates_yaw=gy, passed=passed,
        target=np.stack([t["target"][slot] for t in tr[:T]]),
        name=f"{name}: camera-only student", hz=fb.cfg.policy_hz)

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
    for g in range(len(gy)):
        _, left = track.gate_axes(gy[g])
        for a, b in ((track.GATE_OPENING / 2, track.GATE_OUTER / 2), (-track.GATE_OUTER / 2, -track.GATE_OPENING / 2)):
            ax.plot([gp[g, 0] + a * left[0], gp[g, 0] + b * left[0]], [gp[g, 1] + a * left[1], gp[g, 1] + b * left[1]], color="m", lw=4)
        ax.annotate(f"G{g + 1}", gp[g, :2], textcoords="offset points", xytext=(8, 8), color="m")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{name}, top-down: camera-only student vs teacher")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.savefig(out / "trajectory_student_vs_teacher.png", dpi=130, bbox_inches="tight")

    gpus = args.video_gpus or list(kw.get("gpus", [4, 5]))
    hl = dagger.highlight_on(renderer)
    video = flight_video.make_videos([rec_path], out, args.views.split(","), args.layout, gpus, args.video_ipg,
                                     kw.get("look", "stock"), sheet_name="sheet.jpg",
                                     highlight=("fpv", "chase", "overview") if hl else ())[0]  # fmt: skip
    info = dict(checkpoint=args.checkpoint, teacher=args.teacher, scenario=name, seed=args.seed, slot=int(slot),
                steps=int(T), seconds=T / 50, gates=ep["gates"], outcome=dagger.OUTCOMES[ep["outcome"]],
                record=str(rec_path), video=video, all_slots=m, pick=args.pick, highlight_target=hl,
                student_frames_with_green_gate=None if green_px is None else float((green_px >= 4).mean()))  # fmt: skip
    (out / "flight.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("scenario", "slot", "steps", "gates", "outcome")}, indent=1))


if __name__ == "__main__":
    main()
