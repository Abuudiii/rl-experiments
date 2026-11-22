"""Run one matched synthetic DAgger arm, then benchmark its best completed round.

Run detached, with one GPU visible, through scripts/node_step.sh. This script does
not publish or overwrite existing experiments. --final-only repeats only analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from fpv import dagger, student as st, track
from fpv.closed_loop import FlightBatch
from fpv.render_synth import project_points


def failure_breakdown(cfg, params, teacher, seed, n=128, steps=1000):
    """Geometric evidence, not a claim about the student's internal intent."""
    renderer = dagger.make_renderer("synth", n, cfg.width, cfg.height, seed=seed, chunk=64)
    fb = FlightBatch(n, teacher, seed=seed, randomize=True)
    metrics, episodes, info = dagger.evaluate(fb, renderer, st.StudentRunner(cfg, params, n),
                                             steps=steps, trace="noframes", log=dagger.log)
    traces = info.pop("trace")
    pos = np.stack([s["pos"] for s in traces])
    quat = np.stack([s["quat"] for s in traces])
    target = np.stack([s["target"] for s in traces])
    alive = np.stack([s["alive"] for s in traces])
    gp = np.stack([s["gates_pos"] for s in traces])
    gy = np.stack([s["gates_yaw"] for s in traces])
    motion = np.diff(pos, axis=0)
    rel = gp[:-1] - pos[:-1, :, None]
    dist = np.linalg.norm(rel, axis=-1)
    speed = np.linalg.norm(motion, axis=-1) * fb.cfg.policy_hz
    cosine = (rel * motion[:, :, None]).sum(-1) / np.maximum(dist * np.linalg.norm(motion, axis=-1)[..., None], 1e-6)
    tgt_cos = np.take_along_axis(cosine, target[:-1, :, None], -1)[..., 0]
    wrong = np.arange(6) != target[:-1, :, None]
    candidates = (cosine > 0.94) & wrong & (dist > 0.5) & (dist < 8.0)
    heading = candidates.any(-1) & (tgt_cos < 0.5) & (speed > 0.5) & alive[:-1] & alive[1:]
    sustained = np.zeros(n, bool)
    runs = np.zeros(n, int)
    longest = np.zeros(n, int)
    for row in heading:
        runs = np.where(row, runs + 1, 0)
        longest = np.maximum(longest, runs)
        sustained |= runs >= 10
    # Crossings of a non-target opening, in either direction. These are stronger
    # evidence than heading alone and do not depend on the environment's target logic.
    normal = np.stack([np.cos(gy[:-1]), np.sin(gy[:-1]), np.zeros_like(gy[:-1])], -1)
    left = np.stack([-np.sin(gy[:-1]), np.cos(gy[:-1]), np.zeros_like(gy[:-1])], -1)
    r0 = pos[:-1, :, None] - gp[:-1]
    r1 = pos[1:, :, None] - gp[:-1]
    d0, d1 = (r0 * normal).sum(-1), (r1 * normal).sum(-1)
    fraction = -d0 / np.where(np.abs(d1 - d0) > 1e-8, d1 - d0, 1.0)
    at = r0 + fraction[..., None] * (r1 - r0)
    opening = (np.abs((at * left).sum(-1)) < track.GATE_OPENING / 2) & (np.abs(at[..., 2]) < track.GATE_OPENING / 2)
    crossing = (d0 * d1 < 0) & opening & wrong & alive[:-1, :, None] & alive[1:, :, None]
    uv, depth = project_points(pos.reshape(-1, 3), quat.reshape(-1, 4), gp.reshape(-1, 6, 3), cfg.width, cfg.height)
    visible = ((depth > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < cfg.width)
               & (uv[..., 1] >= 0) & (uv[..., 1] < cfg.height)).reshape(*target.shape, 6)
    target_visible = np.take_along_axis(visible, target[..., None], -1)[..., 0]
    crash = np.array([e["outcome"] > 0 for e in sorted(episodes, key=lambda e: e["slot"])])
    counts = np.asarray(metrics["crashes_by_target_gate"])
    result = dict(seed=seed, episodes=n, metrics=metrics, timing=info,
                  crash_share_by_target_gate=(counts / max(counts.sum(), 1)).tolist(),
                  crash_progress=dict(before_first_gate=sum(e["outcome"] > 0 and e["gates"] == 0 for e in episodes),
                                      before_one_lap=sum(e["outcome"] > 0 and e["gates"] < 6 for e in episodes),
                                      after_one_lap=sum(e["outcome"] > 0 and e["gates"] >= 6 for e in episodes),
                                      time_limit_without_three_laps=sum(e["outcome"] == 0 and e["gates"] < 18 for e in episodes)),
                  wrong_gate_heading_proxy=dict(
                      definition="speed>0.5m/s, non-target center 0.5..8m away, velocity cosine>0.94 to it, cosine<0.5 to target, sustained>=0.2s",
                      episodes=int(sustained.sum()), frac_episodes=float(sustained.mean()),
                      crashed_episodes=int((sustained & crash).sum()),
                      longest_heading_s=(longest / fb.cfg.policy_hz).tolist()),
                  non_target_opening_crossings=dict(events=int(crossing.sum()),
                                                     episodes=int(crossing.any((0, 2)).sum())),
                  target_center_visible_fraction=float(target_visible[alive].mean()),
                  episodes_detail=episodes)
    renderer.close()
    return result


def main():
    import platform
    import shlex

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("S", "S_seed2", "S_prev", "S_large"), required=True)
    ap.add_argument("--baseline-config", type=Path, default=Path("runs/student/ssm/baselines/B/config.json"))
    ap.add_argument("--data-root", type=Path, default=Path("/tmp/fpv_student_ssm"))
    ap.add_argument("--stop-at", type=float, required=True)
    ap.add_argument("--rounds", type=int, default=14)
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--sequence-microbatch", type=int, default=0)
    ap.add_argument("--render-chunk", type=int, default=None, help="optional memory-only renderer batching override")
    args = ap.parse_args()
    baseline = json.loads(args.baseline_config.read_text())
    b, c = baseline["args"], baseline["student"]
    seed = b["seed"] + (1 if args.arm == "S_seed2" else 0)
    out = Path(f"runs/student/ssm/{args.arm}")
    data = args.data_root / args.arm / "data"
    renderer_args = json.loads(b["renderer_args"])
    if args.render_chunk is not None:
        renderer_args["chunk"] = args.render_chunk
    teacher = b["teacher"]
    command = [sys.executable, "scripts/run_dagger.py", "--out", str(out), "--data-dir", str(data),
               "--teacher", teacher, "--arch", "ssm", "--renderer", b["renderer"],
               "--renderer-args", json.dumps(renderer_args), "--width", str(c["width"]),
               "--height", str(c["height"]), "--frames", str(c["n_frames"]),
               "--channels", ",".join(map(str, c["channels"])), "--n-envs", str(b["n_envs"]),
               "--steps", b["steps"], "--beta", b["beta"], "--train-steps", b["train_steps"],
               "--batch", str(c["batch"]), "--lr", str(c["lr"]), "--aux-weight", str(c["aux_weight"]),
               "--prev-action-drop", str(c["prev_action_drop"]), "--recent-weight", str(b["recent_weight"]),
               "--extra-frac", str(b["extra_frac"]), "--teacher-noise", str(b["teacher_noise"]),
               "--burst-p", str(b["burst_p"]), "--burst-mag", str(b["burst_mag"]),
               "--wide-spawn-p", str(b["wide_spawn_p"]), "--eval-episodes", str(b["eval_episodes"]),
               "--eval-steps", str(b["eval_steps"]), "--eval-scenarios", b["eval_scenarios"],
               "--scenario-episodes", str(b["scenario_episodes"]), "--seed", str(seed),
               "--eval-seed", str(b["eval_seed"]), "--sequence-length", "32", "--burn-in", "16",
               "--sequence-batch", "8", "--sequence-microbatch", str(args.sequence_microbatch),
               "--ssm-input-frames", "2" if args.arm == "S_prev" else "1",
               "--ssm-state", "64" if args.arm == "S_large" else "16", "--ssm-layers", "1",
               "--memory-map", "--disk-max-used-frac", "0.9", "--atomic-results",
               "--stop-at", str(args.stop_at), "--resume"]
    # Historical B completed0-2, then restarted at3 with collect seed300300,
    # NumPy mixing seed300051 and fresh seed300050 evaluation streams. Match
    # that boundary, not a fixed/replayed evaluation every round.
    endpoints = [min(3, args.rounds)] + ([args.rounds] if args.rounds > 3 else [])
    commands = [command + ["--rounds", str(end)] for end in endpoints]
    out.mkdir(parents=True, exist_ok=True)
    if not args.final_only:
        (out / "command.json").write_text(json.dumps(commands, indent=1))
        (out / "run_metadata.json").write_text(json.dumps(dict(
            arm=args.arm, node=platform.node(), slurm_job=os.environ.get("SLURM_JOB_ID"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), baseline=str(args.baseline_config),
            seed=seed, eval_seed=b["eval_seed"], episode_steps=b["eval_steps"],
            round3_restart=True, sequence_microbatch=args.sequence_microbatch,
            render_chunk=args.render_chunk, baseline_args=b), indent=1))
        for endpoint, cmd in zip(endpoints, commands):
            dagger.log("command: " + shlex.join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
            path = out / "rounds.jsonl"
            completed = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            if not completed or max(row["round"] for row in completed) < endpoint - 1:
                break  # time/disk guard stopped before this phase endpoint
    rounds = [json.loads(line) for line in (out / "rounds.jsonl").read_text().splitlines()]
    best = max(rounds, key=lambda row: (row["eval"]["frac_3_laps"], -row["eval"].get("action_mae_vs_teacher", 1e9)))
    checkpoint = out / f"ckpt_r{best['round']:02d}.pkl"
    cfg, params, _ = st.load_student(checkpoint)
    dagger.log(f"best checkpoint: {checkpoint}")
    result = dict(arm=args.arm, checkpoint=str(checkpoint),
                  selection="max random-track three-lap rate on baseline evaluation stream; tie lower action MAE",
                  parameters=st.n_params(params), best_round=best, inference=dagger.inference_speed(cfg, params),
                  analysis_time=time.time())
    (out / "best.json").write_text(json.dumps(result, indent=1))
    dagger.log(f"inference: {result['inference']}")
    if args.arm == "S":
        dagger.check_held_out(300070, [data])
        result["failures"] = failure_breakdown(cfg, params, teacher, 300070, steps=b["eval_steps"])
        (out / "best.json").write_text(json.dumps(result, indent=1))
        dagger.log("failure breakdown: " + json.dumps({k: v for k, v in result["failures"].items()
                                                      if k not in ("episodes_detail", "metrics")}))
    dagger.log(f"finished {args.arm}: {out / 'best.json'}")


if __name__ == "__main__":
    main()
