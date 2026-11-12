"""Closed-loop evaluation of the camera-only student, with the teacher as the reference row.

    CUDA_VISIBLE_DEVICES=2 .venv-sim/bin/python scripts/eval_student.py runs/student/<run>/ckpt_latest.pkl \
        --renderer synth --episodes 256 --nominal-episodes 64 --out runs/student/<run>/eval.json

The student flies alone and deterministically from camera frames (plus its previous action); the
teacher is flown through the same harness (`fpv.dagger.fly`) on the same seeds, i.e. the same
layouts, physics and spawns. Episodes are time-limited (30 s). Metrics per setting ("randomised":
per-episode gate layout and physics; "nominal": the nominal circuit and physics):

    frac_1_lap / frac_3_laps     episodes that passed >= 6 / >= 18 gates (gates only count before a
                                 crash, since a crash ends the episode)
    frac_full_episode_no_crash   episodes that reached the time limit
    gates_per_episode, crash_rate, crash_rate_by_cause, crashes_by_target_gate
    lap_time_s                   mean time between passing a gate and passing it again (6 gates on)
    speed_m_s                    mean speed over all flown steps
    action_mae_vs_teacher        mean |student action - teacher label| over the states the student
                                 visited (mean over the 4 channels; per channel also given)
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

import numpy as np  # noqa: E402

from fpv import dagger, student as st  # noqa: E402
from fpv.closed_loop import FlightBatch  # noqa: E402


def run(fb, renderer, runner, episodes, steps, rng, log, writer=None):
    eps, tot = [], dict(frames_rendered=0, render_s=0.0, wall_s=0.0)
    while len(eps) < episodes:
        _, e, info = dagger.evaluate(fb, renderer, runner, steps=steps, rng=rng, log=log, writer=writer)
        eps += e
        for k in tot:
            tot[k] += info[k]
    return dict(dagger.summarize(eps), **tot), eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--teacher", default="runs/teacher/cruise_wide_noise2_s1/ckpt_latest.pkl")
    ap.add_argument("--renderer", default="synth")
    ap.add_argument("--renderer-args", default="{}")
    ap.add_argument("--n-envs", type=int, default=256, help="parallel episodes (renderer slots)")
    ap.add_argument("--episodes", type=int, default=256, help="randomised episodes (rounded up to batches)")
    ap.add_argument("--nominal-episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=None, help="cap episodes at this many steps (default: 30 s)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sim-device", default=None, help="Crazyflow device: cpu (faster for few slots) or gpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-data", default=None, help="also store the student's flights as DAgger shards here")
    ap.add_argument("--save-round", type=int, default=99)
    ap.add_argument("--skip-teacher", action="store_true")
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731

    cfg, params, ck = st.load_student(args.checkpoint)
    renderer = dagger.make_renderer(args.renderer, args.n_envs, cfg.width, cfg.height, seed=args.seed,
                                    **json.loads(args.renderer_args))  # fmt: skip
    n = renderer.n
    runner = st.StudentRunner(cfg, params, n)
    writer = dagger.ShardWriter(args.save_data, args.save_round) if args.save_data else None
    res = dict(checkpoint=args.checkpoint, teacher=args.teacher, renderer=args.renderer, seed=args.seed,
               student_config=ck["config"], parameters=st.n_params(params), student={}, teacher_reference={})  # fmt: skip
    eps_out = {}
    for name, randomize, count in (("randomised", True, args.episodes), ("nominal", False, args.nominal_episodes)):
        if count <= 0:
            continue
        for who in ("student", "teacher"):
            if who == "teacher" and args.skip_teacher:
                continue
            fb = FlightBatch(n, args.teacher, seed=args.seed, randomize=randomize, device=args.sim_device)
            m, eps = run(fb, renderer, runner if who == "student" else None, count, args.steps,
                         np.random.default_rng(args.seed), log, writer if who == "student" else None)  # fmt: skip
            res["student" if who == "student" else "teacher_reference"][name] = m
            eps_out[f"{who}_{name}"] = eps
            log(f"{who:8s} {name:10s}: episodes {m['episodes']}  1 lap {m['frac_1_lap']:.3f}  3 laps {m['frac_3_laps']:.3f}  "
                f"no crash {m['frac_full_episode_no_crash']:.3f}  gates/ep {m['gates_per_episode']:.1f}  "
                f"crash {m['crash_rate_by_cause']}  by gate {m['crashes_by_target_gate']}  lap {m['lap_time_s']}  "
                f"speed {m['speed_m_s']:.2f}  mae {m.get('action_mae_vs_teacher')}")  # fmt: skip
    res["inference"] = dagger.inference_speed(cfg, params)
    log(f"inference: {res['inference']}")
    renderer.close()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        Path(args.out).with_suffix(".episodes.json").write_text(json.dumps(eps_out))
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
