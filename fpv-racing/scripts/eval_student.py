"""Closed-loop evaluation of the camera-only student, with the teacher as the reference row.

    CUDA_VISIBLE_DEVICES=2 .venv-sim/bin/python scripts/eval_student.py runs/student/<run>/ckpt_latest.pkl \
        --teacher runs/teacher/<run>/ckpt_latest.pkl --renderer synth --scenarios all,random \
        --episodes 256 --nominal-episodes 64 --steps 1000 --seed 910001 \
        --held-out-from runs/student/<run>/data --out runs/student/<run>/eval.json

The student flies alone and deterministically from camera frames (plus its previous action); the
teacher is flown through the same harness (`fpv.dagger.fly`) on the same seeds, i.e. the same
layouts, physics and spawns. Per layout in ``--scenarios`` (names in `fpv.track.SCENARIOS`, ``all``
for every one of them, ``random`` for the teacher's training distribution: a new procedural track
per episode for a random-track teacher) and setting ("randomised": per-gate layout and physics
randomisation; "nominal": nominal gates and physics):

    frac_1_lap / frac_3_laps     episodes that passed >= 6 / >= 18 gates (gates only count before a
                                 crash, since a crash ends the episode)
    frac_full_episode_no_crash   episodes that reached the time limit
    gates_per_episode, crash_rate, crash_rate_by_cause, crashes_by_target_gate (gate index 0..5 the
                                 drone was heading for)
    crash_time_hist              share of episodes crashing within each time window after the start
    lap_time_s                   mean time between passing a gate and passing it again (6 gates on)
    speed_m_s                    mean speed over all flown steps
    action_mae_vs_teacher        mean |student action - teacher label| over the states the student
                                 visited (mean over the 4 channels; per channel also given)

Held-out evaluation: ``--seed`` has no default. ``--held-out-from <data dirs>`` refuses a seed that
produced training data there (`fpv.dagger.check_held_out`). ``--save-data`` stores the student's
flights for failure analysis only (the directory is marked NOT_TRAINING_DATA; round id = index of
the layout in the list, mapping in ``<dir>/layouts.json``); ``--save-training-data`` stores them as
training data and is only allowed for ``--scenarios random`` (named layouts are held out).
``--resume`` continues an interrupted evaluation: results and episodes are written after every
(layout, pilot, setting) cell; cells already present for the same checkpoint, seed and step cap are
kept.
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

from fpv import dagger, student as st, track  # noqa: E402
from fpv.closed_loop import FlightBatch  # noqa: E402


def run(fb, renderer, runner, episodes, steps, rng, log, writer=None):
    eps, tot = [], dict(frames_rendered=0, render_s=0.0, wall_s=0.0)
    while len(eps) < episodes:
        _, e, info = dagger.evaluate(fb, renderer, runner, steps=steps, rng=rng, log=log, writer=writer)
        eps += e
        for k in tot:
            tot[k] += info[k]
    return dict(dagger.summarize(eps), **tot), eps


def layouts(text: str) -> list[str]:
    out = []
    for x in text.split(","):
        out += list(track.SCENARIOS) if x == "all" else [x]
    for x in out:
        if x != "random" and x not in track.SCENARIOS:
            raise SystemExit(f"unknown layout {x!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--renderer", default="synth")
    ap.add_argument("--renderer-args", default="{}")
    ap.add_argument("--n-envs", type=int, default=256, help="parallel episodes (renderer slots)")
    ap.add_argument("--scenarios", default="random", help="comma list: names, 'all', 'random'")
    ap.add_argument("--episodes", type=int, default=256, help="randomised episodes per layout (rounded up to batches)")
    ap.add_argument("--nominal-episodes", type=int, default=64)
    ap.add_argument("--random-episodes", type=int, default=None, help="randomised episodes for 'random' (default --episodes)")
    ap.add_argument("--steps", type=int, default=None, help="cap episodes at this many steps (default: the time limit)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--held-out-from", nargs="*", default=[], help="training data dirs the seed must not come from")
    ap.add_argument("--sim-device", default=None, help="Crazyflow device: cpu (faster for few slots) or gpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-data", default=None, help="store the student's flights here (failure analysis only)")
    ap.add_argument("--save-training-data", default=None, help="store them as DAgger training data ('random' only)")
    ap.add_argument("--save-round", type=int, default=99, help="round id of --save-training-data")
    ap.add_argument("--skip-teacher", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep the cells (layout, pilot, setting) already in --out for the same checkpoint and seed")
    args = ap.parse_args()
    log = dagger.log
    names = layouts(args.scenarios)
    if args.save_training_data and names != ["random"]:
        raise SystemExit("named layouts are held out: --save-training-data only with --scenarios random")
    if args.save_training_data and args.out:
        raise SystemExit("an evaluation stored as training data is not a held-out result: drop --out or the data")
    dagger.check_held_out(args.seed, args.held_out_from)

    cfg, params, ck = st.load_student(args.checkpoint)
    renderer = dagger.make_renderer(args.renderer, args.n_envs, cfg.width, cfg.height, seed=args.seed,
                                    **json.loads(args.renderer_args))  # fmt: skip
    n = renderer.n
    runner = st.StudentRunner(cfg, params, n)
    res = dict(checkpoint=args.checkpoint, teacher=args.teacher, renderer=args.renderer,
               renderer_args=json.loads(args.renderer_args), seed=args.seed, held_out_from=args.held_out_from,
               steps=args.steps, student_config=ck["config"], parameters=st.n_params(params),
               highlight_target=dagger.highlight_on(renderer), scenarios={})  # fmt: skip
    if args.save_data:
        Path(args.save_data).mkdir(parents=True, exist_ok=True)
        (Path(args.save_data) / "layouts.json").write_text(json.dumps(dict(enumerate(names))))
    eps_out = {}
    out = Path(args.out) if args.out else None
    if args.resume and out and out.exists():
        old = json.loads(out.read_text())
        if ((old["checkpoint"], old["seed"], old.get("steps"), old.get("highlight_target"))
                == (args.checkpoint, args.seed, args.steps, res["highlight_target"])):
            res["scenarios"] = old["scenarios"]
            eps_out = json.loads(out.with_suffix(".episodes.json").read_text())
            log(f"resuming: {sum(len(v) for r in res['scenarios'].values() for v in r.values())} cells done")
    for li, name in enumerate(names):
        res["scenarios"].setdefault(name, dict(student={}, teacher_reference={}))
        count_r = args.random_episodes if (name == "random" and args.random_episodes is not None) else args.episodes
        for setting, randomize, count in (("randomised", True, count_r), ("nominal", False, args.nominal_episodes)):
            if count <= 0:
                continue
            for who in ("student", "teacher"):
                if who == "teacher" and args.skip_teacher:
                    continue
                if f"{name}/{who}_{setting}" in eps_out:
                    continue
                writer = None
                if who == "student" and args.save_training_data:
                    writer = dagger.ShardWriter(args.save_training_data, args.save_round, seed=args.seed, kind="eval_as_data",
                                                setting=setting, teacher=args.teacher, renderer=args.renderer,
                                                highlight_target=res["highlight_target"])  # fmt: skip
                elif who == "student" and args.save_data:
                    writer = dagger.ShardWriter(args.save_data, li, seed=args.seed, training=False, layout=name,
                                                setting=setting, checkpoint=args.checkpoint)  # fmt: skip
                fb = FlightBatch(n, args.teacher, seed=args.seed, randomize=randomize, device=args.sim_device,
                                 scenario=None if name == "random" else name)  # fmt: skip
                m, eps = run(fb, renderer, runner if who == "student" else None, count, args.steps,
                             np.random.default_rng(args.seed), log, writer)  # fmt: skip
                res["scenarios"][name]["student" if who == "student" else "teacher_reference"][setting] = m
                eps_out[f"{name}/{who}_{setting}"] = eps
                log(f"{name:15s} {who:8s} {setting:10s}: episodes {m['episodes']}  1 lap {m['frac_1_lap']:.3f}  "
                    f"3 laps {m['frac_3_laps']:.3f}  no crash {m['frac_full_episode_no_crash']:.3f}  gates/ep "
                    f"{m['gates_per_episode']:.1f}  crash {m['crash_rate_by_cause']}  by gate {m['crashes_by_target_gate']}  "
                    f"lap {m['lap_time_s']}  speed {m['speed_m_s']:.2f}  mae {m.get('action_mae_vs_teacher')}")  # fmt: skip
                if out:  # partial results survive an interruption (see --resume)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.with_suffix(".episodes.json").write_text(json.dumps(eps_out))
                    out.write_text(json.dumps(res, indent=1))
    res["inference"] = dagger.inference_speed(cfg, params)
    log(f"inference: {res['inference']}")
    renderer.close()
    if out:
        out.write_text(json.dumps(res, indent=1))
        out.with_suffix(".episodes.json").write_text(json.dumps(eps_out))
        log(f"wrote {out}")


if __name__ == "__main__":
    main()
