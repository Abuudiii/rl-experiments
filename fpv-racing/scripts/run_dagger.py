"""DAgger for the camera-only student, renderer-agnostic.

    CUDA_VISIBLE_DEVICES=2 .venv-sim/bin/python scripts/run_dagger.py --out runs/student/<name> \
        --renderer synth --teacher runs/teacher/cruise_wide_noise2_s1/ckpt_latest.pkl --rounds 8

Round r: (1) collect ``--steps`` policy steps in ``--n-envs`` parallel slots, flown by the teacher
with probability beta_r per segment (else by the current student); teacher-flown steps carry action
noise and bursts; every state is labelled with the teacher's action and written to
``<out>/data``; (2) fine-tune (or ``--retrain``) the student on all rounds; (3) fly the student
alone (``--eval-episodes``, randomised layouts) and append the metrics to ``<out>/rounds.jsonl``.
With ``--eval-as-data`` the evaluation flights are stored too (they are student-flown, teacher
labelled states) and count as data of the next round. ``--resume`` continues an existing run.
List arguments (``--beta``, ``--steps``, ``--train-steps``) give one value per round; the last one
repeats.
"""

from __future__ import annotations

import argparse
import dataclasses
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


def per_round(text, cast):
    vals = [cast(v) for v in str(text).split(",")]
    return lambda r: vals[min(r, len(vals) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--renderer", default="synth")
    ap.add_argument("--renderer-args", default="{}", help="JSON kwargs for the renderer")
    ap.add_argument("--teacher", default="runs/teacher/cruise_wide_noise2_s1/ckpt_latest.pkl")
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--steps", default="600", help="collection steps per round (x n-envs frames)")
    ap.add_argument("--beta", default="1,0.5,0.3,0.15,0", help="probability that the teacher flies a segment")
    ap.add_argument("--train-steps", default="6000,3000")
    ap.add_argument("--ft-lr", type=float, default=None, help="learning rate of rounds >= 1 (default --lr)")
    ap.add_argument("--retrain", action="store_true", help="train from scratch every round")
    ap.add_argument("--recent-weight", type=float, default=1.0)
    ap.add_argument("--extra-frac", type=float, default=0.25, help="share of each batch drawn from --extra-data")
    ap.add_argument("--teacher-noise", type=float, default=0.2)
    ap.add_argument("--burst-p", type=float, default=0.01)
    ap.add_argument("--burst-mag", type=float, default=0.3)
    ap.add_argument("--wide-spawn-p", type=float, default=0.0)
    ap.add_argument("--eval-episodes", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=None, help="cap of the per-round evaluation (default 30 s)")
    ap.add_argument("--eval-as-data", action="store_true")
    ap.add_argument("--init", default=None, help="student checkpoint to start from")
    ap.add_argument("--extra-data", nargs="*", default=[], help="other data dirs to train on as well")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-device", default=None, help="Crazyflow device: cpu (faster for few slots) or gpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--first-round", type=int, default=0,
                    help="number of the first round (when <data> already holds earlier rounds, e.g. re-rendered ones)")
    ap.add_argument("--data-dir", default=None, help="where the shards go (default <out>/data)")
    ap.add_argument("--defer-render", action="store_true",
                    help="teacher-flown rounds: fly first, then render with the open-loop fast path")
    dagger.add_student_args(ap)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir, rounds_path = Path(args.data_dir) if args.data_dir else out / "data", out / "rounds.jsonl"
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    beta, steps, train_steps = per_round(args.beta, float), per_round(args.steps, int), per_round(args.train_steps, int)

    start = args.first_round
    if args.resume and rounds_path.exists():
        start = max(json.loads(line)["round"] for line in rounds_path.read_text().splitlines()) + 1
        cfg, params, _ = st.load_student(out / "ckpt_latest.pkl")
        log(f"resuming at round {start}")
    elif args.init:
        cfg, params, _ = st.load_student(args.init)
        log(f"initial weights from {args.init}")
    else:
        cfg = dagger.cfg_from_args(args)
        _, params = st.init_params(cfg, args.seed)
    has_student = args.init is not None or start > 0
    (out / "config.json").write_text(json.dumps(dict(student=dataclasses.asdict(cfg), args=vars(args)), indent=1))
    log(f"student: {st.n_params(params) / 1e6:.2f} M parameters, input {cfg.n_frames}x{cfg.height}x{cfg.width}")

    n = args.n_envs
    renderer = dagger.make_renderer(args.renderer, n, cfg.width, cfg.height, seed=args.seed, **json.loads(args.renderer_args))
    n = renderer.n
    fb = FlightBatch(n, args.teacher, seed=args.seed, randomize=True, wide_spawn_p=args.wide_spawn_p, device=args.sim_device)
    fb_eval = FlightBatch(n, args.teacher, seed=args.seed + 1000, randomize=True, device=args.sim_device)
    rng = np.random.default_rng(args.seed + 17 * start)
    data = dagger.Dataset()
    for d in args.extra_data:
        data.add_dir(d, extra=True)
    data.add_dir(data_dir)

    for r in range(start, args.rounds):
        row = dict(round=r, beta=beta(r) if has_student else 1.0)
        # 1. collect
        if steps(r) > 0:
            runner = st.StudentRunner(cfg, params, n) if has_student and row["beta"] < 1 else None
            writer = dagger.ShardWriter(data_dir, r)
            eps, info = dagger.fly(fb, renderer, runner, steps=steps(r), beta=row["beta"], reset_done=True, rng=rng,
                                   teacher_noise=args.teacher_noise, burst_p=args.burst_p, burst_mag=args.burst_mag,
                                   writer=writer, defer_render=args.defer_render and runner is None, log=log)  # fmt: skip
            crashes = sum(e["outcome"] > 0 for e in eps)
            row.update(collect_frames=writer.frames, collect_wall_s=info["wall_s"], collect_render_s=info["render_s"],
                       collect_crashes=crashes, collect_crashes_per_min=crashes / (writer.frames / 3000))  # fmt: skip
            log(f"round {r}: beta {row['beta']:.2f} collected {writer.frames} frames in {info['wall_s']:.0f} s "
                f"(render {info['render_s']:.0f} s), {crashes} crashes = {row['collect_crashes_per_min']:.2f} per flight minute")  # fmt: skip
        data.add_dir(data_dir)
        row.update(data_frames=len(data), data_by_round=data.sizes())
        # 2. train
        if args.retrain:
            _, params = st.init_params(cfg, args.seed + r)
        lr = args.ft_lr if (r > 0 and not args.retrain and args.ft_lr) else None
        t0 = time.time()
        params, hist = dagger.train(cfg, params, data, train_steps(r), seed=args.seed + r, lr=lr, log=log,
                                    recent_weight=args.recent_weight, extra_frac=args.extra_frac)  # fmt: skip
        has_student = True
        row.update(train_steps=train_steps(r), train_wall_s=time.time() - t0, train_mae=hist[-1]["mae"],
                   train_loss_act=hist[-1]["act"], train_loss_aux=hist[-1]["aux"])  # fmt: skip
        st.save_student(out / f"ckpt_r{r:02d}.pkl", params, cfg, dict(round=r))
        st.save_student(out / "ckpt_latest.pkl", params, cfg, dict(round=r))
        # 3. student-only evaluation
        runner = st.StudentRunner(cfg, params, n)
        eps, ev = [], dict(frames_rendered=0, render_s=0.0, wall_s=0.0)
        writer = dagger.ShardWriter(data_dir, r + 1) if args.eval_as_data else None
        while len(eps) < args.eval_episodes:
            _, e, info = dagger.evaluate(fb_eval, renderer, runner, steps=args.eval_steps, rng=rng, writer=writer, log=log)
            eps += e
            for k in ev:
                ev[k] += info[k]
        m = dagger.summarize(eps)
        row.update(eval=m, eval_frames=ev["frames_rendered"], eval_wall_s=ev["wall_s"], eval_render_s=ev["render_s"])
        dagger.append_jsonl(rounds_path, row)
        log(f"round {r}: EVAL 1 lap {m['frac_1_lap']:.3f}  3 laps {m['frac_3_laps']:.3f}  no crash "
            f"{m['frac_full_episode_no_crash']:.3f}  gates/ep {m['gates_per_episode']:.1f}  crashes {m['crash_rate_by_cause']} "
            f"lap {m['lap_time_s']}  mae {m.get('action_mae_vs_teacher', float('nan')):.4f}  "
            f"({ev['frames_rendered']} frames, {ev['wall_s']:.0f} s)")  # fmt: skip
    renderer.close()


if __name__ == "__main__":
    main()
