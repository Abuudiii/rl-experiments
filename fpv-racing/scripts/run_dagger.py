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
List arguments (``--beta``, ``--steps``, ``--train-steps``) give one value per round, counted from
``--first-round``; the last one repeats.

Layouts: data are flown on the teacher's training distribution (a new procedural track per episode
for a random-track teacher), or with ``--train-scenarios`` on named layouts (`fpv.track.SCENARIOS`,
randomised; the slots are split evenly between them). ``--eval-scenarios`` adds, per round,
student-only flights on named layouts (randomised); those are never stored.

Seeds (all recorded in ``<data>/seeds.jsonl``): collection of round r uses FlightBatch seed
``--seed + 100 * (first round of this invocation)`` (a resumed run continues with new layouts), the
per-round evaluation ``--eval-seed`` (default ``--seed + 50``; with ``--eval-as-data`` plus
``100 * (rounds done before this invocation)``); with ``--eval-as-data`` that seed has
produced training data, so a reported evaluation must use another one (``eval_student.py`` checks).
``--relabel`` relabels all loaded data with ``--teacher`` (after a teacher swap).
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

from fpv import dagger, student as st, track  # noqa: E402
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
    ap.add_argument("--hairpin-frac", type=float, default=0.0,
                    help="share of collection slots flying fixed 'hairpin then S' tracks (fpv.dagger.hairpin_s_tracks)")
    ap.add_argument("--eval-episodes", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=None, help="cap of the per-round evaluation (default 30 s)")
    ap.add_argument("--eval-as-data", action="store_true")
    ap.add_argument("--init", default=None, help="student checkpoint to start from")
    ap.add_argument("--extra-data", nargs="*", default=[], help="other data dirs to train on as well")
    ap.add_argument("--data-also", nargs="*", default=[], help="more data dirs counted as this run's own data (re-scanned every round)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-device", default=None, help="Crazyflow device: cpu (faster for few slots) or gpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--first-round", type=int, default=0,
                    help="number of the first round (when <data> already holds earlier rounds, e.g. re-rendered ones)")
    ap.add_argument("--data-dir", default=None, help="where the shards go (default <out>/data)")
    ap.add_argument("--defer-render", action="store_true",
                    help="teacher-flown rounds: fly first, then render with the open-loop fast path")
    ap.add_argument("--eval-seed", type=int, default=None, help="FlightBatch seed of the per-round evaluation")
    ap.add_argument("--eval-scenarios", default="", help="comma list of fpv.track.SCENARIOS names (or 'all')")
    ap.add_argument("--scenario-episodes", type=int, default=64, help="per scenario and round")
    ap.add_argument("--relabel", action="store_true", help="recompute all labels with --teacher")
    ap.add_argument("--train-scenarios", default="",
                    help="fly these named layouts (comma list or 'all'; n-envs split evenly) instead of the "
                         "teacher's training distribution, for collection and the per-round evaluation")
    ap.add_argument("--stop-at", type=float, default=None,
                    help="unix time: no new round is started that would (by the last round's duration) end after it")
    dagger.add_student_args(ap)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir, rounds_path = Path(args.data_dir) if args.data_dir else out / "data", out / "rounds.jsonl"
    log = dagger.log
    b_, s_, t_ = per_round(args.beta, float), per_round(args.steps, int), per_round(args.train_steps, int)
    beta, steps, train_steps = (lambda r, f=f: f(r - args.first_round) for f in (b_, s_, t_))

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
    log(f"student: {st.n_params(params) / 1e6:.2f} M parameters, input {cfg.n_frames}x{cfg.height}x{cfg.width}")

    n = args.n_envs
    renderer = dagger.make_renderer(args.renderer, n, cfg.width, cfg.height, seed=args.seed, **json.loads(args.renderer_args))
    n = renderer.n
    (out / "config.json").write_text(json.dumps(dict(student=dataclasses.asdict(cfg), args=vars(args),
                                                     highlight_target=dagger.highlight_on(renderer)), indent=1))  # fmt: skip
    collect_seed = args.seed + 100 * start
    eval_seed = args.seed + 50 if args.eval_seed is None else args.eval_seed
    if args.eval_as_data:  # a resumed run stores new evaluation layouts, not a replay of the first ones
        eval_seed += 100 * (start - args.first_round)
    # A run that died inside a round resumes at the same round: its collect / eval seeds already made data
    # (seeds.jsonl is written when a call starts). Move on by 100 until both are fresh.
    used = dagger.data_seeds([*args.data_also, data_dir])
    while collect_seed in used:
        collect_seed += 100
    while args.eval_as_data and (eval_seed in used or eval_seed == collect_seed):
        eval_seed += 100
    assert eval_seed != collect_seed
    log(f"seeds: collect {collect_seed}, round evaluation {eval_seed}")
    train_layouts = (list(track.SCENARIOS) if args.train_scenarios == "all"
                     else [x for x in args.train_scenarios.split(",") if x])  # fmt: skip
    fb = dagger.make_flights(n, args.teacher, collect_seed, train_layouts, randomize=True, wide_spawn_p=args.wide_spawn_p,
                             hairpin_frac=args.hairpin_frac, device=args.sim_device)  # fmt: skip
    fb_eval = dagger.make_flights(n, args.teacher, eval_seed, train_layouts, randomize=True, device=args.sim_device)
    scen = list(track.SCENARIOS) if args.eval_scenarios == "all" else [x for x in args.eval_scenarios.split(",") if x]
    if scen:  # the synthetic renderer gets its own slots for these; Unreal shares its instances
        r_s = renderer if args.renderer != "synth" else dagger.make_renderer(
            "synth", min(args.scenario_episodes, n), cfg.width, cfg.height, seed=args.seed + 7, **json.loads(args.renderer_args))
        fb_s = {name: FlightBatch(r_s.n, args.teacher, seed=eval_seed, randomize=True, device=args.sim_device,
                                  scenario=name) for name in scen}  # fmt: skip
    rng = np.random.default_rng(args.seed + 17 * start)
    data = dagger.Dataset(relabel=args.teacher if args.relabel else None)
    for d in args.extra_data:
        data.add_dir(d, extra=True)
    for d in args.data_also:
        data.add_dir(d)
    data.add_dir(data_dir)

    round_s = 0.0
    for r in range(start, args.rounds):
        if args.stop_at and time.time() + round_s > args.stop_at:
            log(f"stop: round {r} would end after --stop-at ({time.strftime('%H:%M', time.localtime(args.stop_at))}); "
                f"last checkpoint ckpt_r{r - 1:02d}")  # fmt: skip
            break
        t_round = time.time()
        row = dict(round=r, beta=beta(r) if has_student else 1.0)
        # 1. collect
        if steps(r) > 0:
            runner = st.StudentRunner(cfg, params, n) if has_student and row["beta"] < 1 else None
            call_seed = collect_seed
            if args.hairpin_frac and r > start:  # new hairpin tracks and FlightBatch streams every round
                call_seed = collect_seed + 2 * (r - start)  # steps of 2: dagger.hairpin_seed(s) = s + 1
                clash = {call_seed, dagger.hairpin_seed(call_seed)} & (dagger.data_seeds([*args.data_also, data_dir]) | {eval_seed})
                assert not clash, f"round {r}: hairpin collection seeds {clash} already used; resume with another --seed"
                fb = dagger.make_flights(n, args.teacher, call_seed, train_layouts, randomize=True,
                                         wide_spawn_p=args.wide_spawn_p, hairpin_frac=args.hairpin_frac, device=args.sim_device)  # fmt: skip
            writer = dagger.ShardWriter(data_dir, r, seed=call_seed, kind="collect", layouts=train_layouts or "teacher",
                                        layout_seeds=[collect_seed + 7919 * i for i in range(len(train_layouts))], beta=row["beta"],
                                        teacher=args.teacher, renderer=args.renderer, hairpin_frac=args.hairpin_frac,
                                        hairpin_seed=dagger.hairpin_seed(call_seed) if args.hairpin_frac else None,
                                        highlight_target=dagger.highlight_on(renderer))  # fmt: skip
            eps, info = dagger.fly(fb, renderer, runner, steps=steps(r), beta=row["beta"], reset_done=True, rng=rng,
                                   teacher_noise=args.teacher_noise, burst_p=args.burst_p, burst_mag=args.burst_mag,
                                   writer=writer, defer_render=args.defer_render and runner is None, log=log)  # fmt: skip
            crashes = sum(e["outcome"] > 0 for e in eps)
            row.update(collect_frames=writer.frames, collect_wall_s=info["wall_s"], collect_render_s=info["render_s"],
                       collect_crashes=crashes, collect_crashes_per_min=crashes / (writer.frames / 3000))  # fmt: skip
            log(f"round {r}: beta {row['beta']:.2f} collected {writer.frames} frames in {info['wall_s']:.0f} s "
                f"(render {info['render_s']:.0f} s), {crashes} crashes = {row['collect_crashes_per_min']:.2f} per flight minute")  # fmt: skip
        for d in [*args.data_also, data_dir]:  # also calls that extra collectors (scripts/extra_collect.py) added
            data.add_dir(d)
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
        writer = dagger.ShardWriter(data_dir, r + 1, seed=eval_seed, kind="round_eval", teacher=args.teacher,
                                    layout_seeds=[eval_seed + 7919 * i for i in range(len(train_layouts))],
                                    renderer=args.renderer, highlight_target=dagger.highlight_on(renderer)
                                    ) if args.eval_as_data else None  # fmt: skip
        while len(eps) < args.eval_episodes:
            _, e, info = dagger.evaluate(fb_eval, renderer, runner, steps=args.eval_steps, rng=rng, writer=writer, log=log)
            eps += e
            for k in ev:
                ev[k] += info[k]
        m = dagger.summarize(eps)
        row.update(eval=m, eval_frames=ev["frames_rendered"], eval_wall_s=ev["wall_s"], eval_render_s=ev["render_s"],
                   collect_seed=collect_seed, eval_seed=eval_seed)  # fmt: skip
        if scen:
            t0, row["eval_scenarios"] = time.time(), {}
            runner_s = runner if r_s is renderer else st.StudentRunner(cfg, params, r_s.n)
            for name, fbs in fb_s.items():  # held out: never written
                se = []
                while len(se) < args.scenario_episodes:
                    se += dagger.evaluate(fbs, r_s, runner_s, steps=args.eval_steps, rng=rng)[1]
                row["eval_scenarios"][name] = dagger.summarize(se[: args.scenario_episodes])
            row["eval_scenarios_wall_s"] = time.time() - t0
            log(f"round {r}: scenarios 3 laps " + "  ".join(
                f"{k} {v['frac_3_laps']:.3f}" for k, v in row["eval_scenarios"].items()))  # fmt: skip
        round_s = time.time() - t_round
        row["round_wall_s"] = round_s
        dagger.append_jsonl(rounds_path, row)
        log(f"round {r}: EVAL 1 lap {m['frac_1_lap']:.3f}  3 laps {m['frac_3_laps']:.3f}  no crash "
            f"{m['frac_full_episode_no_crash']:.3f}  gates/ep {m['gates_per_episode']:.1f}  crashes {m['crash_rate_by_cause']} "
            f"lap {m['lap_time_s']}  mae {m.get('action_mae_vs_teacher', float('nan')):.4f}  "
            f"({ev['frames_rendered']} frames, {ev['wall_s']:.0f} s)")  # fmt: skip
    renderer.close()
    if scen and r_s is not renderer:
        r_s.close()


if __name__ == "__main__":
    main()
