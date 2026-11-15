"""Extra DAgger collector feeding a running `scripts/run_dagger.py` (more student-flown data per round).

    CUDA_VISIBLE_DEVICES=1 .venv-sim/bin/python scripts/extra_collect.py --run runs/student/<run> \
        --data-dir <run data dir> --worker 1 --calls 20

Loop: load the run's ``ckpt_latest.pkl`` and its last finished round R (``rounds.jsonl``), fly
``--n-envs`` slots for ``--steps`` steps on the run's layouts (teacher's training distribution or
``--train-scenarios``), the student flying alone with probability 1 - ``--beta`` per segment
(teacher-flown segments get the run's action noise and bursts), label every state with the teacher
and write the call as ONE shard (atomic rename) ``r{R+1}_c{50 + 10 * (worker - 1) + k}_0000.npz``,
so the run picks it up at its next `Dataset.add_dir` and no partly written call is ever loaded.
With ``--stage-dir`` (a RAM disk, e.g. /dev/shm/...) the shard is written there and only a symlink
goes into the data directory, so the run reads it from memory; once the run has finished the round
that loaded it, the file is moved to ``--archive-dir`` (local disk) and the symlink repointed, which
keeps the RAM disk small.
Seeds: worker w flies one FlightBatch stream with seed ``--seed + 1000 * w`` (recorded in seeds.jsonl);
with hairpin tracks (``--hairpin-frac``, default the run's) call k uses ``--seed + 1000 * w + k`` and new tracks.
Settings (renderer and its args, off-course starts, teacher, noise) are read from the run's config.json;
``--renderer`` / ``--renderer-args`` override them (e.g. Unreal instances on GPUs the run does not use).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

from fpv import dagger, student as st, track  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--worker", type=int, required=True, help="1..5 (call ids and seeds)")
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-round", type=int, default=999, help="stop when the run has finished this round")
    ap.add_argument("--stage-dir", default=None, help="write shards here (RAM disk) and symlink them into --data-dir")
    ap.add_argument("--archive-dir", default=None, help="where staged shards go once the run has loaded them")
    ap.add_argument("--calls-per-round", type=int, default=None,
                    help="at most this many calls for one round of the run, then wait for its next round")
    ap.add_argument("--renderer", default=None, help="override the run's renderer (e.g. airsim on other GPUs)")
    ap.add_argument("--renderer-args", default=None, help="override the run's renderer args (JSON)")
    ap.add_argument("--hairpin-frac", type=float, default=None, help="default: the run's --hairpin-frac")
    ap.add_argument("--sim-device", default=None)
    ap.add_argument("--stop-at", type=float, default=None, help="unix time: no call is started that would end after it")
    ap.add_argument("--layouts", default=None,
                    help="comma list of named layouts replacing the run's (e.g. weight collection toward one layout)")
    args = ap.parse_args()
    log = dagger.log
    run = Path(args.run)
    ra = json.loads((run / "config.json").read_text())["args"]
    layouts = args.layouts.split(",") if args.layouts else list(track.SCENARIOS) if ra.get("train_scenarios") == "all" else [
        x for x in (ra.get("train_scenarios") or "").split(",") if x]  # fmt: skip
    renderer = fb = None
    seed = args.seed + 1000 * args.worker
    used = dagger.data_seeds([args.data_dir])
    while seed in used:  # a restarted worker: new stream (seeds.jsonl is written when a call starts)
        seed += 500
    kind = args.renderer or ra.get("renderer", "synth")
    rargs = json.loads(args.renderer_args or ra["renderer_args"])
    hairpin = ra.get("hairpin_frac", 0.0) if args.hairpin_frac is None else args.hairpin_frac
    data_dir = Path(args.data_dir)
    staged: list[tuple[int, Path]] = []  # (round, staged file)
    made: dict[int, int] = {}  # calls written per round of the run
    k, call_s = 0, 0.0
    log(f"worker {args.worker}: seed {seed}")
    while k < args.calls:
        if args.stop_at and time.time() + call_s > args.stop_at:
            log("stop: --stop-at reached")
            break
        t_call = time.time()
        rounds_path = run / "rounds.jsonl"
        if rounds_path.exists():  # the student of the run's last finished round flies
            r = max(json.loads(line)["round"] for line in rounds_path.read_text().splitlines())
            ckpt = run / "ckpt_latest.pkl"
        else:  # the run has not finished a round yet: its initial weights fly, data go to its first round
            r, ckpt = ra["first_round"] - 1, Path(ra["init"])
        if r >= args.max_round:
            break
        if args.calls_per_round is not None and made.get(r + 1, 0) >= args.calls_per_round:
            time.sleep(20)
            continue
        for rr, f in [x for x in staged if x[0] < r]:  # the run has surely loaded these (add_dir of round x+1): to disk
            dst = Path(args.archive_dir) / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst.with_suffix(".tmp"))
            dagger.drop_page_cache(dst.with_suffix(".tmp"), sync=True)
            dst.with_suffix(".tmp").rename(dst)
            link = data_dir / f.name
            tmp_link = link.with_name(link.name + ".tmp")
            tmp_link.symlink_to(dst.resolve())
            tmp_link.rename(link)
            f.unlink()
            staged.remove((rr, f))
        cfg, params, _ = st.load_student(ckpt)
        if renderer is None:
            renderer = dagger.make_renderer(kind, args.n_envs, cfg.width, cfg.height, seed=seed, **rargs)
        call_seed = seed + 2 * k if hairpin else seed  # hairpin: fresh tracks / stream every call (dagger.hairpin_seed = +1)
        if fb is None or hairpin:
            fb = dagger.make_flights(renderer.n, ra["teacher"], call_seed, layouts, randomize=True,
                                     wide_spawn_p=ra["wide_spawn_p"], hairpin_frac=hairpin, device=args.sim_device)  # fmt: skip
        call = 50 + 10 * (args.worker - 1) + k % 10
        while list(data_dir.glob(f"r{r + 1:03d}_c{call:02d}_*.npz")):
            call += 1000  # a worker that wraps within one round
        out_dir = Path(args.stage_dir) if args.stage_dir else data_dir
        writer = dagger.ShardWriter(out_dir, r + 1, seed=call_seed, call=call, kind="extra_collect", beta=args.beta,
                                    layout_seeds=[seed + 7919 * i for i in range(len(layouts))], layouts=layouts, student_round=r, keep_cache=True,
                                    teacher=ra["teacher"], renderer=kind, hairpin_frac=hairpin, checkpoint=str(ckpt),
                                    hairpin_seed=dagger.hairpin_seed(call_seed) if hairpin else None,
                                    highlight_target=dagger.highlight_on(renderer))  # fmt: skip
        eps, info = dagger.fly(fb, renderer, st.StudentRunner(cfg, params, renderer.n), steps=args.steps, beta=args.beta,
                               reset_done=True, rng=np.random.default_rng([seed, k]), teacher_noise=ra["teacher_noise"],
                               burst_p=ra["burst_p"], burst_mag=ra["burst_mag"], writer=writer, shard_steps=args.steps)  # fmt: skip
        if args.stage_dir:
            f = out_dir / f"{writer.prefix}_0000.npz"
            (data_dir / f.name).symlink_to(f.resolve())
            staged.append((r + 1, f))
            rec = json.loads((out_dir / "seeds.jsonl").read_text().splitlines()[-1])
            dagger.append_jsonl(data_dir / "seeds.jsonl", rec)
        crashes = sum(e["outcome"] > 0 for e in eps)
        log(f"call {k}: student of round {r} -> {writer.prefix}: {writer.frames} frames in {info['wall_s']:.0f} s "
            f"({writer.frames / info['wall_s']:.1f} frames/s; render {info['render_s']:.0f} s, policy {info['policy_s']:.0f} s, "
            f"sim {info['sim_s']:.0f} s), {crashes} crashes")  # fmt: skip
        made[r + 1] = made.get(r + 1, 0) + 1
        k += 1
        call_s = time.time() - t_call
    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()
