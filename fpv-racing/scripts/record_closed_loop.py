"""Closed-loop flights recorded with the frames the pilot saw, for checks of the rendering path.

    .venv-sim/bin/python scripts/record_closed_loop.py --teacher runs/teacher/<run>/ckpt_latest.pkl \
        [--student <ckpt>] --renderer airsim --renderer-args '<json>' --scenarios all --seed 960001 \
        --steps 1500 --out-dir /tmp/<dir>

The teacher (default) or the student (``--student``) flies one episode per renderer slot through
`fpv.dagger.fly` with rendering ON at every step, i.e. the closed-loop ``render(..., target=)`` path
the student uses (not the open-loop `render_trajectories`). Slots are split evenly between the
layouts in ``--scenarios`` (`fpv.track.SCENARIOS` names, ``all``, or ``random``). Every step is
stored in the DAgger shard format (pose, gates, target index, frame), one shard per batch, in a
directory marked NOT_TRAINING_DATA. Check them with ``scripts/check_highlight_frames.py --near-change``.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", default=None, help="student checkpoint (default: the teacher flies)")
    ap.add_argument("--renderer", default="airsim")
    ap.add_argument("--renderer-args", default="{}")
    ap.add_argument("--n-envs", type=int, default=12, help="slots (synthetic renderer only)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sim-device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    names = list(track.SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")
    layouts = None if names == ["random"] else names
    cfg = params = None
    if args.student:
        cfg, params, _ = st.load_student(args.student)
    w, h = (cfg.width, cfg.height) if cfg else (args.width, args.height)
    renderer = dagger.make_renderer(args.renderer, args.n_envs, w, h, seed=args.seed, **json.loads(args.renderer_args))
    n = renderer.n
    fb = dagger.make_flights(n, args.teacher, args.seed, layouts, randomize=True, device=args.sim_device)
    runner = st.StudentRunner(cfg, params, n) if cfg else None
    for b in range(args.batches):
        writer = dagger.ShardWriter(args.out_dir, 0, seed=args.seed, training=False, kind="record_closed_loop",
                                    pilot=args.student or "teacher", layouts=names, renderer=args.renderer,
                                    renderer_args=json.loads(args.renderer_args),
                                    highlight_target=dagger.highlight_on(renderer))  # fmt: skip
        eps, info = dagger.fly(fb, renderer, runner, steps=args.steps, beta=0.0, reset_done=False, render=True,
                               rng=np.random.default_rng([args.seed, b]), writer=writer, shard_steps=args.steps, log=log)  # fmt: skip
        m = dagger.summarize(eps)
        log(f"batch {b}: {writer.frames} frames in {info['wall_s']:.0f} s; 3 laps {m['frac_3_laps']:.3f} "
            f"gates/ep {m['gates_per_episode']:.1f} crashes {m['crash_rate_by_cause']}")  # fmt: skip
    renderer.close()


if __name__ == "__main__":
    main()
