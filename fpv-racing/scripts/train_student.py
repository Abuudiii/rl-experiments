"""Train (or fine-tune) the student on existing DAgger shards, without flying.

    CUDA_VISIBLE_DEVICES=2 .venv-sim/bin/python scripts/train_student.py --data runs/student/<run>/data \
        --out runs/student/<name>/ckpt_latest.pkl --steps 20000 [--init <ckpt>] [--rounds 0,1,2]

Network / input options (``--width --height --frames --no-prev-action --aux-weight --channels``) apply
when training from scratch; with ``--init`` the checkpoint's configuration is used. The frames in
the shards must have the student's resolution.
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

from fpv import dagger, student as st  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--rounds", default=None, help="comma list of rounds to use (default all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--recent-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    dagger.add_student_args(ap)
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731

    if args.init:
        cfg, params, _ = st.load_student(args.init)
    else:
        cfg = dagger.cfg_from_args(args)
        _, params = st.init_params(cfg, args.seed)
    rounds = None if args.rounds is None else {int(r) for r in args.rounds.split(",")}
    data = dagger.Dataset()
    for d in args.data:
        data.add_dir(d, rounds)
    h, w = data.blocks[0].frames.shape[1:3]
    assert (h, w) == (cfg.height, cfg.width), f"shards are {w}x{h}, student wants {cfg.width}x{cfg.height}"
    log(f"{len(data)} frames by round {data.sizes()}; student {st.n_params(params) / 1e6:.2f} M parameters")
    t0 = time.time()
    params, hist = dagger.train(cfg, params, data, args.steps, seed=args.seed, lr=args.lr if args.init else None,
                                log=log, recent_weight=args.recent_weight)  # fmt: skip
    st.save_student(args.out, params, cfg, dict(train=hist, data=[str(d) for d in args.data]))
    Path(args.out).with_suffix(".train.json").write_text(
        json.dumps(dict(config=dataclasses.asdict(cfg), frames=len(data), by_round=data.sizes(), history=hist,
                        wall_s=time.time() - t0), indent=1))  # fmt: skip
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
