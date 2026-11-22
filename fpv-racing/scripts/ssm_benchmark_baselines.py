"""Benchmark existing B/G checkpoints, without training or re-flying either baseline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from fpv import student as st


def speed(cfg, params, batches=(1, 64), reps=200):
    out = {}
    for batch in batches:
        runner = st.StudentRunner(cfg, params, batch)
        frame = jnp.zeros((batch, cfg.height, cfg.width, 3), jnp.uint8)
        pa = jnp.zeros((batch, 4))
        new = jnp.zeros(batch, bool)
        hist, action, aux = runner._act(params, runner._hist, frame, new, pa)
        action.block_until_ready()
        start = time.perf_counter()
        for _ in range(reps):
            hist, action, aux = runner._act(params, hist, frame, new, pa)
            action.block_until_ready()
        dt = (time.perf_counter() - start) / reps
        out[f"batch_{batch}"] = dict(ms_per_call=dt * 1000, calls_per_s=1 / dt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/student/ssm/baselines/benchmark.json"))
    args = ap.parse_args()
    result = dict(method="200 synchronized GPU-resident runner steps; includes frame-history shift, action and aux heads; no host frame copies", baselines={})
    for arm in ("B", "G"):
        directory = Path(f"runs/student/tracks_synth_{arm}")
        rows = [json.loads(line) for line in (directory / "rounds.jsonl").read_text().splitlines()]
        best = max(rows, key=lambda row: (row["eval"]["frac_3_laps"], -row["eval"].get("action_mae_vs_teacher", 1e9)))
        checkpoint = directory / f"ckpt_r{best['round']:02d}.pkl"
        cfg, params, _ = st.load_student(checkpoint)
        result["baselines"][arm] = dict(checkpoint=str(checkpoint), round=best["round"],
                                        parameters=st.n_params(params), inference=speed(cfg, params))
        print(arm, result["baselines"][arm], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=1))
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
