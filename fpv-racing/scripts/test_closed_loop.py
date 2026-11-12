"""Tests of `fpv.closed_loop.FlightBatch`.

    CUDA_VISIBLE_DEVICES=0 .venv-sim/bin/python scripts/test_closed_loop.py runs/teacher/<run>/ckpt_latest.pkl
    JAX_PLATFORMS=cpu      .venv-sim/bin/python scripts/test_closed_loop.py <ckpt> --latency-only

(i)   n=256, feed back ``teacher_action`` for one full episode: must reproduce the teacher's
      evaluation (gates per episode, crashes).
(ii)  ``teacher_action`` + Gaussian noise (std 0.3), resetting finished environments: crash rate,
      labels finite; plus API checks (frozen after done, partial reset, wide spawn).
(iii) ``step`` latency for n = 16, 64, 256 on the current device.
Prints one JSON line starting with ``RESULT``.
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

import jax  # noqa: E402
import numpy as np  # noqa: E402

from fpv.closed_loop import FlightBatch  # noqa: E402

SHAPES = dict(pos=(3,), quat=(4,), vel=(3,), ang_vel=(3,), prev_action=(4,), target=(), gates_pos=(6, 3),
              gates_yaw=(6,), teacher_action=(4,), teacher_obs=(43,), done=(), outcome=(), passed=(),
              gates_passed_total=(), t=())  # fmt: skip


def check_state(s, n):
    assert set(s) == set(SHAPES), set(s) ^ set(SHAPES)
    for k, shape in SHAPES.items():
        assert isinstance(s[k], np.ndarray) and s[k].shape == (n, *shape), (k, s[k].shape)
    assert s["done"].dtype == bool and s["passed"].dtype == bool
    return bool(np.isfinite(s["teacher_action"]).all() and np.isfinite(s["teacher_obs"]).all())


def test_teacher(ckpt, n=256):
    fb = FlightBatch(n, ckpt, seed=0)
    s = fb.reset()
    finite = check_state(s, n)
    assert not s["done"].any() and (s["t"] == 0).all()
    steps = fb.cfg.max_steps
    speed, done_at = [], np.full(n, -1)
    for i in range(steps):
        s = fb.step(s["teacher_action"])
        finite &= check_state(s, n)
        speed.append(np.linalg.norm(s["vel"], axis=-1)[~s["done"]])
    assert s["done"].all(), "every episode must end at the time limit"
    return dict(
        n=n, steps=steps, crashes=int((s["outcome"] > 0).sum()), time_limit=int((s["outcome"] == 0).sum()),
        gates_per_episode=float(s["gates_passed_total"].mean()),
        frac_3_laps=float((s["gates_passed_total"] >= 18).mean()),
        speed_mean=float(np.concatenate(speed).mean()), labels_finite=finite,
    )  # fmt: skip


def test_noise(ckpt, n=256, steps=3000, std=0.3, wide=0.0):
    fb = FlightBatch(n, ckpt, seed=1, wide_spawn_p=wide)
    rng = np.random.default_rng(0)
    s = fb.reset()
    finite, episodes, crashes, gates, flown = True, 0, np.zeros(4, int), 0, 0
    for _ in range(steps):
        a = s["teacher_action"] + rng.normal(0, std, (n, 4)).astype(np.float32)
        s = fb.step(a)
        finite &= check_state(s, n)
        flown += n
        if s["done"].any():
            d = s["done"]
            episodes += int(d.sum())
            crashes += np.bincount(s["outcome"][d], minlength=4)
            gates += int(s["gates_passed_total"][d].sum())
            s = fb.reset(d)
            assert not s["done"].any() and (s["t"][d] == 0).all() and (s["gates_passed_total"][d] == 0).all()
    gates += int(s["gates_passed_total"].sum())
    hours = flown / 50 / 3600
    return dict(
        n=n, steps=steps, noise_std=std, wide_spawn_p=wide, finished_episodes=episodes,
        ended_by=dict(zip(("time_limit", "gate_crash", "ground", "out_of_bounds"), crashes.tolist())),
        crashes_per_flight_minute=float(crashes[1:].sum() / (hours * 60)),
        crashes_per_gate=float(crashes[1:].sum() / max(gates, 1)), gates=gates, labels_finite=finite,
    )  # fmt: skip


def test_api(ckpt, n=16):
    """Frozen after done, partial reset leaves the others untouched, resets redraw the layout."""
    fb = FlightBatch(n, ckpt, seed=2)
    s = fb.reset()
    down = np.tile(np.float32([-1, 0, 0, 0]), (n, 1))  # zero thrust: everything hits the ground
    for _ in range(200):
        s = fb.step(down)
    assert s["done"].all() and (s["outcome"] == 2).all(), s["outcome"]
    frozen = fb.step(np.zeros((n, 4), np.float32))
    same = all(np.array_equal(frozen[k], s[k]) for k in SHAPES)
    mask = np.arange(n) < n // 2
    r = fb.reset(mask)
    untouched = all(np.array_equal(r[k][~mask], s[k][~mask]) for k in SHAPES)
    fresh = bool((~r["done"][mask]).all() and (r["t"][mask] == 0).all() and (r["outcome"][mask] == 0).all())
    new_layout = bool((np.abs(r["gates_pos"][mask] - s["gates_pos"][mask]).max(axis=(1, 2)) > 0).all())
    nominal = FlightBatch(n, ckpt, seed=3, randomize=False).reset()
    from fpv import track

    nominal_ok = bool(np.allclose(nominal["gates_pos"], track.circuit_arrays()[0]))
    wide = FlightBatch(256, ckpt, seed=4, wide_spawn_p=1.0).reset()
    return dict(frozen_after_done=same, partial_reset_untouched=untouched, reset_fresh=fresh,
                reset_new_layout=new_layout, nominal_layout=nominal_ok,
                wide_spawn_speed_max=float(np.linalg.norm(wide["vel"], axis=-1).max()),
                wide_spawn_labels_finite=bool(np.isfinite(wide["teacher_action"]).all()))  # fmt: skip


def test_latency(ckpt, sizes=(16, 64, 256), steps=300):
    out = {}
    for n in sizes:
        fb = FlightBatch(n, ckpt, seed=0)
        s = fb.reset()
        for _ in range(20):  # compile + warm up
            s = fb.step(s["teacher_action"])
        times = []
        for _ in range(steps):
            t0 = time.perf_counter()
            s = fb.step(s["teacher_action"])
            times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fb.reset(np.arange(n) % 4 == 0)
        out[n] = dict(step_ms_mean=1e3 * float(np.mean(times)), step_ms_median=1e3 * float(np.median(times)),
                      step_ms_p99=1e3 * float(np.percentile(times, 99)), reset_ms=1e3 * (time.perf_counter() - t0))  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--latency-only", action="store_true")
    args = ap.parse_args()
    res = {"device": str(jax.devices()[0]), "checkpoint": args.checkpoint}
    if not args.latency_only:
        res["teacher_feedback"] = test_teacher(args.checkpoint)
        print("(i)", res["teacher_feedback"], flush=True)
        for std in (0.1, 0.2, 0.3):
            res[f"noise_{std}"] = test_noise(args.checkpoint, std=std)
            print(f"(ii) noise std {std}", res[f"noise_{std}"], flush=True)
        res["api"] = test_api(args.checkpoint)
        print("api", res["api"], flush=True)
    res["latency"] = test_latency(args.checkpoint)
    print("(iii)", res["latency"], flush=True)
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
