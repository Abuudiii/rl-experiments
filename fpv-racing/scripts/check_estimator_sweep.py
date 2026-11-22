"""Estimator parameter sweep on the component check (teacher flying from its true observation, open loop,
same flights as `scripts/check_estimator.py`: random tracks, standard spawns, seed 4242), plus a breakdown of
the next-gate pose errors that start right after a true gate pass (the dominant phase).

    CUDA_VISIBLE_DEVICES=1 .venv-sim/bin/python scripts/check_estimator_sweep.py --conditions x1,drop \
        --variants "base;pass_window=0.8;geo_steps=5" --out runs/percep/policy/checks/sweep.json

Each variant is ';'-separated, fields within a variant ','-separated ``EstConfig`` overrides ("base" = none);
fields prefixed ``det.`` override the detection-noise model (`DetNoiseConfig`, e.g. det.p_false=0.01).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from check_estimator import TEACHER, make_rollout, runs, summarise  # noqa: E402
from fpv import ppo  # noqa: E402
from fpv.env_est import EstEnv, condition  # noqa: E402


def parse_variant(v: str) -> dict:
    if v.strip() in ("", "base"):
        return {}
    out = {}
    for kv in v.split(","):
        k, val = kv.split("=", 1)
        out[k.strip()] = ast.literal_eval(val.strip())
    return out


def after_pass_breakdown(rec, dt):
    """Pose-wrong runs (>= 3 steps) of the next gate that start <= 0.2 s after a true pass: where the shown
    next gate came from at the run start (seen now / remembered filter track / list placement), its true
    distance, and whether the estimator's index was right."""
    alive, lk = rec["alive"], rec["alive"] & rec["locked"]
    tol = np.maximum(0.5, 0.15 * rec["ng_dist"])
    pw = (rec["ng_err"] > tol) & lk
    tt = np.arange(alive.shape[0])[:, None]
    last = np.maximum.accumulate(np.where(rec["passed"], tt, -10**6), axis=0)
    sp = tt - last
    pr = [r for r in runs(pw) if r[2] >= 3 and sp[r[1], r[0]] <= 10]
    if not pr:
        return None
    e, s = np.array([r[0] for r in pr]), np.array([r[1] for r in pr])
    n = np.array([r[2] for r in pr])
    src = rec["ng_src"][s, e]
    idx_right = rec["nxt"][s, e] == rec["target"][s, e]
    out = {"runs": len(pr), "len_s_p50": float(np.median(n) * dt), "index_right_frac": float(idx_right.mean()),
           "src_at_start": {nm: float((src == k).mean()) for k, nm in enumerate(["seen_now", "remembered", "placement"])},
           "true_dist_m_p50": float(np.median(rec["ng_dist"][s, e])), "err_m_p50": float(np.median(rec["ng_err"][s, e])),
           "steps_after_pass_p50": float(np.median(sp[s, e]))}  # fmt: skip
    # how the run ended: the source when it became right again
    end = np.minimum(s + n, alive.shape[0] - 1)
    src_end = rec["ng_src"][end, e]
    out["src_at_end"] = {nm: float((src_end == k).mean()) for k, nm in enumerate(["seen_now", "remembered", "placement"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="x1")
    ap.add_argument("--variants", default="base")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    net, params, norm, env_cfg, _, _ = ppo.load_checkpoint(TEACHER)
    env_cfg = dataclasses.replace(env_cfg, n_envs=args.episodes, p_wide=0.0, rand_action_noise=0.0)
    dt = 1 / env_cfg.policy_hz
    results = []
    for cond in args.conditions.split(","):
        for var in args.variants.split(";"):
            t0 = time.time()
            ov = parse_variant(var)
            base = condition(cond)
            est_ov = {k: v for k, v in ov.items() if not k.startswith("det.")}
            det_ov = {k[4:]: v for k, v in ov.items() if k.startswith("det.")}
            ecfg = dataclasses.replace(base, est=dataclasses.replace(base.est, **est_ov),
                                       det=dataclasses.replace(base.det, **det_ov))  # fmt: skip
            rec = make_rollout(EstEnv(env_cfg, ecfg), net, params, norm, args.steps)(jax.random.key(args.seed))
            s = summarise(rec, dt)
            row = {"condition": cond, "variant": var, "overrides": ov,
                   "next_pose_wrong": s["next_pose_wrong_frac"], "after_next_pose_wrong": s["after_next_pose_wrong_frac"],
                   "runs": s["next_pose_wrong_runs"], "by_phase": s["next_pose_wrong_by_phase"],
                   "causes": s["wrong_next_event_causes"], "next_gate_err_m": s["next_gate_err_m"],
                   "next_gate_err_m_by_source": s["next_gate_err_m_by_source"],
                   "passes_true": s["gate_passes_true"], "passes_est": s["est_passes"], "relocks": s["relocks"],
                   "geo_jumps": s["geo_jumps"], "after_pass": after_pass_breakdown(rec, dt), "wall_s": round(time.time() - t0, 1)}  # fmt: skip
            results.append(row)
            print(f"[{cond}] {var:40s} next-pose wrong {row['next_pose_wrong']:.4f} after-next {row['after_next_pose_wrong']:.4f} "
                  f"runs/min {row['runs']['per_flight_minute']:.2f} passes {row['passes_true']}/{row['passes_est']} "
                  f"geo {row['geo_jumps']} relock {row['relocks']}\n    after-pass {json.dumps(row['after_pass'])}", flush=True)  # fmt: skip
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
