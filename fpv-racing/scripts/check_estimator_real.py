"""Open-loop validation of the estimator on REAL detector outputs (not stage 5: the teacher flies from true state).

    CUDA_VISIBLE_DEVICES=1 .venv-sim/bin/python scripts/check_estimator_real.py \
        --detector runs/percep/detector/A_amd/ckpt_best.pkl --episodes 64 --steps 1500 --out runs/percep/policy/checks

Same flights and metrics as `scripts/check_estimator.py` (teacher flying from its true observation, random
tracks, standard spawns, seed 4242), but the detections come from rendering every frame with the synthetic
renderer (`fpv.render_synth` via `labels.make_renderer`, per-episode random look from `labels.scene_pool`,
highlight off, 320x180) and running the stage-1 detector on it (`fpv.perception.detector.load_detector`,
confidence threshold ``--thr``, default 0.4, i.e. the estimator's conf_min is set to it here). The IMU is simulated at 1x.
For the det_stats table, detector slots are matched to true gates (nearest true quad, mean corner distance
< 8 px), so the real detector's corner error / misses are reported in the noise model's format.
Writes <out>/check_real_detector.json.
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
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from check_estimator import det_stats, make_rollout, summarise, TEACHER  # noqa: E402
from fpv import ppo  # noqa: E402
from fpv.env_est import EstEnv, condition  # noqa: E402
from fpv.perception import detnoise, estimator, labels  # noqa: E402
from fpv.perception.detector import load_detector  # noqa: E402


class RealDetEnv(EstEnv):
    """EstEnv whose detections are the stage-1 detector run on synthetic renders of the true state."""

    def __init__(self, cfg, ecfg, det_path, scene_seed=0, thr=0.4):
        super().__init__(cfg, ecfg)
        self.thr = thr
        self.det = load_detector(det_path)
        self.render = labels.make_renderer()
        sc = labels.scene_pool(cfg.n_envs, scene_seed)
        self.scene = {k: jnp.asarray(v) for k, v in sc.items()}

    def _perceive(self, race, det_st, est_st, latency, gyro, acc, dt, key, start_idx):
        s = race.data.states
        pos, quat = s.pos[:, 0], s.quat[:, 0]
        frames, _ = self.render(pos, quat, race.gates_pos, race.gates_yaw, self.scene, key)
        out = self.det.forward(self.det.params, frames)
        corners, conf = out["corners"], out["conf"]
        conf = jnp.where(conf >= self.thr, conf, 0.0)
        # slot -> true gate (for statistics only)
        uv, _, visible, _, _ = detnoise.project_gates(pos, quat, race.gates_pos, race.gates_yaw)
        uv = labels.canonical_order(uv)
        dmat = jnp.linalg.norm(corners[:, :, None] - uv[:, None], axis=-1).mean(-1)  # (N, K, G) (cyclic shifts below)
        for r in range(1, 4):
            dmat = jnp.minimum(dmat, jnp.linalg.norm(corners[:, :, None] - jnp.roll(uv, r, -2)[:, None], axis=-1).mean(-1))
        gid = jnp.where((conf > 0) & (dmat.min(-1) < 8.0), jnp.argmin(dmat, -1), -1)
        est_st, einfo = estimator.est_update(est_st, corners, conf, gyro, acc, race.track_pos, race.track_yaw, dt,
                                             self.ecfg.est, start_idx)  # fmt: skip
        obs = estimator.observation(est_st, race.track_pos, race.track_yaw, race.prev_action, self.ecfg.est)
        return det_st, est_st, obs, dict(det=dict(gate=gid, visible=visible), est=einfo, corners=corners, conf=conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="runs/percep/detector/A_amd/ckpt_best.pkl")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--known-start", action="store_true")
    ap.add_argument("--thr", type=float, default=0.4, help="detector confidence threshold (= estimator conf_min)")
    ap.add_argument("--tag", default="", help="output name suffix, e.g. _E_thr03")
    ap.add_argument("--est", nargs="*", default=[], metavar="field=value", help="EstConfig overrides (e.g. v5 options)")
    ap.add_argument("--out", default="runs/percep/policy/checks")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net, params, norm, env_cfg, _, _ = ppo.load_checkpoint(TEACHER)
    env_cfg = dataclasses.replace(env_cfg, n_envs=args.episodes, p_wide=0.0, rand_action_noise=0.0)
    ecfg = condition("x1")
    from check_estimator_sweep import parse_variant

    est_ov = parse_variant(",".join(args.est))
    ecfg = dataclasses.replace(ecfg, known_start=args.known_start,
                               est=dataclasses.replace(ecfg.est, conf_min=args.thr, **est_ov))  # fmt: skip
    t0 = time.time()
    env = RealDetEnv(env_cfg, ecfg, args.detector, thr=args.thr)
    rec = make_rollout(env, net, params, norm, args.steps)(jax.random.key(args.seed))
    dt = 1 / env_cfg.policy_hz
    res = {"source": "real detector " + args.detector, "thr": args.thr, "est_overrides": est_ov, "episodes": args.episodes,
           "steps": args.steps,
           "seed": args.seed, "known_start": args.known_start, "summary": summarise(rec, dt), "det_stats": det_stats(rec),
           "wall_s": round(time.time() - t0, 1)}  # fmt: skip
    name = "check_real_detector" + args.tag + ("_known" if args.known_start else "") + ".json"
    (out / name).write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("tilt_err_deg", "vel_err_m_s", "next_gate_err_m", "next_pose_wrong_frac",
                                        "after_next_pose_wrong_frac", "wrong_next_frac_of_locked_steps",
                                        "wrong_next_events_per_flight_minute", "next_pose_wrong_by_phase",
                                        "next_pose_wrong_runs")}, indent=1))  # fmt: skip
    print("det stats", json.dumps(res["det_stats"]))
    print(f"wrote {out / name} in {res['wall_s']} s")


if __name__ == "__main__":
    main()
