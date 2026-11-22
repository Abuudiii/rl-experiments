"""Check that the acro additions leave the standard tracks unchanged: the published teacher on a standard
layout, same seed, before and after `fpv.acro.register()` in the same process (rollout arrays compared
bit for bit), plus the eval metrics.

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 .venv-rocm/bin/python scripts/acro_check_defaults.py \
        [--layout circuit] [--episodes 512] [--out runs/teacher/acro/defaults_check.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_teacher as et  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from fpv import ppo, track  # noqa: E402
from fpv.env import EnvConfig, RaceEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/teacher/rt_G_D_noise3/ckpt_latest.pkl"))
    ap.add_argument("--layout", default="circuit")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--out", default=str(ROOT / "runs/teacher/acro/defaults_check.json"))
    args = ap.parse_args()
    net, params, norm, cfg, _, _ = ppo.load_checkpoint(args.checkpoint)
    cfg = dataclasses.replace(cfg, n_envs=args.episodes, p_wide=0.0, rand_action_noise=0.0, track=args.layout)

    def run():
        env = RaceEnv(cfg)
        rec = et.make_rollout(env, net, params, norm, np.arange(1))(jax.random.key(args.seed))
        h = hashlib.sha256()
        for k in ("pos", "quat", "vel", "action", "passed", "alive"):
            h.update(np.ascontiguousarray(rec[k]).tobytes())
        return et.metrics(rec, env.n_gates, 1 / cfg.policy_hz), h.hexdigest()

    names_before = list(track.SCENARIOS)
    m0, h0 = run()
    from fpv import acro

    acro.register()
    m1, h1 = run()
    res = dict(
        checkpoint=args.checkpoint, layout=args.layout, episodes=args.episodes, seed=args.seed,
        default_track=EnvConfig().track, default_z_max=EnvConfig().z_max,
        scenarios_before_register=names_before, scenarios_after_register=list(track.SCENARIOS),
        rollout_sha256_before=h0, rollout_sha256_after=h1, identical=h0 == h1,
        before={k: m0[k] for k in ("frac_3_laps", "crash_rate", "lap_time_s", "speed_m_s")},
        after={k: m1[k] for k in ("frac_3_laps", "crash_rate", "lap_time_s", "speed_m_s")},
    )  # fmt: skip
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
