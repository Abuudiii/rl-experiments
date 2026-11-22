"""Stage 4: retrain the flight policy on the perception pipeline's ESTIMATED observations (PPO).

    HIP_VISIBLE_DEVICES=0 .venv-rocm/bin/python scripts/train_est.py --variant x1 --name est_x1 --seed 401 \
        --init runs/teacher/rt_G_D_noise3/ckpt_latest.pkl --set ppo.total_steps=1e9

Environment = the teacher's training config (dynamics, rewards, randomisation, random tracks, wide spawns,
stick noise) from the ``--init`` checkpoint, observed through `fpv.env_est.EstEnv` with the noise condition
``--variant`` (`fpv.env_est.condition`: perfect, x05, x1, x2, x4, drop, rand). Asymmetric actor-critic
warm-started from the teacher (`fpv.ppo_est`). ``--set env.<f>=v ppo.<f>=v`` overrides as in
train_teacher.py. Outputs in ``--out/<name>/``: config.json, metrics.jsonl, ckpt_<iter>.pkl, ckpt_latest.pkl.
``--stop-at`` (epoch seconds): save and exit cleanly before that time (Slurm job end).
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

import jax  # noqa: E402
import numpy as np  # noqa: E402

from fpv import ppo, ppo_est  # noqa: E402
from fpv.env_est import EstEnv, condition  # noqa: E402


def parse_overrides(pairs, env_cfg, ppo_cfg, ecfg):
    """``scope.field=value`` overrides as scripts/train_teacher.py (not imported: it sets CUDA_VISIBLE_DEVICES,
    which conflicts with HIP_VISIBLE_DEVICES on ROCm); scope ``est`` = the estimator's `EstConfig` (e.g. the
    estimator v5 options), saved with the checkpoint and used by eval_est.py."""
    import ast

    for pair in pairs:
        key, value = pair.split("=", 1)
        scope, field = key.split(".", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        cfg = {"env": env_cfg, "ppo": ppo_cfg, "est": ecfg.est}.get(scope)
        assert cfg is not None and field in {f.name for f in dataclasses.fields(cfg)}, f"unknown option {key}"
        value = tuple(value) if isinstance(value, list) else value
        if scope == "env":
            env_cfg = dataclasses.replace(env_cfg, **{field: value})
        elif scope == "ppo":
            ppo_cfg = dataclasses.replace(ppo_cfg, **{field: value})
        else:
            ecfg = dataclasses.replace(ecfg, est=dataclasses.replace(ecfg.est, **{field: value}))
    return env_cfg, ppo_cfg, ecfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="runs/teacher/rt_G_D_noise3/ckpt_latest.pkl", help="teacher checkpoint (warm start + env config)")
    ap.add_argument("--resume", default=None, help="est-policy checkpoint to continue from (params + normaliser)")
    ap.add_argument("--log-std", type=float, default=None, help="reset the policy std (log) at the warm start")
    ap.add_argument("--set", nargs="*", default=[], metavar="scope.field=value")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "percep" / "policy" / "train"))
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--stop-at", type=float, default=None)
    ap.add_argument("--max-iters", type=int, default=None, help="stop after this many iterations (smoke tests)")
    args = ap.parse_args()

    _, t_params, t_norm, env_cfg, t_ppo, _ = ppo.load_checkpoint(args.init)
    ppo_cfg = dataclasses.replace(t_ppo, seed=args.seed)
    env_cfg, ppo_cfg, ecfg = parse_overrides(args.set, env_cfg, ppo_cfg, condition(args.variant))
    if args.n_envs:
        env_cfg = dataclasses.replace(env_cfg, n_envs=args.n_envs)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    ppo_est.write_config(out / "config.json", env_cfg, ecfg, ppo_cfg, {"argv": sys.argv[1:], "variant": args.variant})

    env = EstEnv(env_cfg, ecfg)
    net, init, iterate, n_iters = ppo_est.make_trainer(env, ppo_cfg)
    key = jax.random.key(ppo_cfg.seed)
    if args.resume:
        _, params, norm, *_ = ppo_est.load_checkpoint(args.resume)
    else:
        params0 = net.init(jax.random.key(0), jax.numpy.zeros((1, ppo_est.FULL_OBS_DIM)))
        params, norm = ppo_est.warm_start(params0, ppo_est.norm_init(), t_params, t_norm, args.log_std)
    runner = jax.jit(init)(key, params, norm)
    batch = env_cfg.n_envs * ppo_cfg.n_steps
    print(f"{args.name}: variant {args.variant}, {n_iters} iterations x {batch} steps, device {jax.devices()[0]}",
          flush=True)  # fmt: skip

    def save(name, it, wall):
        extra = dict(iteration=it, env_steps=it * batch, wall_s=wall, variant=args.variant)
        ppo_est.save_checkpoint(out / name, runner["params"], runner["norm"], env_cfg, ecfg, ppo_cfg, extra)

    t_start = time.perf_counter()
    runner, m = iterate(runner)
    jax.block_until_ready(m)
    print(f"compiled + first iteration in {time.perf_counter() - t_start:.1f} s", flush=True)
    t0, it0, acc = time.perf_counter(), 1, []
    last = n_iters if args.max_iters is None else min(n_iters, args.max_iters)
    it = 1
    with open(out / "metrics.jsonl", "a") as log:
        for it in range(2, last + 1):
            runner, m = iterate(runner)
            acc.append(m)
            if it % args.log_every == 0 or it == last:
                acc = jax.device_get(acc)
                now = time.perf_counter()
                row = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
                row["episodes"] = float(np.sum([a["episodes"] for a in acc]))
                row |= dict(iteration=it, env_steps=it * batch, wall_s=now - t_start,
                            steps_per_s=(it - it0) * batch / (now - t0))  # fmt: skip
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"it {it:6d} steps {it * batch:.3g} ret {row['ep_return']:8.2f} gates/ep {row['ep_gates']:7.2f} "
                      f"crash {row['crash_rate']:.3f} len {row['ep_len']:7.1f} wrong {row['wrong_next']:.3f} "
                      f"v {row['speed']:.2f} std {row['std']:.3f} kl {row['kl']:.4f} ev {row['explained_var']:.3f} "
                      f"sps {row['steps_per_s']:.3g}", flush=True)  # fmt: skip
                t0, it0, acc = now, it, []
                if not np.isfinite(row["reward"]):
                    raise SystemExit("non-finite reward, aborting")
            if it % args.ckpt_every == 0:
                save(f"ckpt_{it:06d}.pkl", it, time.perf_counter() - t_start)
                save("ckpt_latest.pkl", it, time.perf_counter() - t_start)
            if args.stop_at and time.time() > args.stop_at:
                print("stop-at reached", flush=True)
                break
    save("ckpt_latest.pkl", it, time.perf_counter() - t_start)
    print(f"done in {time.perf_counter() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
