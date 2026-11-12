"""Train the state-based teacher with PPO in Crazyflow.

    CUDA_VISIBLE_DEVICES=0 .venv-sim/bin/python scripts/train_teacher.py --name base --seed 0 \
        --set ppo.total_steps=2e9 env.w_cam=0.05

Any field of `fpv.env.EnvConfig` / `fpv.ppo.PPOConfig` can be overridden with
``--set env.<field>=<value> ppo.<field>=<value>``. Outputs in ``runs/teacher/<name>/``:
``config.json``, ``metrics.jsonl`` (one line per logged iteration), ``ckpt_<iter>.pkl`` and
``ckpt_latest.pkl`` (params + observation normaliser + full config).
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from fpv import ppo  # noqa: E402
from fpv.env import EnvConfig, RaceEnv  # noqa: E402


def parse_overrides(pairs: list[str], env_cfg: EnvConfig, ppo_cfg: ppo.PPOConfig):
    for pair in pairs:
        key, value = pair.split("=", 1)
        scope, field = key.split(".", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        cfg = env_cfg if scope == "env" else ppo_cfg
        assert scope in ("env", "ppo") and field in {f.name for f in dataclasses.fields(cfg)}, f"unknown option {key}"
        if isinstance(value, list):
            value = tuple(value)
        if scope == "env":
            env_cfg = dataclasses.replace(env_cfg, **{field: value})
        else:
            ppo_cfg = dataclasses.replace(ppo_cfg, **{field: value})
    return env_cfg, ppo_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="scope.field=value")
    ap.add_argument("--out", default=str(ROOT / "runs" / "teacher"))
    ap.add_argument("--log-every", type=int, default=20, help="iterations between metric lines")
    ap.add_argument("--ckpt-every", type=int, default=500, help="iterations between numbered checkpoints")
    ap.add_argument("--init", default=None, help="checkpoint to warm-start params + normaliser from")
    args = ap.parse_args()

    env_cfg, ppo_cfg = parse_overrides(args.set, EnvConfig(), ppo.PPOConfig(seed=args.seed))
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    ppo.write_config(out / "config.json", env_cfg, ppo_cfg, {"argv": sys.argv[1:]})

    env = RaceEnv(env_cfg)
    net, init, iterate, n_iters = ppo.make_trainer(env, ppo_cfg)
    runner = init(jax.random.key(ppo_cfg.seed))
    if args.init:
        _, params, norm, *_ = ppo.load_checkpoint(args.init)
        runner = runner | dict(params=params, norm=norm)
    batch = env_cfg.n_envs * ppo_cfg.n_steps
    print(f"{args.name}: {n_iters} iterations x {batch} steps, device {jax.devices()[0]}", flush=True)

    def save(name, it, wall):
        extra = dict(iteration=it, env_steps=it * batch, wall_s=wall)
        ppo.save_checkpoint(out / name, runner["params"], runner["norm"], env_cfg, ppo_cfg, extra)

    t_start = time.perf_counter()
    runner, m = iterate(runner)  # compile
    jax.block_until_ready(m)
    compile_s = time.perf_counter() - t_start
    print(f"compiled in {compile_s:.1f} s", flush=True)
    t0, it0, acc = time.perf_counter(), 1, []
    with open(out / "metrics.jsonl", "a") as log:
        for it in range(2, n_iters + 1):
            runner, m = iterate(runner)
            acc.append(m)
            if it % args.log_every == 0 or it == n_iters:
                acc = jax.device_get(acc)
                now = time.perf_counter()
                row = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
                row["episodes"] = float(np.sum([a["episodes"] for a in acc]))
                row |= dict(iteration=it, env_steps=it * batch, wall_s=now - t_start,
                            steps_per_s=(it - it0) * batch / (now - t0))  # fmt: skip
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    f"it {it:6d} steps {it * batch:.3g} ret {row['ep_return']:8.2f} gates/ep {row['ep_gates']:7.2f} "
                    f"crash {row['crash_rate']:.3f} len {row['ep_len']:7.1f} fov {row['in_fov']:.3f} "
                    f"v {row['speed']:.2f} std {row['std']:.3f} kl {row['kl']:.4f} sps {row['steps_per_s']:.3g}",
                    flush=True,
                )
                t0, it0, acc = now, it, []
                if not np.isfinite(row["reward"]):
                    raise SystemExit("non-finite reward, aborting")
            if it % args.ckpt_every == 0:
                save(f"ckpt_{it:06d}.pkl", it, time.perf_counter() - t_start)
                save("ckpt_latest.pkl", it, time.perf_counter() - t_start)
    save("ckpt_latest.pkl", n_iters, time.perf_counter() - t_start)
    print(f"done in {time.perf_counter() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
