"""Perception pipeline on an acrobatic layout: estimator checks (teacher flying from its TRUE state, the estimator
running alongside, open loop) and the closed-loop evaluation of an estimated-observation policy.

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 .venv-rocm/bin/python scripts/acro_est.py \
        --policy teacher:runs/teacher/acro/ss_inv20/ckpt_latest.pkl --conditions perfect,x1 --out <dir>
    ... --policy runs/percep/policy/train/est_acro_v5/ckpt_latest.pkl --conditions perfect,x1 --out <dir>

Episodes: the policy checkpoint's env config (track split_s, z_max 6 m, its randomisation), stick noise off,
standard spawns, 256 x 30 s, key ``--seed`` (the same episodes for every condition, as `scripts/eval_est.py`).
The estimator config: the est-policy checkpoint's, or for ``teacher:`` the stage-4 v5 settings
(place_mem_w=0.3, rescue_after_next=False). Conditions: `fpv.env_est.condition` (perfect, x1, ...).

<out>/acro_est_<condition>.json:
  standard   `eval_teacher.metrics` (>= 3 laps, crashes, lap time, per-target crash share ...)
  dive / laps / camera   `scripts/eval_acro.acro_metrics` (inversion per dive and per lap, next gate in view)
  crash_phase  share of crashes by where they happen: approaching the high gate G1 ("before_dive", target G1),
               between G1 and G2 ("dive"), after G2 ("after_dive", targets G3..G6)
  perception   per phase (dive steps = target G2 after a G1 pass; inverted = tilt > 90 deg; rest): estimated vs
               true up vector angle (deg, mean / p95 / max), wrong-next-gate share, next-gate pose wrong share
               (|estimated - true next-gate centre| in the body frame > max(0.5 m, 0.15 * distance), as eval_est),
               lock share
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from fpv import acro  # noqa: E402

acro.register()

import eval_acro  # noqa: E402  (imports eval_teacher)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fpv import ppo, ppo_est  # noqa: E402
from fpv.env import quat_to_mat  # noqa: E402
from fpv.env_est import EstEnv, condition  # noqa: E402
from fpv.perception import estimator  # noqa: E402
from fpv.perception.estimator import EST_OBS_DIM  # noqa: E402

V5 = dict(place_mem_w=0.3, rescue_after_next=False)


def make_policy(spec):
    if spec.startswith("teacher:"):
        net, params, norm, env_cfg, _, _ = ppo.load_checkpoint(spec.split(":", 1)[1])
        return (lambda full: ppo.policy_action(net, params, norm, full[:, EST_OBS_DIM:])), env_cfg, None
    net, params, norm, env_cfg, ecfg, _, _ = ppo_est.load_checkpoint(spec)
    return (lambda full: ppo_est.policy_action(net, params, norm, full[:, :EST_OBS_DIM])), env_cfg, ecfg


def make_rollout(env: EstEnv, act):
    @jax.jit
    def rollout(key):
        state, full = env.reset(key)

        def step(carry, _):
            state, full, alive = carry
            race, est = state.race, state.est
            action = act(full)
            s = race.data.states
            Pc, Py, own, T = estimator.gate_estimates(est, race.track_pos, race.track_yaw, env.ecfg.est)
            gate = jnp.take_along_axis(race.gates_pos, race.target[:, None, None], 1)[:, 0]
            R = quat_to_mat(s.quat[:, 0])
            c_true = jnp.einsum("nji,nj->ni", R, gate - s.pos[:, 0])
            c_est = jnp.einsum("nji,nj->ni", T, jnp.take_along_axis(Pc, est.nxt[:, None, None], 1)[:, 0])
            d_true = jnp.linalg.norm(c_true, axis=-1)
            u_est, u_true = full[:, 9:12], full[:, EST_OBS_DIM + 9 : EST_OBS_DIM + 12]
            cosu = (u_est * u_true).sum(-1) / (jnp.linalg.norm(u_est, axis=-1) * jnp.linalg.norm(u_true, axis=-1) + 1e-9)
            rec = dict(
                pos=s.pos[:, 0], quat=s.quat[:, 0], vel=s.vel[:, 0], ang_vel=s.ang_vel[:, 0], action=action,
                target=race.target, alive=alive, wrong=(est.nxt != race.target) & alive,
                pose_wrong=(jnp.linalg.norm(c_est - c_true, axis=-1) > jnp.maximum(0.5, 0.15 * d_true)) & alive,
                up_err_deg=jnp.degrees(jnp.arccos(jnp.clip(cosu, -1, 1))), locked=est.locked, nxt=est.nxt,
            )  # fmt: skip
            state2, full2, reward, done, ev = env.step_core_est(state, action, reset_done=False)
            outcome = jnp.where(ev["crash_gate"], 1, jnp.where(ev["ground"], 2, jnp.where(ev["oob"], 3, 0)))
            rec |= dict(passed=ev["passed"] & alive, in_fov=ev["in_fov"], reward=reward, outcome=outcome,
                        ended=alive & (ev["terminated"] | ev["truncated"]))  # fmt: skip
            return (state2, full2, alive & ~(ev["terminated"] | ev["truncated"])), rec

        _, rec = jax.lax.scan(step, (state, full, jnp.ones(env.cfg.n_envs, bool)), None, env.cfg.max_steps)
        return rec, state.race.gates_pos, state.race.gates_yaw

    def run(key):
        rec, gp, gy = jax.device_get(rollout(key))
        return {k: np.asarray(v) for k, v in rec.items()} | dict(gates_pos=gp, gates_yaw=gy)

    return run


def phase_masks(rec):
    alive, passed, target = rec["alive"], rec["passed"], rec["target"]
    T, N = alive.shape
    dive = np.zeros_like(alive)
    for e in range(N):
        for a, b in zip(*eval_acro.runs_of(alive[:, e] & (target[:, e] == 1))):
            if a > 0 and passed[a - 1, e] and target[a - 1, e] == 0:
                dive[a:b, e] = True
    _, bz = eval_acro.body_axes(rec["quat"])
    inv = alive & (bz[..., 2] < 0)
    return dict(dive=dive, inverted=inv, rest=alive & ~dive, all=alive)


def percep(rec):
    out = {}
    for name, m in phase_masks(rec).items():
        if not m.any():
            out[name] = None
            continue
        ue = rec["up_err_deg"][m]
        out[name] = dict(steps=int(m.sum()), up_err_deg_mean=round(float(ue.mean()), 2),
                         up_err_deg_p95=round(float(np.percentile(ue, 95)), 2), up_err_deg_max=round(float(ue.max()), 2),
                         wrong_next=round(float(rec["wrong"][m].mean()), 4),
                         next_pose_wrong=round(float(rec["pose_wrong"][m].mean()), 4),
                         locked=round(float(rec["locked"][m].mean()), 4),
                         next_gate_in_fov=round(float(rec["in_fov"][m].mean()), 4),
                         # which gate the estimator shows as next on the wrong steps (offset from the true next gate:
                         # -1 = still the previous gate, +1 = already the one after)
                         wrong_offset_share=_offsets(rec, m))  # fmt: skip
    out["dive_wrong_by_steps_since_G1"] = _dive_timing(rec, phase_masks(rec)["dive"])
    return out


def _dive_timing(rec, dive):
    """Wrong-next share of dive steps by steps since the G1 pass (50 Hz)."""
    k = np.zeros(dive.shape, int)
    for t in range(1, dive.shape[0]):
        k[t] = np.where(dive[t], np.where(dive[t - 1], k[t - 1] + 1, 0), 0)
    res = {}
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 1000)):
        m = dive & (k >= lo) & (k < hi)
        res[f"{lo}-{hi}"] = dict(steps=int(m.sum()), wrong=round(float(rec["wrong"][m].mean()), 4) if m.any() else None,
                                 in_fov=round(float(rec["in_fov"][m].mean()), 4) if m.any() else None)
    return res


def _offsets(rec, m):
    w = m & rec["wrong"]
    off = (rec["nxt"].astype(int) - rec["target"].astype(int) + 3) % 6 - 3
    vals, cnt = np.unique(off[w], return_counts=True)
    return {str(int(v)): round(float(c / max(w.sum(), 1)), 4) for v, c in zip(vals, cnt)}


def crash_phase(rec):
    end = rec["ended"] & (rec["outcome"] > 0)
    tg = rec["target"][end]
    n = max(len(tg), 1)
    return dict(n_crashes=int(len(tg)), before_dive=round(float((tg == 0).sum() / n), 4),
                dive=round(float((tg == 1).sum() / n), 4), after_dive=round(float((tg >= 2).sum() / n), 4))  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--conditions", default="perfect,x1")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--est", nargs="*", default=[], metavar="field=value", help="EstConfig overrides")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    act, env_cfg, p_ecfg = make_policy(args.policy)
    est_cfg = p_ecfg.est if p_ecfg is not None else dataclasses.replace(condition("x1").est, **V5)
    from check_estimator_sweep import parse_variant

    est_cfg = dataclasses.replace(est_cfg, **parse_variant(",".join(args.est)))
    env_cfg = dataclasses.replace(env_cfg, rand_action_noise=0.0, p_wide=0.0, n_envs=args.episodes)
    dt = 1 / env_cfg.policy_hz
    for cond in args.conditions.split(","):
        ecfg = dataclasses.replace(condition(cond), est=est_cfg)
        env = EstEnv(env_cfg, ecfg)
        rec = make_rollout(env, act)(jax.random.key(args.seed))
        m = eval_acro.et.metrics(rec, 6, dt)
        am = eval_acro.acro_metrics(rec, dt)
        am.pop("_dives")
        res = dict(policy=args.policy, condition=cond, seed=args.seed, episodes=args.episodes, track=env_cfg.track,
                   est=dataclasses.asdict(est_cfg), standard=m, crash_phase=crash_phase(rec), perception=percep(rec), **am)  # fmt: skip
        (out / f"acro_est_{cond}.json").write_text(json.dumps(res, indent=1))
        d = am["dive"]
        print(f"[{cond}] " + eval_acro.et.line(env_cfg.track, m), flush=True)
        print(f"[{cond}] laps>150 {am['laps']['frac_tilt_gt_150']}  laps>90 {am['laps']['frac_tilt_gt_90']}  dives {d['n_completed']}"
              f"  crash phase {res['crash_phase']}", flush=True)  # fmt: skip
        for k, v in res["perception"].items():
            print(f"[{cond}]   {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
