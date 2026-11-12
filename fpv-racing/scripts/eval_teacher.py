"""Deterministic evaluation of a teacher checkpoint + rollout export.

    CUDA_VISIBLE_DEVICES=0 .venv-sim/bin/python scripts/eval_teacher.py runs/teacher/<run>/ckpt_latest.pkl

Runs the mean action for one full episode (time limit of the training config) in
``--episodes`` randomised environments and ``--nominal-episodes`` environments on the nominal
circuit (no dynamics / gate randomisation, spawn noise only), both with the standard spawn
(``p_wide=0``), plus ``--recovery-episodes`` randomised episodes that all start from the wide spawn
distribution of the config (``p_wide=1``; section "recovery" of eval.json: fraction passing one
lap without crashing, crash causes, time to the first gate, failure rate by start condition).
Writes next to the checkpoint (or ``--out``): ``eval.json`` (metrics) and ``rollouts.npz``.

rollouts.npz (E = exported episodes, randomised first then nominal; T = max steps; G = gates):
    t          (T,)       time of the step [s], t[k] = k / 50
    pos        (E, T, 3)  world position [m] at the step (before the action is applied)
    quat       (E, T, 4)  body-to-world quaternion, xyzw
    vel        (E, T, 3)  world velocity [m/s]
    ang_vel    (E, T, 3)  body rates [rad/s]
    obs        (E, T, 43) teacher observation (fpv.env.teacher_obs, un-normalised)
    action     (E, T, 4)  policy action in [-1, 1] taken at the step (fpv.env.scale_action)
    target     (E, T)     index of the next gate at the step
    passed     (E, T)     the action of this step carried the drone through its target gate
    length     (E,)       valid steps; entries at index >= length are padding (zeros)
    outcome    (E,)       0 time limit, 1 gate crash, 2 ground, 3 out of bounds
    gates_pos  (E, G, 3)  per-episode gate centres
    gates_yaw  (E, G)     per-episode gate yaws
    nominal    (E,)       episode is on the nominal circuit
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fpv import ppo  # noqa: E402
from fpv.env import RaceEnv  # noqa: E402

OUTCOMES = ("time_limit", "gate_crash", "ground", "out_of_bounds")


def run_episodes(env: RaceEnv, net, params, norm, key, n_export: int) -> dict:
    """One deterministic episode per environment. Returns numpy arrays, time-major (T, N, ...)."""
    n_export = min(n_export, env.cfg.n_envs)

    @jax.jit
    def rollout(key):
        state, obs = env.reset(key)

        def step(carry, _):
            state, obs, alive = carry
            action = ppo.policy_action(net, params, norm, obs)
            s = state.data.states
            rec = dict(
                pos=s.pos[:, 0], quat=s.quat[:, 0], vel=s.vel[:, 0], ang_vel=s.ang_vel[:, 0],
                obs=obs[:n_export], action=action, target=state.target, alive=alive,
            )  # fmt: skip
            state, reward, ev = env.step_core(state, action)
            outcome = jnp.where(ev["crash_gate"], 1, jnp.where(ev["ground"], 2, jnp.where(ev["oob"], 3, 0)))
            rec |= dict(passed=ev["passed"] & alive, in_fov=ev["in_fov"], reward=reward, outcome=outcome,
                        ended=alive & (ev["terminated"] | ev["truncated"]))  # fmt: skip
            return (state, env.obs(state), alive & ~(ev["terminated"] | ev["truncated"])), rec

        carry = (state, obs, jnp.ones(env.cfg.n_envs, bool))
        _, rec = jax.lax.scan(step, carry, None, env.cfg.max_steps)
        return rec, state.gates_pos, state.gates_yaw

    rec, gates_pos, gates_yaw = jax.device_get(rollout(key))
    return rec | dict(gates_pos=gates_pos, gates_yaw=gates_yaw)


def metrics(rec: dict, n_gates: int, dt: float) -> dict:
    alive = rec["alive"]  # (T, N) step is part of the episode
    T, N = alive.shape
    n_alive = alive.sum()
    length = alive.sum(0)
    gates = rec["passed"].sum(0)
    ended = rec["ended"]
    outcome = (rec["outcome"] * ended).sum(0)  # one ending step per episode
    crashed = outcome > 0

    # Lap times: 6 consecutive gate passes. First lap from the spawn (standing start); flying laps
    # between passes k and k + n_gates, non-overlapping from the first pass.
    first, flying = [], []
    for i in np.flatnonzero(gates >= n_gates):
        steps = (np.flatnonzero(rec["passed"][:, i]) + 1) * dt
        first.append(steps[n_gates - 1])
        flying.extend(np.diff(steps[::n_gates]))
    speed = np.linalg.norm(rec["vel"], axis=-1)
    rates = np.abs(rec["ang_vel"])
    a = rec["action"]
    pair = alive[1:] & alive[:-1]
    da = np.abs(np.diff(a, axis=0))[pair]  # (M, 4)
    d1 = np.diff(a, axis=0)
    flip = (d1[1:] * d1[:-1] < 0) & (np.abs(d1[1:]) > 0.2) & (np.abs(d1[:-1]) > 0.2)
    flip = flip[pair[1:] & pair[:-1]]

    def per_axis(x):
        return [round(float(v), 4) for v in x]

    return {
        "episodes": int(N),
        "frac_3_laps": float((gates >= 3 * n_gates).mean()),
        "frac_1_lap": float((gates >= n_gates).mean()),
        "frac_full_episode_no_crash": float((~crashed).mean()),
        "gates_per_episode_mean": float(gates.mean()),
        "gates_per_episode_median": float(np.median(gates)),
        "laps_per_episode_mean": float((gates // n_gates).mean()),
        "crash_rate": float(crashed.mean()),
        "crash_rate_by_cause": {name: float((outcome == k).mean()) for k, name in enumerate(OUTCOMES) if k},
        "crashes_per_gate_attempt": float(crashed.sum() / max(gates.sum() + crashed.sum(), 1)),
        "episode_length_s_mean": float(length.mean() * dt),
        "lap_time_s": {
            "flying_mean": float(np.mean(flying)) if flying else None,
            "flying_median": float(np.median(flying)) if flying else None,
            "flying_best": float(np.min(flying)) if flying else None,
            "flying_worst": float(np.max(flying)) if flying else None,
            "first_lap_from_spawn_mean": float(np.mean(first)) if first else None,
            "n_flying_laps": len(flying),
        },
        "per_target_gate": {
            "crash_share": per_axis(np.bincount(rec["target"][ended & (rec["outcome"] > 0)], minlength=n_gates)
                                    / max(crashed.sum(), 1)),
            "in_fov_frac": per_axis([rec["in_fov"][alive & (rec["target"] == g)].mean() for g in range(n_gates)]),
            "speed_mean": per_axis([speed[alive & (rec["target"] == g)].mean() for g in range(n_gates)]),
        },
        "speed_m_s": {"mean": float(speed[alive].mean()), "p95": float(np.percentile(speed[alive], 95)),
                      "max": float(speed[alive].max())},  # fmt: skip
        "gate_in_fov_frac": float(rec["in_fov"][alive].mean()),
        "body_rate_rad_s": {
            "abs_mean_xyz": per_axis(rates[alive].mean(0)),
            "abs_p99_xyz": per_axis(np.percentile(rates[alive], 99, axis=0)),
            "abs_max_xyz": per_axis(rates[alive].max(0)),
            "norm_mean": float(np.linalg.norm(rec["ang_vel"], axis=-1)[alive].mean()),
        },
        "action": {
            "order": ["thrust", "roll_rate", "pitch_rate", "yaw_rate"],
            "mean": per_axis(a[alive].mean(0)),
            "abs_change_mean": per_axis(da.mean(0)),
            "abs_change_p99": per_axis(np.percentile(da, 99, axis=0)),
            "frac_change_gt_0.5": per_axis((da > 0.5).mean(0)),
            "frac_saturated_gt_0.98": per_axis((np.abs(a[alive]) > 0.98).mean(0)),
            "frac_zigzag_gt_0.2": per_axis(flip.mean(0)),
        },
        "reward_per_step_mean": float(rec["reward"][alive].mean()),
        "steps": int(n_alive),
    }


def export(recs: list[tuple[dict, int, bool]], dt: float) -> dict:
    out = {}
    keys = ("pos", "quat", "vel", "ang_vel", "obs", "action", "target", "passed")
    parts = {k: [] for k in (*keys, "length", "outcome", "gates_pos", "gates_yaw", "nominal")}
    for rec, n, nominal in recs:
        alive = rec["alive"][:, :n]
        for k in keys:
            x = np.swapaxes(rec[k][:, :n], 0, 1)  # (E, T, ...)
            mask = alive.T.reshape(n, -1, *[1] * (x.ndim - 2))
            parts[k].append(np.where(mask, x, 0))
        parts["length"].append(alive.sum(0))
        parts["outcome"].append((rec["outcome"] * rec["ended"]).sum(0)[:n])
        parts["gates_pos"].append(rec["gates_pos"][:n])
        parts["gates_yaw"].append(rec["gates_yaw"][:n])
        parts["nominal"].append(np.full(n, nominal))
    for k, v in parts.items():
        out[k] = np.concatenate(v)
    T = out["pos"].shape[1]
    out["t"] = (np.arange(T) * dt).astype(np.float32)
    for k in ("pos", "quat", "vel", "ang_vel", "obs", "action", "gates_pos", "gates_yaw"):
        out[k] = out[k].astype(np.float32)
    out["target"] = out["target"].astype(np.int8)
    out["passed"] = out["passed"].astype(bool)
    out["length"] = out["length"].astype(np.int32)
    out["outcome"] = out["outcome"].astype(np.int8)
    return out


def recovery_metrics(rec: dict, n_gates: int, dt: float) -> dict:
    """Episodes that all start from the wide spawn distribution: does the policy get back on track?"""
    alive = rec["alive"]
    gates = rec["passed"].sum(0)
    outcome = (rec["outcome"] * rec["ended"]).sum(0)
    crashed = outcome > 0
    ok = gates >= n_gates  # one full lap before any crash
    any_pass = gates > 0
    t_first = (np.argmax(rec["passed"], axis=0) + 1) * dt
    crash_t = alive.sum(0) * dt

    # Start conditions, from the first recorded step
    pos, vel, quat, tgt = rec["pos"][0], rec["vel"][0], rec["quat"][0], rec["target"][0]
    idx = np.arange(len(tgt))
    gate, start = rec["gates_pos"][idx, tgt], rec["gates_pos"][idx, (tgt - 1) % n_gates]
    seg = gate - start
    seg_len = np.linalg.norm(seg, axis=-1)
    seg_dir = seg / seg_len[:, None]
    rel = pos - start
    side = np.stack([-seg_dir[:, 1], seg_dir[:, 0]], -1)
    side /= np.linalg.norm(side, axis=-1, keepdims=True)
    to_gate = gate - pos
    dist = np.linalg.norm(to_gate, axis=-1)
    speed = np.linalg.norm(vel, axis=-1)
    x, y, z, w = quat.T
    heading = np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))
    yaw_err = np.angle(np.exp(1j * (heading - np.arctan2(to_gate[:, 1], to_gate[:, 0]))))
    cond = {
        "segment_fraction": ((rel * seg_dir).sum(-1) / seg_len, [0, 0.25, 0.5, 0.75, 0.9, 1.0001]),
        "lateral_offset_abs_m": (np.abs((rel[:, :2] * side).sum(-1)), [0, 0.25, 0.5, 0.75, 1.0001]),
        "distance_to_target_m": (dist, [0, 0.5, 1, 1.5, 2.5, 10]),
        "height_m": (pos[:, 2], [0, 0.6, 0.9, 1.2, 1.5, 3]),
        "speed_m_s": (speed, [0, 1, 2, 3, 4, 5.0001]),
        "velocity_toward_target_cos": ((vel * to_gate).sum(-1) / (speed * dist + 1e-9), [-1, -0.5, 0, 0.5, 1.0001]),
        "yaw_error_abs_deg": (np.degrees(np.abs(yaw_err)), [0, 20, 40, 60.01]),
        "tilt_deg": (np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))), [0, 10, 20, 30, 45]),
        "body_rate_norm_rad_s": (np.linalg.norm(rec["ang_vel"][0], axis=-1), [0, 1.5, 3, 4.5, 6]),
        "target_gate": (tgt, list(range(n_gates + 1))),
    }
    fail_by = {}
    for name, (v, edges) in cond.items():
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            rows.append({"bin": [round(float(lo), 2), round(float(hi), 2)], "n": int(m.sum()),
                         "fail_rate": round(float((~ok[m]).mean()), 4) if m.any() else None})  # fmt: skip
        fail_by[name] = rows
    return {
        "episodes": int(len(tgt)),
        "frac_1_lap_no_crash": float(ok.mean()),
        "frac_3_laps": float((gates >= 3 * n_gates).mean()),
        "frac_full_episode_no_crash": float((~crashed).mean()),
        "crash_rate": float(crashed.mean()),
        "crash_rate_by_cause": {name: float((outcome == k).mean()) for k, name in enumerate(OUTCOMES) if k},
        "frac_crash_before_first_gate": float((crashed & ~any_pass).mean()),
        "frac_crash_after_first_lap": float((crashed & ok).mean()),
        "frac_no_gate_no_crash": float((~crashed & ~any_pass).mean()),
        "crash_time_s_median": float(np.median(crash_t[crashed])) if crashed.any() else None,
        "frac_crash_within_0.5s": float((crashed & (crash_t <= 0.5)).mean()),
        "frac_1_lap_no_crash_start_ge_1m_from_target": float(ok[dist >= 1.0].mean()),
        "frac_1_lap_no_crash_start_lt_1m_from_target": float(ok[dist < 1.0].mean()),
        "frac_starts_lt_1m_from_target": float((dist < 1.0).mean()),
        "time_to_first_gate_s": {
            "mean": float(t_first[any_pass].mean()), "median": float(np.median(t_first[any_pass])),
            "p95": float(np.percentile(t_first[any_pass], 95)), "max": float(t_first[any_pass].max()),
        },  # fmt: skip
        "gates_per_episode_mean": float(gates.mean()),
        "gate_in_fov_frac": float(rec["in_fov"][alive].mean()),
        "gate_in_fov_frac_first_2s": float(rec["in_fov"][:100][alive[:100]].mean()),
        "fail_rate_by_start_condition": fail_by,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--nominal-episodes", type=int, default=1024)
    ap.add_argument("--export", type=int, default=192, help="randomised episodes in rollouts.npz")
    ap.add_argument("--export-nominal", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--recovery-episodes", type=int, default=4096, help="wide-spawn episodes (0 = skip)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-rollouts", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out = Path(args.out) if args.out else ckpt_path.parent
    out.mkdir(parents=True, exist_ok=True)
    net, params, norm, env_cfg, _, ckpt = ppo.load_checkpoint(ckpt_path)
    env_cfg = dataclasses.replace(env_cfg, rand_action_noise=0.0)  # evaluate the unperturbed policy
    dt = 1 / env_cfg.policy_hz
    result = {"checkpoint": str(ckpt_path), "iteration": ckpt.get("iteration"), "env_steps": ckpt.get("env_steps"),
              "train_wall_s": ckpt.get("wall_s"), "seed": args.seed}  # fmt: skip
    recs = []
    cases = (
        ("randomised", dataclasses.replace(env_cfg, n_envs=args.episodes, p_wide=0.0), args.export, False),
        ("nominal", env_cfg.nominal(n_envs=args.nominal_episodes, p_wide=0.0), args.export_nominal, True),
    )
    for i, (name, cfg, n_export, nominal) in enumerate(cases):
        env = RaceEnv(cfg)
        rec = run_episodes(env, net, params, norm, jax.random.key(args.seed + i), n_export)
        result[name] = metrics(rec, env.n_gates, dt)
        recs.append((rec, min(n_export, cfg.n_envs), nominal))
        m = result[name]
        print(
            f"{name:>10}: >=3 laps {m['frac_3_laps']:.4f}  gates/ep {m['gates_per_episode_mean']:.1f}  "
            f"crash {m['crash_rate']:.4f} {m['crash_rate_by_cause']}  lap {m['lap_time_s']['flying_mean']} "
            f"(best {m['lap_time_s']['flying_best']})  v {m['speed_m_s']['mean']:.2f}/{m['speed_m_s']['max']:.2f}  "
            f"fov {m['gate_in_fov_frac']:.4f}  |da| {m['action']['abs_change_mean']}",
            flush=True,
        )
    if args.recovery_episodes:
        env = RaceEnv(dataclasses.replace(env_cfg, n_envs=args.recovery_episodes, p_wide=1.0))
        rec = run_episodes(env, net, params, norm, jax.random.key(args.seed + 2), 0)
        m = result["recovery"] = recovery_metrics(rec, env.n_gates, dt)
        print(
            f"  recovery: 1 lap no crash {m['frac_1_lap_no_crash']:.4f}  crash {m['crash_rate']:.4f} "
            f"{m['crash_rate_by_cause']}  before first gate {m['frac_crash_before_first_gate']:.4f}  "
            f"t first gate {m['time_to_first_gate_s']}  fov {m['gate_in_fov_frac']:.4f}",
            flush=True,
        )
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    if not args.no_rollouts:
        np.savez_compressed(out / "rollouts.npz", **export(recs, dt))
    print(f"wrote {out}/eval.json" + ("" if args.no_rollouts else " and rollouts.npz"))


if __name__ == "__main__":
    main()
