"""Deterministic evaluation of a teacher checkpoint + rollout export + flight records.

    CUDA_VISIBLE_DEVICES=0 .venv-sim/bin/python scripts/eval_teacher.py runs/teacher/<run>/ckpt_latest.pkl --seed 7000

Runs the mean action for one full episode (time limit of the training config) per environment.
Sections of eval.json:
  randomised / nominal  the checkpoint's training distribution (``cfg.track``: the circuit, or
        procedural tracks), ``--episodes`` with gate + physics randomisation and
        ``--nominal-episodes`` without (spawn noise only), standard spawn (``p_wide=0``).
  recovery  ``--recovery-episodes`` randomised episodes that all start from the wide spawn
        distribution of the config (``p_wide=1``): fraction passing one lap without crashing, crash
        causes, time to the first gate, failure rate by start condition.
  noise     the randomised setting with Gaussian noise of std ``--noise`` added to every action
        (an imperfect pilot), ``--noise-episodes`` each: crashes per flight-minute etc.
  scenarios per name in fpv.track.SCENARIOS (``--scenarios``): "randomised" (``--scenario-episodes``,
        the config's per-gate and physics randomisation on the fixed layout), "nominal"
        (``--scenario-nominal-episodes``), "noise" (as above) and "recovery"
        (``--scenario-recovery-episodes``).
Every metrics block holds: >= 3 laps, crash rate by cause, crashes per flight-minute, lap time, speed,
fraction of steps with the next gate in the FPV camera's field of view ("gate_in_fov_frac") and how
often it is lost right after a gate ("after_gate_pass").

Writes next to the checkpoint (or ``--out``): ``eval.json``, ``rollouts.npz`` and with
``--records`` one nominal flight per scenario as a flight record, ``records/<scenario>.npz``:
    pos [T,3] world, quat [T,4] xyzw body-to-world, vel [T,3], action [T,4] applied action,
    label [T,4] teacher action for the same state (= action: the teacher flies), gates_pos [6,3],
    gates_yaw [6], passed [T] bool (the step carried the drone through its target gate), target [T],
    hz (50), name, outcome (0 time limit, 1 gate crash, 2 ground, 3 out of bounds). T = 1500 (30 s)
    unless the flight crashed; the episode with the most gates of the nominal block is used.

rollouts.npz (E = exported episodes: training distribution randomised, nominal, then the named
scenarios, randomised then nominal for each; T = max steps; G = gates):
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
    track_pos  (E, G, 3)  per-episode nominal track (gates before the per-gate randomisation)
    track_yaw  (E, G)
    nominal    (E,)       episode without gate / physics randomisation
    scenario   (E,)       index into scenario_names, -1 = the training distribution
    scenario_names (S,)
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

from fpv import ppo, track  # noqa: E402
from fpv.env import RaceEnv  # noqa: E402

OUTCOMES = ("time_limit", "gate_crash", "ground", "out_of_bounds")


def make_rollout(env: RaceEnv, net, params, norm, export_idx):
    """Jitted ``rollout(key, noise) -> rec``: one deterministic episode per environment, Gaussian
    noise of std ``noise`` (N,) added to the flown action. ``rec``: numpy, time-major (T, N, ...);
    ``obs`` only for the environments in ``export_idx``."""
    export_idx = jnp.asarray(np.asarray(export_idx, np.int32))

    @jax.jit
    def rollout(key, noise):
        k_noise = jax.random.fold_in(key, 99)
        state, obs = env.reset(key)

        def step(carry, _):
            state, obs, alive, kn = carry
            action = ppo.policy_action(net, params, norm, obs)
            kn, k = jax.random.split(kn)
            s = state.data.states
            rec = dict(
                pos=s.pos[:, 0], quat=s.quat[:, 0], vel=s.vel[:, 0], ang_vel=s.ang_vel[:, 0],
                obs=obs[export_idx], action=action, target=state.target, alive=alive,
            )  # fmt: skip
            state, reward, ev = env.step_core(state, action + noise[:, None] * jax.random.normal(k, action.shape))
            outcome = jnp.where(ev["crash_gate"], 1, jnp.where(ev["ground"], 2, jnp.where(ev["oob"], 3, 0)))
            rec |= dict(passed=ev["passed"] & alive, in_fov=ev["in_fov"], reward=reward, outcome=outcome,
                        ended=alive & (ev["terminated"] | ev["truncated"]))  # fmt: skip
            return (state, env.obs(state), alive & ~(ev["terminated"] | ev["truncated"]), kn), rec

        carry = (state, obs, jnp.ones(env.cfg.n_envs, bool), k_noise)
        _, rec = jax.lax.scan(step, carry, None, env.cfg.max_steps)
        return rec, state.gates_pos, state.gates_yaw, state.track_pos, state.track_yaw

    def run(key, noise=0.0):
        noise = jnp.broadcast_to(jnp.asarray(noise, jnp.float32), (env.cfg.n_envs,))
        rec, gp, gy, tp, ty = jax.device_get(rollout(key, noise))
        return rec | dict(gates_pos=gp, gates_yaw=gy, track_pos=tp, track_yaw=ty)

    return run


def columns(rec: dict, idx) -> dict:
    """The environments ``idx`` of a time-major record (``obs`` and per-episode arrays included)."""
    out = {}
    for k, v in rec.items():
        if k == "obs":
            continue
        out[k] = v[idx] if k in ("gates_pos", "gates_yaw", "track_pos", "track_yaw") else v[:, idx]
    return out


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

    # Next gate out of the camera right after a gate: per pass, steps out of view in the next 0.5 s
    out = ~rec["in_fov"] & alive
    cs = np.concatenate([np.zeros((1, N), int), np.cumsum(out, 0)])  # cs[k] = out-of-view steps before k
    kk, ee = np.nonzero(rec["passed"])  # pass during step k: the new target is shown from step k + 1
    w = round(0.5 / dt)
    lost = cs[np.minimum(kk + 1 + w, T), ee] - cs[np.minimum(kk + 1, T), ee]

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
        "crashes_per_flight_minute": float(crashed.sum() / max(n_alive * dt / 60, 1e-9)),
        "flight_minutes": float(n_alive * dt / 60),
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
        "after_gate_pass": {
            "passes": int(len(kk)),
            "frac_next_gate_out_of_view_within_0.5s": float((lost > 0).mean()) if len(kk) else None,
            "out_of_view_steps_mean": float(lost.mean()) if len(kk) else None,
            "out_of_view_steps_p95": float(np.percentile(lost, 95)) if len(kk) else None,
        },
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


def export(parts: list[dict], dt: float) -> dict:
    """parts: dict(rec, cols (env indices), obs (T, len(cols), 43), nominal, scenario) each."""
    keys = ("pos", "quat", "vel", "ang_vel", "obs", "action", "target", "passed")
    per_ep = ("length", "outcome", "gates_pos", "gates_yaw", "track_pos", "track_yaw", "nominal", "scenario")
    out = {k: [] for k in (*keys, *per_ep)}
    for p in parts:
        rec, cols = p["rec"], np.asarray(p["cols"])
        n = len(cols)
        alive = rec["alive"][:, cols]
        for k in keys:
            x = np.swapaxes(p["obs"] if k == "obs" else rec[k][:, cols], 0, 1)  # (E, T, ...)
            mask = alive.T.reshape(n, -1, *[1] * (x.ndim - 2))
            out[k].append(np.where(mask, x, 0))
        out["length"].append(alive.sum(0))
        out["outcome"].append((rec["outcome"][:, cols] * rec["ended"][:, cols]).sum(0))
        for k in ("gates_pos", "gates_yaw", "track_pos", "track_yaw"):
            out[k].append(rec[k][cols])
        out["nominal"].append(np.full(n, p["nominal"]))
        out["scenario"].append(np.full(n, p["scenario"]))
    out = {k: np.concatenate(v) for k, v in out.items()}
    T = out["pos"].shape[1]
    out["t"] = (np.arange(T) * dt).astype(np.float32)
    for k in ("pos", "quat", "vel", "ang_vel", "obs", "action", "gates_pos", "gates_yaw", "track_pos", "track_yaw"):
        out[k] = out[k].astype(np.float32)
    out["target"] = out["target"].astype(np.int8)
    out["passed"] = out["passed"].astype(bool)
    out["length"] = out["length"].astype(np.int32)
    out["outcome"] = out["outcome"].astype(np.int8)
    out["scenario"] = out["scenario"].astype(np.int8)
    return out


def write_record(path: Path, rec: dict, i: int, name: str, dt: float):
    """Flight record (the hand-off to the video tools) of environment ``i`` of a nominal rollout."""
    n = int(rec["alive"][:, i].sum())
    np.savez_compressed(
        path, pos=rec["pos"][:n, i].astype(np.float32), quat=rec["quat"][:n, i].astype(np.float32),
        vel=rec["vel"][:n, i].astype(np.float32), action=rec["action"][:n, i].astype(np.float32),
        label=rec["action"][:n, i].astype(np.float32), gates_pos=rec["gates_pos"][i].astype(np.float32),
        gates_yaw=rec["gates_yaw"][i].astype(np.float32), passed=rec["passed"][:n, i].astype(bool),
        target=rec["target"][:n, i].astype(np.int8), hz=np.int32(round(1 / dt)), name=np.str_(name),
        outcome=np.int8((rec["outcome"][:, i] * rec["ended"][:, i]).sum()),
    )  # fmt: skip


def line(name: str, m: dict) -> str:
    return (
        f"{name:>34}: >=3 laps {m['frac_3_laps']:.4f}  crash {m['crash_rate']:.4f} "
        f"{ {k: round(v, 4) for k, v in m['crash_rate_by_cause'].items()} }  /min {m['crashes_per_flight_minute']:.3f}  "
        f"lap {m['lap_time_s']['flying_mean'] and round(m['lap_time_s']['flying_mean'], 2)}  "
        f"v {m['speed_m_s']['mean']:.2f}  fov {m['gate_in_fov_frac']:.4f}  "
        f"lost after gate {m['after_gate_pass']['frac_next_gate_out_of_view_within_0.5s']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=4096, help="training distribution, randomised (and per noise level)")
    ap.add_argument("--nominal-episodes", type=int, default=1024)
    ap.add_argument("--export", type=int, default=192, help="randomised episodes in rollouts.npz")
    ap.add_argument("--export-nominal", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--recovery-episodes", type=int, default=4096, help="wide-spawn episodes (0 = skip)")
    ap.add_argument("--noise", default="0.1,0.2,0.3", help="action-noise stds ('' = skip)")
    ap.add_argument("--scenarios", default="all", help="'all', 'none' or comma-separated names of fpv.track.SCENARIOS")
    ap.add_argument("--scenario-episodes", type=int, default=2048, help="randomised, per scenario and noise level")
    ap.add_argument("--scenario-nominal-episodes", type=int, default=256)
    ap.add_argument("--scenario-recovery-episodes", type=int, default=1024)
    ap.add_argument("--export-scenario", type=int, default=16, help="per scenario, randomised; + 4 nominal")
    ap.add_argument("--records", action="store_true", help="write records/<scenario>.npz (one nominal flight each)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-rollouts", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out = Path(args.out) if args.out else ckpt_path.parent
    out.mkdir(parents=True, exist_ok=True)
    net, params, norm, env_cfg, _, ckpt = ppo.load_checkpoint(ckpt_path)
    env_cfg = dataclasses.replace(env_cfg, rand_action_noise=0.0)  # evaluate the unperturbed policy
    dt = 1 / env_cfg.policy_hz
    noise = [float(v) for v in args.noise.split(",") if v]
    result = {"checkpoint": str(ckpt_path), "iteration": ckpt.get("iteration"), "env_steps": ckpt.get("env_steps"),
              "train_wall_s": ckpt.get("wall_s"), "seed": args.seed, "track": env_cfg.track}  # fmt: skip
    parts = []

    def roll(cfg, key_offs, n_export=0, circuit=None, export_idx=None, noise_levels=(0.0,)):
        """One rollout per noise level with key seed + key_offs[j]; the first exports obs."""
        env = RaceEnv(cfg, circuit=circuit)
        idx = np.arange(min(n_export, cfg.n_envs)) if export_idx is None else export_idx
        run = make_rollout(env, net, params, norm, idx)
        return env, idx, [run(jax.random.key(args.seed + k), sd) for k, sd in zip(key_offs, noise_levels)]

    # Training distribution: randomised (+ action noise), nominal, wide spawns
    cfg = dataclasses.replace(env_cfg, n_envs=args.episodes, p_wide=0.0)
    env, idx, recs = roll(cfg, [0, *range(10, 10 + len(noise))], args.export, noise_levels=(0.0, *noise))
    n_gates = env.n_gates
    rec = recs[0]
    result["randomised"] = metrics(rec, n_gates, dt)
    parts.append(dict(rec=rec, cols=idx, obs=rec["obs"], nominal=False, scenario=-1))
    print(line(f"{env_cfg.track} randomised", result["randomised"]), flush=True)
    if noise:
        result["noise"] = {}
        for sd, r in zip(noise, recs[1:]):
            m = result["noise"][str(sd)] = metrics(r, n_gates, dt)
            print(line(f"{env_cfg.track} randomised, noise {sd}", m), flush=True)
    del recs
    _, idx, (rec,) = roll(env_cfg.nominal(n_envs=args.nominal_episodes, p_wide=0.0), [1], args.export_nominal)
    result["nominal"] = metrics(rec, n_gates, dt)
    parts.append(dict(rec=rec, cols=idx, obs=rec["obs"], nominal=True, scenario=-1))
    print(line(f"{env_cfg.track} nominal", result["nominal"]), flush=True)
    if args.recovery_episodes:
        _, _, (rec,) = roll(dataclasses.replace(env_cfg, n_envs=args.recovery_episodes, p_wide=1.0), [2])
        m = result["recovery"] = recovery_metrics(rec, n_gates, dt)
        print(
            f"  recovery: 1 lap no crash {m['frac_1_lap_no_crash']:.4f}  crash {m['crash_rate']:.4f} "
            f"{m['crash_rate_by_cause']}  before first gate {m['frac_crash_before_first_gate']:.4f}  "
            f"t first gate {m['time_to_first_gate_s']}  fov {m['gate_in_fov_frac']:.4f}",
            flush=True,
        )

    # Named scenarios: one batch holds every scenario, E consecutive environments each
    names = {"all": list(track.SCENARIOS), "none": []}.get(args.scenarios, args.scenarios.split(","))
    if names:
        result["scenarios"] = {name: {} for name in names}
        layouts = [track.circuit_arrays(track.resolve(name)) for name in names]

        def tiled(e):
            return (np.repeat(np.stack([p for p, _ in layouts]), e, 0), np.repeat(np.stack([y for _, y in layouts]), e, 0))

        def blocks(e):
            return [np.arange(b * e, (b + 1) * e) for b in range(len(names))]

        E, En, Er = args.scenario_episodes, args.scenario_nominal_episodes, args.scenario_recovery_episodes
        k_exp = min(args.export_scenario, E)
        exp_idx = np.concatenate([b[:k_exp] for b in blocks(E)])
        cfg = dataclasses.replace(env_cfg, n_envs=len(names) * E, p_wide=0.0)
        _, _, recs = roll(cfg, range(20, 21 + len(noise)), circuit=tiled(E), export_idx=exp_idx,
                          noise_levels=(0.0, *noise))  # fmt: skip
        for b, (name, cols) in enumerate(zip(names, blocks(E))):
            sc = result["scenarios"][name]
            sc["randomised"] = metrics(columns(recs[0], cols), n_gates, dt)
            print(line(f"{name} randomised", sc["randomised"]), flush=True)
            sc["noise"] = {}
            for sd, r in zip(noise, recs[1:]):
                m = sc["noise"][str(sd)] = metrics(columns(r, cols), n_gates, dt)
                print(line(f"{name} randomised, noise {sd}", m), flush=True)
            o = recs[0]["obs"][:, b * k_exp : (b + 1) * k_exp]
            parts.append(dict(rec=recs[0], cols=cols[:k_exp], obs=o, nominal=False, scenario=b))
        k_nom = min(4, En)
        exp_idx = np.concatenate([b[:k_nom] for b in blocks(En)])
        _, _, (rec,) = roll(env_cfg.nominal(n_envs=len(names) * En, p_wide=0.0), [30], circuit=tiled(En),
                            export_idx=exp_idx)  # fmt: skip
        if args.records:
            (out / "records").mkdir(exist_ok=True)
        for b, (name, cols) in enumerate(zip(names, blocks(En))):
            sc = result["scenarios"][name]
            sc["nominal"] = metrics(columns(rec, cols), n_gates, dt)
            print(line(f"{name} nominal", sc["nominal"]), flush=True)
            parts.append(dict(rec=rec, cols=cols[:k_nom], obs=rec["obs"][:, b * k_nom : (b + 1) * k_nom],
                              nominal=True, scenario=b))  # fmt: skip
            if args.records:
                score = rec["passed"][:, cols].sum(0) * 10 + rec["alive"][:, cols].sum(0) / 1e4
                i = int(cols[np.argmax(score)])
                write_record(out / "records" / f"{name}.npz", rec, i, name, dt)
                sc["record"] = {"episode": i, "gates": int(rec["passed"][:, i].sum()),
                                "steps": int(rec["alive"][:, i].sum())}  # fmt: skip
        if Er:
            _, _, (rec,) = roll(dataclasses.replace(env_cfg, n_envs=len(names) * Er, p_wide=1.0), [31],
                                circuit=tiled(Er))  # fmt: skip
            for name, cols in zip(names, blocks(Er)):
                m = result["scenarios"][name]["recovery"] = recovery_metrics(columns(rec, cols), n_gates, dt)
                print(f"{name:>34}: recovery 1 lap no crash {m['frac_1_lap_no_crash']:.4f}  crash {m['crash_rate']:.4f}",
                      flush=True)  # fmt: skip
        result["scenario_names"] = names
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    if not args.no_rollouts:
        np.savez_compressed(out / "rollouts.npz", **export(parts, dt), scenario_names=np.array(names or [""]))
    print(f"wrote {out}/eval.json" + ("" if args.no_rollouts else " and rollouts.npz")
          + (f" and {len(names)} flight records in {out}/records" if args.records and names else ""))  # fmt: skip


if __name__ == "__main__":
    main()
