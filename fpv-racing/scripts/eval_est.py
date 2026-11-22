"""Stage 4 evaluation: estimated-observation policies (and the teacher) through the same harness and seeds as
`scripts/eval_teacher.py`.

    HIP_VISIBLE_DEVICES=0 .venv-rocm/bin/python scripts/eval_est.py --policy runs/percep/policy/train/est_x1/ckpt_latest.pkl \
        --conditions perfect,x1,x2,x4,drop --episodes 2048 --scenario-episodes 2048 --seed 7000 --out <dir>

``--policy``: an est-policy checkpoint (`fpv.ppo_est`), or ``teacher`` (perfect state, the reference),
``teacher_tilt`` (perfect state but the attitude matrix without world yaw, as the estimator provides it),
``teacher_est`` (the teacher net fed the estimated observation, i.e. the warm start before retraining).
Episodes: exactly eval_teacher's: env config of the teacher checkpoint with stick noise off and standard
spawns; random tracks with key ``seed`` (n = --episodes), the six named layouts tiled in ONE batch with key
``seed + 20`` (--scenario-episodes each), each environment's episode drawn by `RaceEnv.reset` with the same
key and batch size, so the teacher row (``--policy teacher``) flies the identical episodes. 30 s each.
Every condition (`fpv.env_est.condition`) reuses the same episodes; only the perception noise differs.

Writes <out>/eval_<condition>.json: per block the eval_teacher metrics (`eval_teacher.metrics`: >= 3 laps,
crash rate by cause, crashes per flight-minute, lap time, speed, ...) + perception: wrong-next-gate share of
steps and events (runs >= 3 steps) per flight-minute, + crash analysis (state at the crash).
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

_had_cuda_env = "CUDA_VISIBLE_DEVICES" in os.environ
from eval_teacher import metrics  # noqa: E402  (its import sets CUDA_VISIBLE_DEVICES=0 by default)

if not _had_cuda_env:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # on ROCm it would conflict with HIP_VISIBLE_DEVICES
from fpv import ppo, ppo_est, track  # noqa: E402
from fpv.env_est import EstEnv, condition  # noqa: E402
from fpv.perception import estimator  # noqa: E402
from fpv.perception.estimator import EST_OBS_DIM  # noqa: E402

TEACHER = "runs/teacher/rt_G_D_noise3/ckpt_latest.pkl"


def make_policy(spec: str):
    """-> (act(full_obs, race_state) -> action, env_cfg of the episodes, est-env base config or None)."""
    t_net, t_params, t_norm, t_env, _, _ = ppo.load_checkpoint(TEACHER)
    if spec in ("teacher", "teacher_tilt", "teacher_est"):
        def act(full, race):
            if spec == "teacher":
                o = full[:, EST_OBS_DIM:]
            elif spec == "teacher_tilt":
                T = estimator.tilt_only_from_quat(race.data.states.quat[:, 0])
                o = full[:, EST_OBS_DIM:].at[:, 3:12].set(T.reshape(-1, 9))
            else:
                o = full[:, :43]
            return ppo.policy_action(t_net, t_params, t_norm, o)

        return act, t_env, None
    net, params, norm, env_cfg, ecfg, _, _ = ppo_est.load_checkpoint(spec)

    def act(full, race):
        return ppo_est.policy_action(net, params, norm, full[:, :EST_OBS_DIM])

    return act, env_cfg, ecfg


def make_rollout(env: EstEnv, act):
    @jax.jit
    def rollout(key):
        state, full = env.reset(key)

        def step(carry, _):
            state, full, alive = carry
            race = state.race
            action = act(full, race)
            s = race.data.states
            est = state.est
            Pc, Py, own, T = estimator.gate_estimates(est, race.track_pos, race.track_yaw, env.ecfg.est)
            gate = jnp.take_along_axis(race.gates_pos, race.target[:, None, None], 1)[:, 0]
            # pose of the gate shown as "next" vs the true next gate (body frame; drone at the H origin)
            from fpv.env import quat_to_mat

            R = quat_to_mat(s.quat[:, 0])
            c_true = jnp.einsum("nji,nj->ni", R, gate - s.pos[:, 0])
            c_est = jnp.einsum("nji,nj->ni", T, jnp.take_along_axis(Pc, est.nxt[:, None, None], 1)[:, 0])
            d_true = jnp.linalg.norm(c_true, axis=-1)
            rec = dict(
                vel=s.vel[:, 0].astype(jnp.float16), ang_vel=s.ang_vel[:, 0].astype(jnp.float16),
                action=action.astype(jnp.float16), target=race.target.astype(jnp.int8), alive=alive,
                wrong=(est.nxt != race.target) & alive,
                pose_wrong=(jnp.linalg.norm(c_est - c_true, axis=-1) > jnp.maximum(0.5, 0.15 * d_true)) & alive,
                next_age=jnp.take_along_axis(est.gage, est.nxt[:, None], 1)[:, 0].astype(jnp.int16),
                locked=est.locked & alive,
                dist_target=d_true.astype(jnp.float16),
            )  # fmt: skip
            state2, full2, reward, done, info = env.step_core_est(state, action, reset_done=False)
            ev = info
            outcome = jnp.where(ev["crash_gate"], 1, jnp.where(ev["ground"], 2, jnp.where(ev["oob"], 3, 0)))
            rec |= dict(passed=ev["passed"] & alive, in_fov=ev["in_fov"], reward=reward.astype(jnp.float16),
                        outcome=outcome.astype(jnp.int8), ended=alive & (ev["terminated"] | ev["truncated"]))  # fmt: skip
            return (state2, full2, alive & ~(ev["terminated"] | ev["truncated"])), rec

        _, rec = jax.lax.scan(step, (state, full, jnp.ones(env.cfg.n_envs, bool)), None, env.cfg.max_steps)
        return rec

    def run(key):
        rec = jax.device_get(rollout(key))
        return {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in rec.items()}

    return run


def runs_of(mask):
    out = []
    for e in range(mask.shape[1]):
        d = np.diff(np.concatenate([[0], mask[:, e].astype(int), [0]]))
        out += [(e, s, t - s) for s, t in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return out


def cause_split(rec, mask, dt, crash_t):
    """Runs (>= 3 steps) of ``mask`` split by how they began: 'start' (before the episode's first true gate pass,
    incl. before the first lock), 'after_pass' (within 1 s after a true pass: advance / re-lock), 'after_unseen'
    (the next gate's information was > 0.2 s old at the start: dropout / occlusion / out of view), 'steady'."""
    passed = rec["passed"]
    first_pass = np.where(passed.any(0), passed.argmax(0), 10**9)
    out = {}
    minutes = rec["alive"].sum() * dt / 60
    groups = {"start": [], "after_pass": [], "after_unseen": [], "steady": []}
    for e, s, n in runs_of(mask):
        if n < 3:
            continue
        if s <= first_pass[e]:
            c = "start"
        elif passed[max(s - 50, 0) : s + 1, e].any():
            c = "after_pass"
        elif rec["next_age"][s, e] > 10:
            c = "after_unseen"
        else:
            c = "steady"
        groups[c].append((e, s, n))
    total = sum(len(g) for g in groups.values())
    for c, g in groups.items():
        lens = np.array([n for _, _, n in g]) * dt
        out[c] = {
            "runs": len(g),
            "frac_of_runs": len(g) / max(total, 1),
            "per_flight_minute": len(g) / max(minutes, 1e-9),
            "frac_of_flight_time": float(lens.sum() / max(rec["alive"].sum() * dt, 1e-9)),
            "len_s": {"p50": float(np.median(lens)), "p90": float(np.percentile(lens, 90))} if len(g) else None,
            # the flight crashed while the wrong belief lasted or within 0.5 s after it ended
            "ending_in_crash_frac": float(np.mean([crash_t[e] <= s + n + 25 for e, s, n in g])) if len(g) else None,
        }
    return out


def percep_metrics(rec, dt):
    alive = rec["alive"]
    ev = [r for r in runs_of(rec["wrong"]) if r[2] >= 3]
    minutes = alive.sum() * dt / 60
    ended = rec["ended"]
    crashed_step = ended & (rec["outcome"] > 0)
    t_idx, e_idx = np.nonzero(crashed_step)
    crash = {
        "n": int(len(e_idx)),
        "wrong_next_at_crash_frac": float(rec["wrong"][t_idx, e_idx].mean()) if len(e_idx) else None,
        # the gate shown as next was not where the true next gate is, at the crash step or within the 0.5 s before
        "next_pose_wrong_within_0.5s_before_crash_frac": float(np.mean([rec["pose_wrong"][max(t - 25, 0) : t + 1, e].any()
                                                                        for t, e in zip(t_idx, e_idx)])) if len(e_idx) else None,
        "next_gate_unseen_over_0.2s_at_crash_frac": float((rec["next_age"][t_idx, e_idx] > 10).mean()) if len(e_idx) else None,
        "dist_to_true_next_gate_m": {"p25": None, "p50": None, "p75": None},
        "time_s": None,
        "true_next_gate_in_fov_frac": float(rec["in_fov"][t_idx, e_idx].mean()) if len(e_idx) else None,
    }
    if len(e_idx):
        d = rec["dist_target"][t_idx, e_idx]
        crash["dist_to_true_next_gate_m"] = {f"p{p}": round(float(np.percentile(d, p)), 3) for p in (25, 50, 75)}
        crash["time_s"] = {f"p{p}": round(float(np.percentile(t_idx * dt, p)), 2) for p in (25, 50, 75)}
        crash["frac_within_0.5m_of_next_gate"] = float((d < 0.5).mean())
        crash["frac_in_first_2s"] = float((t_idx * dt < 2).mean())
    pr = [r for r in runs_of(rec["pose_wrong"]) if r[2] >= 3]
    ends = alive.sum(0)
    crash_t = np.where((ended & (rec["outcome"] > 0)).any(0), ends, 10**9)
    # crash causes: wrong gate belief (the gate shown as next was not where the true next gate is, at the crash
    # step or within the 0.5 s before) vs ordinary gate hit vs other (ground / out of bounds), belief right
    causes = None
    if len(e_idx):
        wb = np.array([rec["pose_wrong"][max(t - 25, 0) : t + 1, e].any() for t, e in zip(t_idx, e_idx)])
        oc = rec["outcome"][t_idx, e_idx]
        n_ep = alive.shape[1]
        causes = {
            "wrong_belief": int(wb.sum()), "gate_hit_belief_right": int(((oc == 1) & ~wb).sum()),
            "other_belief_right": int(((oc > 1) & ~wb).sum()),
            "wrong_belief_by_outcome": {"gate": int(((oc == 1) & wb).sum()), "ground": int(((oc == 2) & wb).sum()),
                                        "oob": int(((oc == 3) & wb).sum())},
            "per_episode": {"wrong_belief": float(wb.sum() / n_ep), "gate_hit_belief_right": float(((oc == 1) & ~wb).sum() / n_ep),
                            "other_belief_right": float(((oc > 1) & ~wb).sum() / n_ep)},
        }  # fmt: skip
    crash["causes"] = causes
    # chance level of the "wrong belief" label: share of ALL alive steps (and of steps < 1 m from the true next
    # gate, where crashes cluster) with the next-gate pose wrong at the step or in the 0.5 s before
    pw_c = np.cumsum(np.concatenate([np.zeros((1, alive.shape[1])), rec["pose_wrong"]], 0), 0)
    win = (pw_c[26:] - pw_c[:-26]) > 0  # window [t - 25, t] for t >= 25
    a25, near = alive[25:], alive[25:] & (rec["dist_target"][25:] < 1.0)
    crash["wrong_belief_chance_level"] = {"all_steps": float(win[a25].mean()), "within_1m_of_next_gate": float(win[near].mean())}
    first_lock = np.where(rec["locked"].any(0), rec["locked"].argmax(0), -1)
    fl = first_lock[first_lock >= 0] * dt
    return {
        "wrong_next_step_frac": float(rec["wrong"][alive].mean()),
        "wrong_next_events": len(ev),
        "wrong_next_events_per_flight_minute": len(ev) / max(minutes, 1e-9),
        "wrong_next_event_len_s_median": float(np.median([r[2] for r in ev]) * dt) if ev else None,
        "next_pose_wrong_step_frac": float(rec["pose_wrong"][alive].mean()),
        "next_pose_wrong_runs_per_flight_minute": len(pr) / max(minutes, 1e-9),
        "next_pose_wrong_run_s": {"p50": float(np.median([r[2] for r in pr]) * dt), "p90": float(np.percentile([r[2] for r in pr], 90) * dt)} if pr else None,
        "next_pose_wrong_runs_ending_in_crash_frac": float(np.mean([crash_t[e] <= s + n + 25 for e, s, n in pr])) if pr else None,
        "first_lock_s": {"p50": float(np.median(fl)), "p90": float(np.percentile(fl, 90)), "never_frac": float((first_lock < 0).mean())} if len(fl) else None,
        "wrong_next_by_cause": cause_split(rec, rec["wrong"], dt, crash_t),
        "next_pose_wrong_by_cause": cause_split(rec, rec["pose_wrong"], dt, crash_t),
        "crash_analysis": crash,
    }


def columns(rec, idx):
    return {k: v[:, idx] for k, v in rec.items()}


def summary_line(name, m, p):
    lt = m["lap_time_s"]["flying_mean"]
    return (f"{name:>16}: 3 laps {m['frac_3_laps']:.4f} crash {m['crash_rate']:.4f} /min {m['crashes_per_flight_minute']:.3f} "
            f"lap {lt and round(lt, 2)} v {m['speed_m_s']['mean']:.2f} wrong {p['wrong_next_step_frac']:.4f} "
            f"ev/min {p['wrong_next_events_per_flight_minute']:.3f} pose-wrong {p['next_pose_wrong_step_frac']:.4f}")  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--conditions", default="perfect,x1,x2,x4,drop")
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--scenario-episodes", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--known-start", action="store_true", help="contract addition: first gate index given")
    ap.add_argument("--est", nargs="*", default=[], metavar="field=value",
                    help="EstConfig overrides on top of the policy checkpoint's estimator config (teacher: defaults)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    act, env_cfg, p_ecfg = make_policy(args.policy)
    from check_estimator_sweep import parse_variant

    # the estimator the policy was trained with (fields unknown to its version take the defaults = estimator v4)
    est_cfg = dataclasses.replace(p_ecfg.est if p_ecfg is not None else condition("x1").est,
                                  **parse_variant(",".join(args.est)))  # fmt: skip
    env_cfg = dataclasses.replace(env_cfg, rand_action_noise=0.0, p_wide=0.0)
    dt = 1 / env_cfg.policy_hz
    names = {"all": list(track.SCENARIOS), "none": []}.get(args.scenarios, [s for s in args.scenarios.split(",") if s])
    for cond in args.conditions.split(","):
        t0 = time.time()
        ecfg = dataclasses.replace(condition(cond), known_start=args.known_start, est=est_cfg)
        res = {"policy": args.policy, "condition": cond, "seed": args.seed, "episodes": args.episodes,
               "scenario_episodes": args.scenario_episodes, "est_env": dataclasses.asdict(ecfg)}  # fmt: skip
        env = EstEnv(dataclasses.replace(env_cfg, n_envs=args.episodes), ecfg)
        rec = make_rollout(env, act)(jax.random.key(args.seed))
        res["random"] = metrics(rec, 6, dt) | {"perception": percep_metrics(rec, dt)}
        print(f"[{cond}] " + summary_line("random", res["random"], res["random"]["perception"]), flush=True)
        del rec
        if names:
            E = args.scenario_episodes
            layouts = [track.circuit_arrays(track.resolve(n)) for n in names]
            tiled = (np.repeat(np.stack([p for p, _ in layouts]), E, 0), np.repeat(np.stack([y for _, y in layouts]), E, 0))
            env = EstEnv(dataclasses.replace(env_cfg, n_envs=len(names) * E), ecfg, circuit=tiled)
            rec = make_rollout(env, act)(jax.random.key(args.seed + 20))
            res["scenarios"] = {}
            for b, name in enumerate(names):
                r = columns(rec, np.arange(b * E, (b + 1) * E))
                res["scenarios"][name] = metrics(r, 6, dt) | {"perception": percep_metrics(r, dt)}
                print(f"[{cond}] " + summary_line(name, res["scenarios"][name], res["scenarios"][name]["perception"]),
                      flush=True)  # fmt: skip
            del rec
        res["wall_s"] = round(time.time() - t0, 1)
        tag = cond + ("_known" if args.known_start else "")
        (out / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
        print(f"[{tag}] wrote {out}/eval_{tag}.json ({res['wall_s']} s)", flush=True)


if __name__ == "__main__":
    main()
