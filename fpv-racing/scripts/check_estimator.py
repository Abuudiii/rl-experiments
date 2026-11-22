"""Stage 2/3 checks: simulated IMU and estimator against simulator truth on teacher flights.

    CUDA_VISIBLE_DEVICES=1 .venv-sim/bin/python scripts/check_estimator.py --out runs/percep/policy/checks \
        --conditions perfect,x05,x1,x2,x4,drop --episodes 512 --steps 1500

The teacher flies from its TRUE observation (perfect state, its evaluation setting: random tracks,
randomised gates and physics, standard spawn; ``--wide`` = wide off-course spawns); the perception
pipeline runs alongside on the same flights (open loop: its estimates do not steer). Per condition
(detection + IMU noise, `fpv.env_est.condition`) it reports:
  IMU       gyro / accelerometer error against the error-free step means; with --imu-checks also the
            physics checks (gyro integration reproduces the attitude change; accelerometer against
            Crazyflow's CoM force; tilt from the accelerometer in near-hover).
  tilt      angle between estimated and true up vector [deg]
  velocity  |v_est - v_true| body frame [m/s]
  rates     |w_est - w_true| [rad/s]
  next gate centre error [m] of the gate the estimator calls next, when that is the true next gate,
            by distance bin and by source (seen this frame / remembered < 2 s / placement prediction)
  wrong next-gate: share of flight steps with est != true next gate; events = runs of >= 3 wrong steps
            (one-step lags at the gate plane excluded); recovery time of each event (runs still open at the
            episode end are reported as unrecovered)
  lock      time to first lock
Writes <out>/check_<condition>[_wide].json.
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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fpv import ppo  # noqa: E402
from fpv.env import quat_to_mat  # noqa: E402
from fpv.env_est import EstEnv, condition  # noqa: E402
from fpv.perception import detnoise, estimator, imu, labels  # noqa: E402
from fpv.perception.estimator import EST_OBS_DIM  # noqa: E402

TEACHER = "runs/teacher/rt_G_D_noise3/ckpt_latest.pkl"


def make_rollout(env: EstEnv, net, params, norm, steps: int):
    @jax.jit
    def rollout(key):
        state, full = env.reset(key)

        def step(carry, _):
            state, full, alive = carry
            race, est = state.race, state.est
            s = race.data.states
            R = quat_to_mat(s.quat[:, 0])
            up_true = R[:, 2, :]
            Pc, Py, own, T = estimator.gate_estimates(est, race.track_pos, race.track_yaw, env.ecfg.est)
            v_true_b = jnp.einsum("nji,nj->ni", R, s.vel[:, 0])
            v_est_b = estimator.velocity_body(est)

            def gate_err(k):
                ie = (est.nxt + k) % 6
                it = (race.target + k) % 6
                c_est = jnp.einsum("nji,nj->ni", T, jnp.take_along_axis(Pc, ie[:, None, None], 1)[:, 0])
                g = jnp.take_along_axis(race.gates_pos, it[:, None, None], 1)[:, 0]
                c_true = jnp.einsum("nji,nj->ni", R, g - s.pos[:, 0])
                src = jnp.where(jnp.take_along_axis(est.gage, ie[:, None], 1)[:, 0] == 0, 0,
                                jnp.where(jnp.take_along_axis(own, ie[:, None], 1)[:, 0], 1, 2))  # fmt: skip
                return jnp.linalg.norm(c_est - c_true, axis=-1), jnp.linalg.norm(c_true, axis=-1), src

            ng_err, ng_dist, ng_src = gate_err(0)
            # identity: median over the 6 list gates of |estimated - true| position (H frame); > 1 m = wrong
            true_b = jnp.einsum("nji,ngj->ngi", R, race.gates_pos - s.pos[:, :1])
            true_H = jnp.einsum("nij,ngj->ngi", T, true_b)
            ident_err = jnp.median(jnp.linalg.norm(Pc - true_H, axis=-1), axis=1)
            an_err, an_dist, an_src = gate_err(1)
            rec = dict(
                alive=alive, locked=est.locked, nxt=est.nxt, target=race.target,
                tilt_err=jnp.degrees(jnp.arccos(jnp.clip((up_true * est.up).sum(-1), -1, 1))),
                vel_err=jnp.linalg.norm(v_est_b - v_true_b, axis=-1), speed=jnp.linalg.norm(v_true_b, axis=-1),
                vel_err_xyz=v_est_b - v_true_b,
                rate_err=jnp.linalg.norm(est.rate - s.ang_vel[:, 0], axis=-1),
                ng_err=ng_err, ng_dist=ng_dist, ng_src=ng_src, an_err=an_err, an_dist=an_dist, an_src=an_src,
                ident_err=ident_err,
            )  # fmt: skip
            action = ppo.policy_action(net, params, norm, full[:, EST_OBS_DIM:])
            state2, full2, reward, done, info = env.step_core_est(state, action, reset_done=False)
            s1 = state2.race.data.states
            g_t, a_t = imu.true_imu(s.quat[:, 0], s1.quat[:, 0], s.vel[:, 0], s1.vel[:, 0], s.ang_vel[:, 0],
                                    s1.ang_vel[:, 0], env.dt)  # fmt: skip
            # physics checks: gyro integration vs attitude change; accel vs CoM force at the step end
            from fpv.perception.geometry import _rodrigues

            R1 = quat_to_mat(s1.quat[:, 0])
            R_pred = R @ _rodrigues(g_t * env.dt)
            R_pred0 = R @ _rodrigues(s.ang_vel[:, 0] * env.dt)

            def ang(A, B):
                # chordal form (arccos of the trace loses precision near 0 deg in float32)
                fro = jnp.sqrt(((A - B) ** 2).sum((-1, -2)))
                return jnp.degrees(2 * jnp.arcsin(jnp.clip(fro / (2 * jnp.sqrt(2.0)), 0, 1)))

            # independent accelerometer reference: Crazyflow's force model (first_principles: rotor thrust
            # k0 + k1 rpm + k2 rpm^2 along body z + body drag D v_b) / mass, trapezoid over the step
            prm = state.race.data.params
            mass = prm.mass[:, 0, 0]

            def f_model(st_, Rm):
                rpm = st_.rotor_vel[:, 0]
                k = prm.rpm2thrust[:, 0]
                thrust = (k[..., 0] + k[..., 1] * rpm + k[..., 2] * rpm**2).sum(-1)
                v_b = jnp.einsum("nji,nj->ni", Rm, st_.vel[:, 0])
                drag = jnp.einsum("nij,nj->ni", prm.drag_matrix[:, 0], v_b)
                return (drag + jnp.stack([jnp.zeros_like(thrust), jnp.zeros_like(thrust), thrust], -1)) / mass[:, None]

            f_mod = 0.5 * (f_model(s, R) + f_model(s1, R1))
            rec |= dict(
                gyro_err=info["gyro"] - g_t, acc_err=info["acc"] - a_t, acc_true=a_t, gyro_true=g_t,
                att_err_mean_rate=ang(R_pred, R1), att_err_start_rate=ang(R_pred0, R1), acc_model=f_mod,
                passed=info["passed"] & alive, relock=info["percep"]["est"]["relock"] & alive,
                geo_jump=info["percep"]["est"]["geo_jump"] & alive, est_passed=info["percep"]["est"]["passed"] & alive,
                up_true=up_true,
            )  # fmt: skip
            # detection-noise statistics (slot -> true gate) and the geometry's range per slot
            r2 = state2.race
            s2 = r2.data.states
            uv, _, vis, gdist, _ = detnoise.project_gates(s2.pos[:, 0], s2.quat[:, 0], r2.gates_pos, r2.gates_yaw)
            uv = labels.canonical_order(uv)
            gid = info["percep"]["det"]["gate"]
            gi = jnp.maximum(gid, 0)
            true_c = jnp.take_along_axis(uv, gi[..., None, None], 1)
            dg = info["percep"]["est"]["det"]
            rec |= dict(
                slot_err=jnp.linalg.norm(info["percep"]["corners"] - true_c, axis=-1), slot_gid=gid,
                slot_conf=info["percep"]["conf"], slot_tdist=jnp.take_along_axis(gdist, gi, 1),
                slot_dist=dg["dist"], slot_dist_pnp=dg["dist_pnp"], slot_dist_size=dg["dist_size"],
                vis=vis, gdist=gdist, target2=r2.target, alive_next=alive & ~(info["terminated"] | info["truncated"]),
            )  # fmt: skip
            alive2 = alive & ~(info["terminated"] | info["truncated"])
            return (state2, full2, alive2), rec

        _, rec = jax.lax.scan(step, (state, full, jnp.ones(env.cfg.n_envs, bool)), None, steps)
        return rec

    return lambda key: jax.device_get(rollout(key))


def runs(mask):
    """Runs of True along axis 0 per column: list of (column, start, length, closed)."""
    out = []
    T = mask.shape[0]
    for e in range(mask.shape[1]):
        m = mask[:, e]
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        for s, t in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            out.append((e, s, t - s, t < T and True))
    return out


def q(x, ps=(50, 90, 99)):
    x = np.asarray(x)
    if x.size == 0:
        return None
    return {"mean": round(float(x.mean()), 4), **{f"p{p}": round(float(np.percentile(x, p)), 4) for p in ps}, "n": int(x.size)}


def summarise(rec, dt):
    alive = rec["alive"]
    lk = alive & rec["locked"]
    ok = lk & (rec["nxt"] == rec["target"])
    res = {"steps_alive": int(alive.sum()), "frac_locked": float(lk.sum() / max(alive.sum(), 1))}
    res["identity_wrong_step_frac"] = float((rec["ident_err"][lk] > 1.0).mean())
    res["identity_wrong_at_first_step_frac"] = float((rec["ident_err"][0] > 1.0).mean())
    # what the policy needs: is the gate it is shown as "next" (and "after next") where the TRUE next gate is?
    # (index-independent: a wrong identity with the right gate in front still shows the right next gate)
    tol = np.maximum(0.5, 0.15 * rec["ng_dist"])
    res["next_pose_wrong_frac"] = float((rec["ng_err"] > tol)[lk].mean())
    res["after_next_pose_wrong_frac"] = float((rec["an_err"] > np.maximum(0.7, 0.15 * rec["an_dist"]))[lk].mean())
    # where the wrong next-gate poses come from: phase of the flight at each wrong step
    pw = (rec["ng_err"] > tol) & lk
    T_, N_ = alive.shape
    tt = np.arange(T_)[:, None]

    def since(ev):  # steps since the last event (large if none yet)
        last = np.where(ev, tt, -10**6)
        return tt - np.maximum.accumulate(last, axis=0)

    before_first_pass = np.cumsum(rec["passed"], 0) - rec["passed"] == 0  # no true pass completed yet
    phase = np.where(before_first_pass, 0, np.where(since(rec["relock"]) < 50, 1, np.where(since(rec["passed"]) < 50, 2, 3)))
    names = ["start_before_first_pass", "within_1s_after_relock", "within_1s_after_a_pass", "steady"]
    res["next_pose_wrong_by_phase"] = {
        nm: {"share_of_wrong_steps": round(float((pw & (phase == k)).sum() / max(pw.sum(), 1)), 4),
             "wrong_frac_in_phase": round(float(pw[(phase == k) & lk].mean()), 4) if ((phase == k) & lk).any() else None,
             "steps_in_phase": int(((phase == k) & lk).sum())}
        for k, nm in enumerate(names)}  # fmt: skip
    pr = [r for r in runs(pw) if r[2] >= 3]
    sp = since(rec["passed"])
    # steps until the next true pass (large if none)
    nxt_ev = np.where(rec["passed"][::-1], tt, -10**6)
    until = (tt - np.maximum.accumulate(nxt_ev, axis=0))[::-1]
    res["next_pose_wrong_runs"] = {"n": len(pr), "per_flight_minute": len(pr) / max(alive.sum() * dt / 60, 1e-9),
                                   "duration_s": q([r[2] * dt for r in pr]),
                                   "share_starting_before_first_pass": round(float(np.mean([before_first_pass[r[1], r[0]] for r in pr])), 4) if pr else None,
                                   # onset relative to the true gate passes: right after a pass (switch to the new
                                   # next gate went wrong) or just before one (estimated pass early / gate lost up close)
                                   "share_starting_within_0.2s_after_a_pass": round(float(np.mean([sp[r[1], r[0]] <= 10 for r in pr])), 4) if pr else None,
                                   "share_starting_within_0.2s_before_a_pass": round(float(np.mean([until[r[1], r[0]] <= 10 for r in pr])), 4) if pr else None}  # fmt: skip
    post = [r for r in pr if sp[r[1], r[0]] <= 10]
    if post:
        nx = np.array([rec["nxt"][r[1], r[0]] for r in post])
        tg = np.array([rec["target"][r[1], r[0]] for r in post])
        res["next_pose_wrong_runs"]["after_pass_index"] = {
            "est_index_right_pose_wrong": round(float(np.mean(nx == tg)), 4),
            "est_still_on_passed_gate": round(float(np.mean(nx == (tg - 1) % 6)), 4),
            "est_other_index": round(float(np.mean((nx != tg) & (nx != (tg - 1) % 6))), 4)}  # fmt: skip
    first_lock = np.argmax(rec["locked"], 0)
    res["time_to_first_lock_s"] = q(first_lock[rec["locked"].any(0)] * dt)
    res["never_locked_episodes"] = int((~rec["locked"].any(0)).sum())
    res["tilt_err_deg"] = q(rec["tilt_err"][lk])
    res["vel_err_m_s"] = q(rec["vel_err"][lk])
    res["vel_err_rms_xyz_body"] = [round(float(v), 4) for v in np.sqrt((rec["vel_err_xyz"][lk] ** 2).mean(0))]
    res["speed_true_mean"] = float(rec["speed"][alive].mean())
    res["rate_err_rad_s"] = q(rec["rate_err"][lk])
    bins = [0, 1, 2, 4, 8, 100]
    ng = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (rec["ng_dist"] >= lo) & (rec["ng_dist"] < hi)
        ng[f"{lo}-{hi}m"] = q(rec["ng_err"][m])
    res["next_gate_err_m_by_dist"] = ng
    res["next_gate_err_m_by_source"] = {name: q(rec["ng_err"][ok & (rec["ng_src"] == k)])
                                        for k, name in enumerate(["seen_now", "remembered", "placement"])}  # fmt: skip
    res["next_gate_err_m"] = q(rec["ng_err"][ok])
    ok2 = ok & (rec["ng_src"] >= 0)
    res["after_next_err_m"] = q(rec["an_err"][ok2])
    res["after_next_err_m_by_source"] = {name: q(rec["an_err"][ok2 & (rec["an_src"] == k)])
                                         for k, name in enumerate(["seen_now", "remembered", "placement"])}  # fmt: skip
    wrong = lk & (rec["nxt"] != rec["target"])
    res["wrong_next_frac_of_locked_steps"] = float(wrong.sum() / max(lk.sum(), 1))
    ev = [r for r in runs(wrong) if r[2] >= 3]
    T = alive.shape[0]
    ends = alive.sum(0)
    closed = [r for r in ev if r[1] + r[2] < ends[r[0]]]
    res["wrong_next_events"] = len(ev)
    cause = {"initial_lock": 0, "relock": 0, "geo_jump": 0, "missed_pass": 0, "spurious_pass": 0, "late_pass": 0,
             "other": 0}  # fmt: skip
    for e, s, _, _ in ev:
        if rec["ident_err"][s, e] > 1.0:
            cause["identity_wrong_at_start"] = cause.get("identity_wrong_at_start", 0) + 1
        if s == 0:
            cause["initial_lock"] += 1
            continue
        p = s - 1
        if rec["relock"][p, e]:
            cause["relock"] += 1
        elif rec["geo_jump"][p, e]:
            cause["geo_jump"] += 1
        elif rec["passed"][p, e] and not rec["est_passed"][p, e]:
            # a pass detected within 3 frames after the true one is a lag, not a miss
            late = rec["est_passed"][s : s + 3, e].any()
            cause["late_pass" if late else "missed_pass"] += 1
        elif rec["est_passed"][p, e] and not rec["passed"][p, e]:
            cause["spurious_pass"] += 1
        else:
            cause["other"] += 1
    res["wrong_next_event_causes"] = cause
    res["wrong_next_events_per_flight_minute"] = len(ev) / max(alive.sum() * dt / 60, 1e-9)
    res["wrong_next_events_per_gate_pass"] = len(ev) / max(int(rec["passed"].sum()), 1)
    res["wrong_next_unrecovered"] = len(ev) - len(closed)
    res["wrong_next_recovery_s"] = q([r[2] * dt for r in closed])
    res["gate_passes_true"] = int(rec["passed"].sum())
    res["est_passes"] = int(rec["est_passed"].sum())
    res["relocks"] = int(rec["relock"].sum())
    res["geo_jumps"] = int(rec["geo_jump"].sum())
    a = alive
    res["imu"] = {
        "gyro_err_rms_rad_s": [round(float(v), 5) for v in np.sqrt((rec["gyro_err"][a] ** 2).mean(0))],
        "acc_err_rms_m_s2": [round(float(v), 4) for v in np.sqrt((rec["acc_err"][a] ** 2).mean(0))],
    }
    return res


def det_stats(rec):
    """The noise model's own statistics, in the detector worker's format (for calibration)."""
    a = rec["alive_next"]
    gid, conf = rec["slot_gid"], rec["slot_conf"]
    real = (gid >= 0) & (conf > 0) & a[..., None]
    bins = [0, 2, 4, 8, 12.01]
    out = {"corner_err_px_by_dist": {}, "miss_rate_by_dist": {}, "range_rel_err_by_dist": {}}
    G = rec["vis"].shape[-1]
    detected = (gid[..., None] == np.arange(G)) & (conf[..., None] > 0)  # (T, N, K, G)
    det_g = detected.any(-2)  # (T, N, G)
    vis = rec["vis"] & a[..., None]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = real & (rec["slot_tdist"] >= lo) & (rec["slot_tdist"] < hi)
        e = rec["slot_err"][m].reshape(-1)
        key = f"{lo}-{int(hi)}m"
        out["corner_err_px_by_dist"][key] = q(e, (50, 90, 99))
        vm = vis & (rec["gdist"] >= lo) & (rec["gdist"] < hi)
        out["miss_rate_by_dist"][key] = float((~det_g[vm]).mean()) if vm.any() else None
        rel = {}
        for name in ("slot_dist", "slot_dist_pnp", "slot_dist_size"):
            r = (rec[name][m] - rec["slot_tdist"][m]) / rec["slot_tdist"][m]
            rel[name.replace("slot_", "")] = {"bias": round(float(r.mean()), 4), "std": round(float(r.std()), 4)} if r.size else None
        out["range_rel_err_by_dist"][key] = rel
    tgt = rec["target2"]
    nv = np.take_along_axis(vis, tgt[..., None], -1)[..., 0]
    nd = np.take_along_axis(det_g, tgt[..., None], -1)[..., 0]
    miss = nv & ~nd
    out["next_gate_visible_frac"] = float(nv[a].mean())
    out["next_gate_miss_rate_when_visible"] = float(miss[nv].mean())
    both = nv[1:] & nv[:-1] & (tgt[1:] == tgt[:-1])
    out["next_gate_p_miss_given_prev_miss"] = float(miss[1:][both & miss[:-1]].mean()) if (both & miss[:-1]).any() else None
    out["next_gate_p_miss_given_prev_hit"] = float(miss[1:][both & ~miss[:-1]].mean())
    out["all_gates_miss_rate"] = float((~det_g[vis]).mean())
    out["false_per_frame"] = float(((gid < 0) & (conf > 0))[a].sum() / max(a.sum(), 1))
    return out


def imu_physics(rec):
    a = rec["alive"]
    out = {
        "attitude_change_err_deg_using_mean_rate": q(rec["att_err_mean_rate"][a]),
        "attitude_change_err_deg_using_start_rate": q(rec["att_err_start_rate"][a]),
        "rate_norm_rad_s": q(np.linalg.norm(rec["gyro_true"][a], axis=-1)),
    }
    # accel (step mean from dv/dt) against the force model (trapezoid of the step's start and end states)
    d1 = rec["acc_model"][a] - rec["acc_true"][a]
    out["acc_vs_force_model_err_rms_xyz"] = [round(float(v), 4) for v in np.sqrt((d1**2).mean(0))]
    out["acc_vs_force_model_err_mean_xyz"] = [round(float(v), 4) for v in d1.mean(0)]
    out["acc_vs_force_model_corr_xyz"] = [round(float(np.corrcoef(rec["acc_model"][a][:, i], rec["acc_true"][a][:, i])[0, 1]), 4)
                                          for i in range(3)]  # fmt: skip
    out["acc_true_rms_xyz"] = [round(float(v), 4) for v in np.sqrt((rec["acc_true"][a] ** 2).mean(0))]
    # tilt from the accelerometer where |a| ~ g and body rates small (near-hover)
    acc = rec["acc_true"][a]
    calm = (np.abs(np.linalg.norm(acc, axis=-1) - 9.81) < 0.3) & (np.linalg.norm(rec["gyro_true"][a], axis=-1) < 0.3)
    up_acc = acc / np.linalg.norm(acc, axis=-1, keepdims=True)
    err = np.degrees(np.arccos(np.clip((up_acc * rec["up_true"][a]).sum(-1), -1, 1)))
    out["near_hover_tilt_from_acc_err_deg"] = q(err[calm])
    out["all_steps_tilt_from_acc_err_deg"] = q(err)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--conditions", default="perfect,x1")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--known-start", action="store_true", help="contract addition: first gate index given")
    ap.add_argument("--out", default="runs/percep/policy/checks")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net, params, norm, env_cfg, _, _ = ppo.load_checkpoint(args.teacher)
    env_cfg = dataclasses.replace(env_cfg, n_envs=args.episodes, p_wide=1.0 if args.wide else 0.0, rand_action_noise=0.0)
    dt = 1 / env_cfg.policy_hz
    for name in args.conditions.split(","):
        t0 = time.time()
        env = EstEnv(env_cfg, dataclasses.replace(condition(name), known_start=args.known_start))
        rec = make_rollout(env, net, params, norm, args.steps)(jax.random.key(args.seed))
        res = {"condition": name, "wide_spawn": args.wide, "episodes": args.episodes, "steps": args.steps,
               "seed": args.seed, "known_start": args.known_start, "summary": summarise(rec, dt),
               "wall_s": round(time.time() - t0, 1)}  # fmt: skip
        if name in ("perfect", "x1"):
            res["imu_physics"] = imu_physics(rec)
        res["det_stats"] = det_stats(rec)
        print("  det stats", json.dumps(res["det_stats"]), flush=True)
        print("  next-pose wrong by phase", json.dumps(res["summary"]["next_pose_wrong_by_phase"]),
              "runs", json.dumps(res["summary"]["next_pose_wrong_runs"]), flush=True)  # fmt: skip
        suffix = ("_wide" if args.wide else "") + ("_known" if args.known_start else "")
        (out / f"check_{name}{suffix}.json").write_text(json.dumps(res, indent=1))
        s = res["summary"]
        print(f"== {name}{suffix}: {res['wall_s']} s  locked {s["frac_locked"]:.4f} ident-wrong {s["identity_wrong_step_frac"]:.4f} at-start {s["identity_wrong_at_first_step_frac"]:.4f}  tilt {s['tilt_err_deg']}\n"
              f"  vel {s['vel_err_m_s']} (speed {s['speed_true_mean']:.2f})\n  rate {s['rate_err_rad_s']}\n"
              f"  next gate {s['next_gate_err_m']}\n  by dist {s['next_gate_err_m_by_dist']}\n"
              f"  by source {s['next_gate_err_m_by_source']}\n  after-next {s['after_next_err_m']}\n"
              f"  next pose wrong {s['next_pose_wrong_frac']:.4f} after-next pose wrong {s['after_next_pose_wrong_frac']:.4f}\n"
              f"  wrong-next frac {s['wrong_next_frac_of_locked_steps']:.4f} events {s['wrong_next_events']} "
              f"/min {s['wrong_next_events_per_flight_minute']:.3f} unrecovered {s['wrong_next_unrecovered']} "
              f"recovery {s['wrong_next_recovery_s']}\n  causes {s['wrong_next_event_causes']}\n"
              f"  passes true {s['gate_passes_true']} est {s['est_passes']} "
              f"relocks {s['relocks']} geo_jumps {s['geo_jumps']} first lock {s['time_to_first_lock_s']} "
              f"never {s['never_locked_episodes']}\n  imu {s['imu']}", flush=True)  # fmt: skip
        if "imu_physics" in res:
            print("  imu physics", json.dumps(res["imu_physics"]), flush=True)


if __name__ == "__main__":
    main()
