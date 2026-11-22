"""Stage 5: closed loop of the full perception chain from rendered camera frames (no green gate).

Every 50 Hz control step, for every parallel drone: Crazyflow steps -> the fast synthetic renderer draws the FPV
frame of the new state (`fpv.render_synth` ray caster via `labels.make_renderer`, 320x180, supersample 2, one random
look per flight from `labels.scene_pool`, i.e. `SynthRenderer`'s randomisation with Unreal-like share 0.7, next-gate
highlight OFF) -> detector E (`load_detector`, snapping on) -> confidences below ``--thr`` zeroed (and the estimator's
conf_min set to ``--thr``) -> estimator (config from the policy checkpoint, v5) with the simulated IMU at x1 and the
course list -> policy -> sticks. Latency convention = stage 4 (`fpv.env_est.EstEnv.step_core_est`, p_latency 0 at
evaluation): the frame of the state AFTER physics step t is processed at once and its estimate feeds the action of
step t + 1 (no extra frame of latency, compute time not simulated).

``--source true`` replaces render + detector by the stage-4 detection model (`detnoise.detect`) at the condition's
noise scale (``--cond perfect``: exactly the visible true corners, conf 1, the stage-4 occlusion rule), through the
same code path otherwise (plumbing check (a); also gives the stage-4 rows on any subset of flights).

Episodes = stage 4's (`scripts/eval_est.py`): the stage-4 batch (random tracks key ``--seed``, batch ``--batch``;
the six layouts tiled in one batch of 6 x ``--batch`` with key ``seed + 20``) is reset in full with the stage-4
EstEnv (episode draws depend on the batch size) and the flights ``--idx`` (of the random batch, or of the layout's
block) are sliced out, including their IMU biases. So ``--idx 0:2048 --batch 2048`` flies exactly stage 4's flights.

Modes:
  run      one block (random or one layout) -> <out>/<block>.json (+ per-step records npz in --rec-dir)
  student  the camera students' evaluation episodes (FlightBatch seed 300050 continuing stream, 256 slots,
           standard spawns, ``--resets`` = which reset of the stream, ``--steps`` 1000 = 20 s for B, 1500 for G)
  bench    throughput of the whole chain (batch 1 and batched)

    HIP_VISIBLE_DEVICES=0 .venv-rocm/bin/python scripts/eval_stage5.py run --policy runs/percep/policy/train/est_rand_v5/ckpt_latest.pkl \
        --block random --idx 0:2048 --out runs/percep/stage5/eval/main
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

from eval_est import TEACHER, make_policy, percep_metrics  # noqa: E402  (eval_est handles CUDA_VISIBLE_DEVICES)
from eval_teacher import metrics  # noqa: E402
from fpv import track  # noqa: E402
from fpv.env import quat_to_mat  # noqa: E402
from fpv.env_est import EstEnv, EstEnvState, condition  # noqa: E402
from fpv.perception import detnoise, estimator, imu, labels  # noqa: E402
from fpv.perception.detector import load_detector  # noqa: E402

DET = "runs/percep/detector/E_amd/ckpt_best.pkl"
MATCH_PX = 8.0  # a detection slot belongs to a true gate if its mean corner distance (cyclic order) is < 8 px


class Stage5Env(EstEnv):
    """EstEnv whose detections come from rendered frames + the detector (source "detector") or from the stage-4
    detection model (source "true"); per-step detection diagnostics against the true projected gates."""

    def __init__(self, cfg, ecfg, source="detector", det_path=DET, thr=0.3, scene=None, chunk=256, circuit=None,
                 sel=(), keep_frames=False):  # fmt: skip
        super().__init__(cfg, ecfg, circuit=circuit)
        self.source, self.thr, self.chunk, self.sel = source, thr, chunk, tuple(sel)
        self.keep_frames = keep_frames and len(self.sel) > 0
        if source == "detector":
            self.det = load_detector(det_path)
            self.render = labels.make_renderer()
            self.scene = {k: jnp.asarray(v) for k, v in scene.items()}

    # render + detect in chunks (memory), inside the jitted rollout: a Python loop over the chunks (lax.map over
    # the chunks failed to compile on ROCm XLA: "BufferAllocation::Slice not assigned"); un-jitted renderer and
    # detector functions.
    def _frames_detect(self, pos, quat, gp, gy, key):
        from fpv.perception.detector import decode

        n = pos.shape[0]
        c = min(self.chunk, n)
        assert n % c == 0, (n, c)
        keep = self.keep_frames
        render = getattr(self.render, "__wrapped__", self.render)
        det = self.det

        def f(args):
            p, q, g, y, sc, k = args
            fr, _ = render(p, q, g, y, sc, k)
            out = decode(det.cfg, det.net.apply(det.params, fr), snap=det.snap)
            return ((fr,) if keep else ()) + (out["corners"], out["conf"])

        def resh(x):
            return x.reshape(n // c, c, *x.shape[1:])

        xs = (resh(pos), resh(quat), resh(gp), resh(gy), jax.tree.map(resh, self.scene), jax.random.split(key, n // c))
        outs = [f(jax.tree.map(lambda x: x[i], xs)) for i in range(n // c)]
        res = [jnp.concatenate([o[j] for o in outs]) for j in range(len(outs[0]))]
        frames = res[0] if keep else None
        return frames, res[-2], res[-1]

    def _perceive(self, race, det_st, est_st, latency, gyro, acc, dt, key, start_idx):
        s = race.data.states
        pos, quat = s.pos[:, 0], s.quat[:, 0]
        frames = None
        if self.source == "detector":
            frames, corners, conf = self._frames_detect(pos, quat, race.gates_pos, race.gates_yaw, key)
            conf = jnp.where(conf >= self.thr, conf, 0.0)
        else:  # stage-4 detection model, exactly EstEnv._perceive
            prev_c, prev_conf = det_st.prev_corners, det_st.prev_conf
            det_st, corners, conf, _ = detnoise.detect(det_st, key, pos, quat, race.gates_pos, race.gates_yaw,
                                                       race.target, self.ecfg.det, self.dt)  # fmt: skip
            corners = jnp.where(latency[:, None, None, None], prev_c, corners)
            conf = jnp.where(latency[:, None], prev_conf, conf)
        est_st, einfo = estimator.est_update(est_st, corners, conf, gyro, acc, race.track_pos, race.track_yaw, dt,
                                             self.ecfg.est, start_idx)  # fmt: skip
        obs = estimator.observation(est_st, race.track_pos, race.track_yaw, race.prev_action, self.ecfg.est)
        diag = det_diag(pos, quat, race.gates_pos, race.gates_yaw, race.target, corners, conf)
        info = dict(est=einfo, corners=corners, conf=conf, diag=diag)
        if self.sel:
            sel = jnp.asarray(self.sel)
            info["sel"] = dict(corners=corners[sel], conf=conf[sel])
            if frames is not None:
                info["sel"]["frames"] = frames[sel]
        return det_st, est_st, obs, info


def det_diag(pos, quat, gp, gy, target, corners, conf):
    """Detections vs the true projected gates (contract visibility rule), per drone."""
    n, g = gp.shape[:2]
    uv, inside, visible, dist, side = detnoise.project_gates(pos, quat, gp, gy)
    uvc = labels.canonical_order(uv)  # (N, G, 4, 2) contract order
    d0 = jnp.linalg.norm(corners[:, :, None] - uvc[:, None], axis=-1).mean(-1)  # (N, K, G) as delivered
    dc = d0
    for r in range(1, 4):
        dc = jnp.minimum(dc, jnp.linalg.norm(corners[:, :, None] - jnp.roll(uvc, r, -2)[:, None], axis=-1).mean(-1))
    valid = conf > 0
    gid = jnp.where(valid & (dc.min(-1) < MATCH_PX), jnp.argmin(dc, -1), -1)  # (N, K)
    k_onehot = gid[..., None] == jnp.arange(g)  # (N, K, G)
    matched_g = k_onehot.any(1)  # (N, G)
    n_match = k_onehot.sum(1)  # slots per gate
    is_t = gid == target[:, None]
    big = 1e4
    e_c = jnp.where(is_t, jnp.take_along_axis(dc, target[:, None, None].repeat(dc.shape[1], 1), 2)[..., 0], big)
    best = jnp.argmin(e_c, 1)
    next_err = jnp.take_along_axis(e_c, best[:, None], 1)[:, 0]
    e_o = jnp.take_along_axis(d0, target[:, None, None].repeat(d0.shape[1], 1), 2)[..., 0]
    next_err_ord = jnp.take_along_axis(e_o, best[:, None], 1)[:, 0]
    next_conf = jnp.take_along_axis(jnp.where(is_t, conf, 0.0), best[:, None], 1)[:, 0]
    next_det = is_t.any(1)
    tsel = target[:, None]
    nvis = jnp.take_along_axis(visible, tsel, 1)[:, 0]
    # view angle of the next gate: between its normal and the line of sight (0 = head-on, 90 = edge-on)
    normal, _ = track.gate_axes(gy)
    gpt = jnp.take_along_axis(gp, target[:, None, None], 1)[:, 0]
    nrm = jnp.take_along_axis(normal, target[:, None, None], 1)[:, 0]
    los = gpt - pos
    cosang = jnp.abs((los * nrm).sum(-1)) / (jnp.linalg.norm(los, axis=-1) * jnp.linalg.norm(nrm, axis=-1) + 1e-9)
    occl = detnoise.occluded_fraction(pos, quat, gp, gy)
    return dict(
        nvis=visible.sum(1), ndet=valid.sum(1), nfp=(valid & (gid < 0)).sum(1),
        ndup=jnp.maximum(n_match - 1, 0).sum(1), nvisdet=(visible & matched_g).sum(1),
        next_vis=nvis, next_det=next_det, next_err=jnp.where(next_det, next_err, jnp.nan),
        next_err_ord=jnp.where(next_det, next_err_ord, jnp.nan), next_conf=next_conf,
        next_inside=jnp.take_along_axis(inside.sum(-1), tsel, 1)[:, 0],
        next_occl=jnp.take_along_axis(occl, tsel, 1)[:, 0],
        next_angle=jnp.degrees(jnp.arccos(jnp.clip(cosang, 0, 1))),
        next_dist=jnp.take_along_axis(dist, tsel, 1)[:, 0],
        next_side=jnp.take_along_axis(side, tsel, 1)[:, 0],
    )  # fmt: skip


_DIAG_DT = dict(nvis=jnp.int8, ndet=jnp.int8, nfp=jnp.int8, ndup=jnp.int8, nvisdet=jnp.int8, next_vis=bool,
                next_det=bool, next_inside=jnp.int8)  # fmt: skip


# ---------------------------------------------------------------- episodes


def slice_tree(full, tmpl, idx, n_full):
    """Per-world leaves of ``full`` (leading dim n_full) sliced to ``idx`` where the template has that shape. The
    result has the template's tree (static metadata such as Crazyflow's n_worlds differs between the two)."""
    m = len(idx)

    def take(a, b):
        if hasattr(a, "shape") and a.ndim > 0 and a.shape[0] == n_full and b.shape == (m, *a.shape[1:]):
            return a[idx]
        return b

    fl = jax.tree.leaves(full)
    tl, tdef = jax.tree.flatten(tmpl)
    assert len(fl) == len(tl), (len(fl), len(tl))
    out, kept = [], []
    for a, b in zip(fl, tl):
        out.append(take(a, b))
        if out[-1] is b and getattr(a, "shape", ()) != getattr(b, "shape", ()):
            kept.append((getattr(a, "shape", None), getattr(b, "shape", None)))
    if kept:
        print(f"slice_tree: {len(kept)} leaves of different shape kept from the template: {kept[:8]}", flush=True)
    return jax.tree.unflatten(tdef, out)


def stage4_state(env: Stage5Env, base_cfg, ecfg, key, n_full, idx, circuit_full=None):
    """Reset the stage-4 batch of n_full flights (stage-4 EstEnv) and slice flights ``idx`` into ``env``'s state.
    With idx = all, the state equals stage 4's reset exactly (same race, IMU, detection-model state and keys)."""
    idx = jnp.asarray(idx)
    full_env = EstEnv(dataclasses.replace(base_cfg, n_envs=n_full), ecfg, circuit=circuit_full)
    st_full, _ = jax.jit(full_env.reset)(key)
    tmpl, _ = jax.jit(env.race.reset)(key)
    race = slice_tree(st_full.race, tmpl, idx, n_full)
    imu_st = jax.tree.map(lambda a: a[idx], st_full.imu)
    det_st = jax.tree.map(lambda a: a[idx], st_full.det)
    lat, sidx = st_full.latency[idx], st_full.start_idx[idx]
    del st_full, full_env
    return finish_reset(env, race, det_st, imu_st, lat, sidx, key)


def finish_reset(env, race, det_st, imu_st, lat, sidx, key):
    """EstEnv.reset's perception part on a given race state (same key use as EstEnv.reset)."""
    n = race.target.shape[0]
    k = jax.random.fold_in(key, 0xE57)
    _, _, _, k4, k5 = jax.random.split(k, 5)

    @jax.jit
    def go(race, det_st, imu_st, lat, sidx):
        est_st = estimator.est_init(n, env.ecfg.est)
        z = jnp.zeros((n, 3))
        det_st2, est_st, obs, _ = env._perceive(race, det_st, est_st, lat, z, z, jnp.zeros(n), k4, sidx)
        st = EstEnvState(race=race, det=det_st2, imu=imu_st, est=est_st, latency=lat, est_obs=obs, start_idx=sidx,
                         key=k5)  # fmt: skip
        return st, jnp.concatenate([obs, env.race.obs(race)], -1)

    return go(race, det_st, imu_st, lat, sidx)


def state_from_race(env, race, key):
    """Perception state for a given race state (student protocol): as EstEnv.reset draws it."""
    n = race.target.shape[0]
    k = jax.random.fold_in(key, 0xE57)
    k1, k2, k3, _, _ = jax.random.split(k, 5)
    ds, is_, lat = env._scales(k1, n)
    det_st = detnoise.det_init(k2, ds, env.race.n_gates)
    imu_st = imu.imu_init(k3, is_, env.ecfg.imu)
    return finish_reset(env, race, det_st, imu_st, lat, env._start_idx(race), key)


# ---------------------------------------------------------------- rollout


def make_rollout(env: Stage5Env, act, steps: int):
    sel = jnp.asarray(env.sel) if env.sel else None

    @jax.jit
    def rollout(state, full):
        def step(carry, _):
            state, full, alive = carry
            race = state.race
            action = act(full, race)
            s = race.data.states
            est = state.est
            Pc, Py, own, T = estimator.gate_estimates(est, race.track_pos, race.track_yaw, env.ecfg.est)
            gate = jnp.take_along_axis(race.gates_pos, race.target[:, None, None], 1)[:, 0]
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
                locked=est.locked & alive, dist_target=d_true.astype(jnp.float16), est_nxt=est.nxt.astype(jnp.int8),
            )  # fmt: skip
            state2, full2, reward, done, info = env.step_core_est(state, action, reset_done=False)
            ev = info
            ended = alive & (ev["terminated"] | ev["truncated"])
            alive2 = alive & ~(ev["terminated"] | ev["truncated"])
            outcome = jnp.where(ev["crash_gate"], 1, jnp.where(ev["ground"], 2, jnp.where(ev["oob"], 3, 0)))
            rec |= dict(passed=ev["passed"] & alive, in_fov=ev["in_fov"], reward=reward.astype(jnp.float16),
                        outcome=outcome.astype(jnp.int8), ended=ended, d_valid=alive2)  # fmt: skip
            for k, v in info["percep"]["diag"].items():
                rec["d_" + k] = v.astype(_DIAG_DT.get(k, jnp.float16))
            if sel is not None:
                s2 = state2.race.data.states
                e2 = state2.est
                Pc2, Py2, _, T2 = estimator.gate_estimates(e2, state2.race.track_pos, state2.race.track_yaw, env.ecfg.est)
                c = jnp.take_along_axis(Pc2, e2.nxt[:, None, None], 1)[:, 0]
                y = jnp.take_along_axis(Py2, e2.nxt[:, None], 1)[:, 0]
                corners_b = estimator._apply_T(T2[:, None], track.gate_corners(c, y))  # (N, 4, 3) body frame
                rec["sel"] = info["percep"]["sel"] | dict(
                    pos=s2.pos[sel, 0], quat=s2.quat[sel, 0], gates_pos=state2.race.gates_pos[sel],
                    gates_yaw=state2.race.gates_yaw[sel], target=state2.race.target[sel], est_nxt=e2.nxt[sel],
                    locked=e2.locked[sel], belief_corners_b=corners_b[sel], alive=alive2[sel], action=action[sel],
                    passed=(ev["passed"] & alive)[sel], outcome=jnp.where(ended, outcome, 0)[sel],
                )  # fmt: skip
            return (state2, full2, alive2), rec

        _, rec = jax.lax.scan(step, (state, full, jnp.ones(env.cfg.n_envs, bool)), None, steps)
        return rec

    def run(state, full):
        rec = jax.device_get(rollout(state, full))
        sel_rec = rec.pop("sel", None)
        rec = {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in rec.items()}
        return rec, sel_rec

    return run


# ---------------------------------------------------------------- analysis


def _q(x, ps=(50, 90, 99)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return {f"p{p}": round(float(np.percentile(x, p)), 4) for p in ps} | {"n": int(len(x))} if len(x) else {"n": 0}


def det_stats(rec, dt):
    """In-flight detection statistics (frames = steps where the drone is still flying after the step)."""
    v = rec["d_valid"]
    nv = rec["d_next_vis"] & v
    nd = rec["d_next_det"] & nv
    miss = nv & ~rec["d_next_det"]
    dist, ang, ins, occ = rec["d_next_dist"], rec["d_next_angle"], rec["d_next_inside"], rec["d_next_occl"]
    out = {
        "frames": int(v.sum()),
        "gates_visible_per_frame": float(rec["d_nvis"][v].mean()),
        "visible_gate_miss_frac": float(1 - rec["d_nvisdet"][v].sum() / max(rec["d_nvis"][v].sum(), 1)),
        "false_det_per_frame": float(rec["d_nfp"][v].mean()),
        "frames_with_false_det_frac": float((rec["d_nfp"][v] > 0).mean()),
        "duplicate_det_per_frame": float(rec["d_ndup"][v].mean()),
        "next_visible_frac": float(nv.sum() / max(v.sum(), 1)),
        "next_detected_frac_when_visible": float(nd.sum() / max(nv.sum(), 1)),
        "next_err_px": _q(rec["d_next_err"][nd]),
        "next_err_px_mean": float(np.nanmean(rec["d_next_err"][nd])) if nd.any() else None,
        # delivered corner order differs from the true contract order (cyclic shift): error as delivered >> best shift
        "next_order_flip_frac": float(((rec["d_next_err_ord"][nd] - rec["d_next_err"][nd]) > 2.0).mean()) if nd.any() else None,
    }
    by = {}
    for name, m in {"0-2m": dist < 2, "2-4m": (dist >= 2) & (dist < 4), "4-8m": (dist >= 4) & (dist < 8),
                    "8-12m": dist >= 8, "partial(<4 corners in image)": ins < 4, "edge-on(>70deg)": ang > 70,
                    "occluded>=0.3": occ >= 0.3, "full_view_head_on(<45deg,4 in)": (ang < 45) & (ins == 4)}.items():  # fmt: skip
        mm = nv & m
        by[name] = {"share_of_visible_frames": float(mm.sum() / max(nv.sum(), 1)),
                    "detected_frac": float((mm & rec["d_next_det"]).sum() / max(mm.sum(), 1)),
                    "err_px": _q(rec["d_next_err"][mm & rec["d_next_det"]], (50, 90))}  # fmt: skip
    out["next_by_condition"] = by
    # miss bursts of the next gate (consecutive visible frames with the same target)
    pm = nv[:-1] & nv[1:] & (rec["target"][:-1] == rec["target"][1:])
    out["next_p_miss_after_miss"] = float((miss[1:] & miss[:-1] & pm).sum() / max((miss[:-1] & pm).sum(), 1))
    out["next_p_miss_after_hit"] = float((miss[1:] & nd[:-1] & pm).sum() / max((nd[:-1] & pm).sum(), 1))
    runs = []
    for e in range(miss.shape[1]):
        d = np.diff(np.concatenate([[0], miss[:, e].astype(int), [0]]))
        runs += list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    out["next_miss_runs"] = {"n": len(runs), "per_flight_minute": len(runs) / max(v.sum() * dt / 60, 1e-9)} | (
        {"len_frames": _q(runs, (50, 90, 99)), "max": int(max(runs))} if runs else {})
    # state the next gate is seen from (policy vs teacher flying)
    out["next_view"] = {"dist_m": _q(dist[nv], (10, 50, 90)), "angle_deg": _q(ang[nv], (50, 90, 99)),
                        "partial_frac": float((ins[nv] < 4).mean()), "edge_on_frac": float((ang[nv] > 70).mean()),
                        "occluded_frac": float((occ[nv] >= 0.3).mean())}  # fmt: skip
    alive = rec["alive"]
    sp = np.linalg.norm(rec["vel"], axis=-1)[alive]
    out["flight_state"] = {"speed_m_s": _q(sp, (50, 90)), "body_rate_rad_s": _q(np.linalg.norm(rec["ang_vel"], axis=-1)[alive], (50, 90, 99))}
    return out


def _window_any(mask, w):
    """any(mask[t - w : t + 1]) per (t, e)."""
    c = np.cumsum(np.concatenate([np.zeros((1, mask.shape[1])), mask], 0), 0)
    lo = np.maximum(np.arange(mask.shape[0]) - w, 0)
    return (c[np.arange(mask.shape[0]) + 1] - c[lo]) > 0


def crash_diag(rec, dt, w_s=0.5):
    """When crashes happen and whether detector errors precede them (window w_s before the crash step), against
    the chance level of the same label over all flying steps."""
    alive, ended = rec["alive"], rec["ended"]
    crashed = ended & (rec["outcome"] > 0)
    t_idx, e_idx = np.nonzero(crashed)
    w = int(round(w_s / dt))
    v = rec["d_valid"] | crashed  # frames of the flight incl. the crash frame
    nv = rec["d_next_vis"] & v
    miss = nv & ~rec["d_next_det"]
    err = np.nan_to_num(rec["d_next_err"], nan=0.0)
    labels_ = {
        "next_gate_miss_while_visible": miss,
        "next_gate_miss_burst>=3": _window_any(np.zeros_like(miss), 0),  # filled below
        "false_detection": (rec["d_nfp"] > 0) & v,
        "next_corner_err>2px": (err > 2) & nv,
        "next_partial_view(<4 corners)": nv & (rec["d_next_inside"] < 4),
        "next_edge_on(>70deg)": nv & (rec["d_next_angle"] > 70),
        "next_occluded>=0.3": nv & (rec["d_next_occl"] >= 0.3),
        "next_gate_not_visible": v & ~rec["d_next_vis"],
        "next_pose_wrong": rec["pose_wrong"],
    }
    # miss bursts >= 3 frames
    b = np.zeros_like(miss)
    for e in range(miss.shape[1]):
        d = np.diff(np.concatenate([[0], miss[:, e].astype(int), [0]]))
        for s0, s1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if s1 - s0 >= 3:
                b[s0:s1, e] = True
    labels_["next_gate_miss_burst>=3"] = b
    out = {"n_crashes": int(len(t_idx)), "window_s": w_s}
    if not len(t_idx):
        return out
    tt = t_idx * dt
    edges = [0, 1, 2, 3, 5, 10, 20, 30.01]
    out["time_hist"] = {"edges_s": edges, "count": np.histogram(tt, edges)[0].tolist()}
    out["time_s"] = _q(tt, (25, 50, 75))
    passed = rec["passed"]
    out["within_1s_after_a_gate_pass_frac"] = float(np.mean([passed[max(t - 50, 0) : t + 1, e].any() for t, e in zip(t_idx, e_idx)]))
    out["before_first_gate_pass_frac"] = float(np.mean([not passed[: t + 1, e].any() for t, e in zip(t_idx, e_idx)]))
    out["outcome"] = {k: int((rec["outcome"][t_idx, e_idx] == c).sum()) for c, k in ((1, "gate"), (2, "ground"), (3, "oob"))}
    pre = {}
    flying = alive & ~crashed
    for name, m in labels_.items():
        win = _window_any(m, w)
        pre[name] = {"crash_frac": float(win[t_idx, e_idx].mean()), "chance_all_steps": float(win[flying].mean())}
    out["preceded_by"] = pre
    # cause classes: wrong belief (pose wrong in window) vs detector error in window with belief right
    wb = _window_any(rec["pose_wrong"], w)[t_idx, e_idx]
    deterr = (_window_any(miss, w) | _window_any((rec["d_nfp"] > 0) & v, w) | _window_any((err > 2) & nv, w))[t_idx, e_idx]
    oc = rec["outcome"][t_idx, e_idx]
    out["classes"] = {
        "wrong_belief_with_detector_error": int((wb & deterr).sum()),
        "wrong_belief_no_detector_error": int((wb & ~deterr).sum()),
        "belief_right_gate_hit": int((~wb & (oc == 1)).sum()),
        "belief_right_ground_or_oob": int((~wb & (oc > 1)).sum()),
        "belief_right_with_detector_error": int((~wb & deterr).sum()),
    }
    return out


def student_laps(rec, dt):
    """eval_student / dagger.summarize lap time: mean over all (p[k+6] - p[k]) of an episode's pass steps."""
    laps = []
    for e in range(rec["passed"].shape[1]):
        p = np.flatnonzero(rec["passed"][:, e])
        if len(p) > 6:
            laps.append((p[6:] - p[:-6]) * dt)
    return float(np.concatenate(laps).mean()) if laps else None


def analyse(rec, dt):
    return metrics(rec, 6, dt) | {"perception": percep_metrics(rec, dt), "det_stats": det_stats(rec, dt),
                                  "crash_diag": crash_diag(rec, dt), "student_lap_time_s": student_laps(rec, dt)}  # fmt: skip


# ---------------------------------------------------------------- modes


def parse_idx(s, n):
    if ":" in s:
        a, b = s.split(":")
        return np.arange(int(a or 0), int(b or n))
    return np.asarray([int(x) for x in s.split(",")])


def cond_cfg(args):
    """Stage-4 condition of the run (x1 IMU with the detector), with the optional --imu-scale override."""
    c = condition("x1" if args.source == "detector" else args.cond)
    if args.imu_scale is not None:
        c = dataclasses.replace(c, imu_scale_min=args.imu_scale, imu_scale_max=args.imu_scale)
    return c


def build_env(args, env_cfg, est_cfg, m, scene, circuit=None, sel=()):
    ecfg = dataclasses.replace(cond_cfg(args), known_start=args.known_start, est=est_cfg)
    if args.source == "detector":
        ecfg = dataclasses.replace(ecfg, est=dataclasses.replace(est_cfg, conf_min=args.thr))
    env = Stage5Env(dataclasses.replace(env_cfg, n_envs=m), ecfg, args.source, args.detector, args.thr, scene,
                    min(args.chunk, m), circuit, sel=sel, keep_frames=bool(sel))  # fmt: skip
    return env, ecfg


def policy_and_est(args):
    act, env_cfg, p_ecfg = make_policy(args.policy)
    from check_estimator_sweep import parse_variant

    est_cfg = dataclasses.replace(p_ecfg.est if p_ecfg is not None else condition("x1").est,
                                  **parse_variant(",".join(args.est)))  # fmt: skip
    env_cfg = dataclasses.replace(env_cfg, rand_action_noise=0.0, p_wide=0.0)
    return act, env_cfg, est_cfg


def save_sel(path, sel_rec, sel, extra):
    if sel_rec is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrs = {k: np.asarray(v) for k, v in sel_rec.items()}
    np.savez_compressed(path, **arrs, sel=np.asarray(sel), meta=json.dumps(extra))


def mode_run(args):
    act, env_cfg, est_cfg = policy_and_est(args)
    dt = 1 / env_cfg.policy_hz
    B = args.batch
    if args.block == "random":
        n_full, key, circ_full, off = B, jax.random.key(args.seed), None, 0
    else:
        names = list(track.SCENARIOS)
        b = names.index(args.block)
        layouts = [track.circuit_arrays(track.resolve(nm)) for nm in names]
        circ_full = (np.repeat(np.stack([p for p, _ in layouts]), B, 0), np.repeat(np.stack([y for _, y in layouts]), B, 0))
        n_full, key, off = len(names) * B, jax.random.key(args.seed + 20), b * B
    idx = parse_idx(args.idx, B) + off
    m = len(idx)
    scene = None
    if args.source == "detector":
        sc = labels.scene_pool(n_full, args.scene_seed if args.scene_seed is not None else args.seed + (0 if args.block == "random" else 20))
        scene = {k: v[idx] for k, v in sc.items()}
    circ = None if circ_full is None else (circ_full[0][idx], circ_full[1][idx])
    sel = [int(x) for x in args.video.split(",")] if args.video else []
    env, ecfg = build_env(args, env_cfg, est_cfg, m, scene, circ, sel)
    base_cond = ("x1" if args.source == "detector" else args.cond) + (f"_imu{args.imu_scale}" if args.imu_scale is not None else "")
    reset_ecfg = dataclasses.replace(cond_cfg(args), known_start=args.known_start, est=ecfg.est)
    t0 = time.time()
    state, full = stage4_state(env, env_cfg, reset_ecfg, key, n_full, idx, circ_full)
    run = make_rollout(env, act, args.steps)
    rec, sel_rec = run(state, full)
    wall = time.time() - t0
    res = {"mode": "run", "policy": args.policy, "source": args.source, "cond": base_cond, "thr": args.thr,
           "detector": args.detector if args.source == "detector" else None, "block": args.block, "batch": B,
           "seed": args.seed if args.block == "random" else args.seed + 20, "idx": [int(idx[0] - off), int(idx[-1] - off) + 1] if args.idx.count(":") else idx.tolist(),
           "flights": m, "steps": args.steps, "known_start": args.known_start, "est": dataclasses.asdict(ecfg.est),
           "scene_seed": args.scene_seed, "wall_s": round(wall, 1), "flights_per_s": m / wall}  # fmt: skip
    res |= analyse(rec, dt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.block + (f"_{args.name}" if args.name else "")
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    if args.rec_dir:
        Path(args.rec_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(args.rec_dir) / f"{Path(args.out).name}_{name}.npz", **rec)
    if sel:
        save_sel(Path(args.rec_dir or out) / f"video_{Path(args.out).name}_{name}.npz", sel_rec, sel,
                 dict(block=args.block, idx=[int(idx[i] - off) for i in sel], policy=args.policy, thr=args.thr))
    print(f"[{name}] 3 laps {res['frac_3_laps']:.4f} 1 lap {res['frac_1_lap']:.4f} crash {res['crash_rate']:.4f} "
          f"lap {res['lap_time_s']['flying_mean']} next det {res['det_stats']['next_detected_frac_when_visible']:.5f} "
          f"err p50 {res['det_stats']['next_err_px'].get('p50')} fp/f {res['det_stats']['false_det_per_frame']:.4f} "
          f"pose-wrong {res['perception']['next_pose_wrong_step_frac']:.4f} ({wall:.0f} s, {m / wall:.2f} flights/s)", flush=True)  # fmt: skip


def mode_student(args):
    from fpv.closed_loop import FlightBatch

    act, _, est_cfg = policy_and_est(args)
    fb = FlightBatch(args.n, TEACHER, seed=args.fb_seed, randomize=True)
    for _ in range(args.resets):
        fb.reset()
    race = fb._state
    env_cfg = fb.cfg
    dt = 1 / env_cfg.policy_hz
    scene = labels.scene_pool(args.n, args.fb_seed) if args.source == "detector" else None
    env, ecfg = build_env(args, env_cfg, est_cfg, args.n, scene)
    t0 = time.time()
    state, full = state_from_race(env, race, jax.random.key(args.fb_seed * 100 + args.resets))
    rec, _ = make_rollout(env, act, args.steps)(state, full)
    wall = time.time() - t0
    res = {"mode": "student", "policy": args.policy, "source": args.source, "thr": args.thr, "fb_seed": args.fb_seed,
           "resets": args.resets, "flights": args.n, "steps": args.steps, "known_start": args.known_start,
           "wall_s": round(wall, 1)} | analyse(rec, dt)  # fmt: skip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"student_{args.name}"
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"[{name}] 3 laps {res['frac_3_laps']:.4f} 1 lap {res['frac_1_lap']:.4f} crash {res['crash_rate']:.4f} "
          f"lap {res['student_lap_time_s']} ({wall:.0f} s)", flush=True)  # fmt: skip


def mode_bench(args):
    """ms per control step of the whole chain (sim + render + detector + estimator + policy) and of the on-board
    part (detector + estimator + policy on a given frame) at several batch sizes."""
    act, env_cfg, est_cfg = policy_and_est(args)
    res = {"device": str(jax.devices()[0]), "policy": args.policy}
    for n in [int(x) for x in args.bench_n.split(",")]:
        sc = labels.scene_pool(n, 1)
        env, ecfg = build_env(args, env_cfg, est_cfg, n, sc)
        state, full = stage4_state(env, env_cfg, dataclasses.replace(condition("x1"), est=ecfg.est), jax.random.key(1), n,
                                   np.arange(n))  # fmt: skip
        steps = args.bench_steps
        roll = make_rollout(env, act, steps)
        t = time.time()
        jax.block_until_ready(jax.tree.leaves(roll(state, full)[0])[0])
        t_compile = time.time() - t
        ts = []
        for _ in range(3):
            t = time.time()
            r, _ = roll(state, full)
            ts.append(time.time() - t)
        ms = 1000 * min(ts) / steps
        # on-board part only: detector + estimator + policy for one frame (frame rendered beforehand)
        s = state.race.data.states
        frames, _ = env.render(s.pos[:, 0], s.quat[:, 0], state.race.gates_pos, state.race.gates_yaw, env.scene,
                               jax.random.key(0))  # fmt: skip

        @jax.jit
        def onboard(est_st, frames, gyro, acc, race):
            out = env.det.forward(env.det.params, frames)
            conf = jnp.where(out["conf"] >= env.thr, out["conf"], 0.0)
            est_st, _ = estimator.est_update(est_st, out["corners"], conf, gyro, acc, race.track_pos, race.track_yaw,
                                             jnp.full(n, 0.02), env.ecfg.est, state.start_idx)  # fmt: skip
            obs = estimator.observation(est_st, race.track_pos, race.track_yaw, race.prev_action, env.ecfg.est)
            full_ = jnp.concatenate([obs, env.race.obs(race)], -1)
            return est_st, act(full_, race)

        z = jnp.zeros((n, 3))
        jax.block_until_ready(onboard(state.est, frames, z, z, state.race))
        tt = []
        for _ in range(args.bench_reps):
            t = time.perf_counter()
            jax.block_until_ready(onboard(state.est, frames, z, z, state.race))
            tt.append(time.perf_counter() - t)
        rf = jax.jit(lambda p, q: env.render(p, q, state.race.gates_pos, state.race.gates_yaw, env.scene, jax.random.key(0))[0])
        jax.block_until_ready(rf(s.pos[:, 0], s.quat[:, 0]))
        tr = []
        for _ in range(min(args.bench_reps, 50)):
            t = time.perf_counter()
            jax.block_until_ready(rf(s.pos[:, 0], s.quat[:, 0]))
            tr.append(time.perf_counter() - t)
        res[f"batch_{n}"] = {
            "whole_chain_ms_per_step": ms, "whole_chain_steps_per_s": 1000 / ms,
            "flights_per_s_30s": n / (ms * 1500 / 1000), "drone_steps_per_s": n * 1000 / ms,
            "compile_plus_first_run_s": t_compile,
            "onboard_det_est_policy_ms": {"median": 1000 * float(np.median(tt)), "p99": 1000 * float(np.percentile(tt, 99))},
            "render_only_ms": {"median": 1000 * float(np.median(tr))},
        }  # fmt: skip
        print(n, json.dumps(res[f"batch_{n}"]), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bench_{args.name or 'dev'}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "student", "bench"])
    ap.add_argument("--policy", default="runs/percep/policy/train/est_rand_v5/ckpt_latest.pkl")
    ap.add_argument("--source", default="detector", choices=["detector", "true"])
    ap.add_argument("--cond", default="perfect", help="stage-4 condition for --source true (perfect, x1, ...)")
    ap.add_argument("--imu-scale", type=float, default=None, help="override the condition's IMU noise scale")
    ap.add_argument("--detector", default=DET)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--known-start", action="store_true")
    ap.add_argument("--est", nargs="*", default=[])
    ap.add_argument("--block", default="random", help="random or a layout name")
    ap.add_argument("--batch", type=int, default=2048, help="stage-4 batch (random) / per-layout block size")
    ap.add_argument("--idx", default=":", help="flights of the block: a:b or a comma list")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--scene-seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--video", default="", help="flight indices (within the run) whose frames are kept")
    ap.add_argument("--rec-dir", default=None)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", required=True)
    # student protocol
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--fb-seed", type=int, default=300050)
    ap.add_argument("--resets", type=int, default=1)
    # bench
    ap.add_argument("--bench-n", default="1,64,256,2048")
    ap.add_argument("--bench-steps", type=int, default=200)
    ap.add_argument("--bench-reps", type=int, default=200)
    args = ap.parse_args()
    {"run": mode_run, "student": mode_student, "bench": mode_bench}[args.mode](args)


if __name__ == "__main__":
    main()
