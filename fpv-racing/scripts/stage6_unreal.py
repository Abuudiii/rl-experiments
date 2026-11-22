"""Stage 6: the perception pipeline (detector -> geometry / ordering -> estimator v5 + IMU -> policy) flying from
Unreal frames (Project AirSim Blocks via `fpv.render_airsim.AirsimRenderer`, stock look, NO next-gate highlight).

The chain is stage 5's (`scripts/eval_stage5.py`): the same flights, estimator, thresholds, latency convention
(the frame of the state after physics step t feeds the action of step t + 1) and metrics; only the frame source
changes. Unreal is not JAX, so every control step is a host loop: jitted physics + IMU (``pre``) -> Unreal frames
of the new poses (one sim instance per flight) -> jitted detector -> jitted estimator + observation (``post``).
`pre` / `post` together are `fpv.env_est.EstEnv.step_core_est(reset_done=False)` split at `_perceive`.

Modes (renderer args: ``--ue '{"gpus":[0,1],"instances_per_gpu":4,"look":"stock"}'``):
  bench    Unreal frames/s per configuration (closed-loop `render` and open-loop `render_trajectories`).
  frames   teacher-flown poses (a stage-1 pose bank, or fresh `labels.collect_flights`) rendered in Unreal ->
           npz shards (frames uint8 + pose + layout + target + episode + t) for the label check, detector
           evaluation and fine-tuning.
  labels   label alignment: projected opening corners (`labels.gate_labels`) vs the rendered gate frame.
  detect   detector on Unreal frame shards: stage-1 tables (`eval_detector.evaluate_set`), and the same poses
           rendered by the synthetic ray caster for a paired row.
  fly      closed loop in Unreal on stage-5 flights (stage-4 batch, ``--block`` layout or random, ``--idx``).
  video    the published-style flight (FlightBatch ``--fb-seed`` 900002, nominal physics, slot 0) flown by the
           pipeline, rendered by `scripts/flight_video.py` (chase / FPV / overview, plain gates) + an FPV video
           with the detections drawn.
  replay-video  the `video` outputs of a `video --record-only` flight from its flight.npz + pipeline_fpv.npz
           (in-flight frames, detections, estimator next gate); nothing is re-flown.
  report   rebuild the per-block JSON from the per-batch records of `fly` (`fly` also writes timing_<block>.json:
           whole-chain ms per control step in Unreal mode, split into physics, render, detector, estimator + policy).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

from fpv import track  # noqa: E402

DET = "runs/percep/detector/E_amd/ckpt_best.pkl"
POLICY = "runs/percep/policy/train/est_rand_v5/ckpt_latest.pkl"
TEACHER = "runs/teacher/rt_G_D_noise3/ckpt_latest.pkl"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def renderer(ue: str, **kw):
    from fpv.render_airsim import AirsimRenderer

    a = json.loads(ue) | kw
    if a.get("highlight_target"):
        raise ValueError("stage 6 flies without the next-gate highlight")
    a.setdefault("startup_timeout", 400)
    a.setdefault("max_restarts", 5)
    t0 = time.time()
    r = AirsimRenderer(**a)
    log(f"Unreal renderer: {r.n} slots on GPUs {r.gpus} in {time.time() - t0:.0f} s")
    return r


def quat_rpy(roll, pitch, yaw):
    """xyzw body->world from intrinsic Z (yaw) -> Y (pitch, nose down positive in the contract frame) -> X (roll)."""
    cr, sr, cp, sp, cy, sy = (np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(yaw / 2),
                              np.sin(yaw / 2))  # fmt: skip
    return np.stack([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy], -1)  # fmt: skip


def approach_poses(rng, gp, gy, n):
    """n random poses in front of random gates of the layout (bench only)."""
    g = rng.integers(0, len(gy), n)
    normal, left = track.gate_axes(gy[g])
    d = rng.uniform(0.8, 6.0, n)[:, None]
    pos = gp[g] - normal * d + left * rng.uniform(-0.8, 0.8, (n, 1))
    pos[:, 2] = np.maximum(pos[:, 2] + rng.uniform(-0.4, 0.4, n), 0.15)
    q = quat_rpy(rng.uniform(-0.3, 0.3, n), rng.uniform(-0.1, 0.4, n), gy[g] + rng.uniform(-0.4, 0.4, n))
    return pos, q


# ============================================================================ bench
def mode_bench(a):
    gp, gy = track.circuit_arrays(track.resolve("figure8"))
    rng = np.random.default_rng(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((out / "bench_fps.json").read_text()) if (out / "bench_fps.json").exists() else []
    for spec in a.configs.split(";"):
        gpus, ipg = spec.split(":")
        gpus = [int(x) for x in gpus.split(",")]
        t0 = time.time()
        r = renderer(json.dumps({"gpus": gpus, "instances_per_gpu": int(ipg), "look": a.look}))
        start = time.time() - t0
        try:
            n = r.n
            r.set_gates(np.arange(n), np.repeat(gp[None], n, 0), np.repeat(gy[None], n, 0))
            pos, q = approach_poses(rng, gp, gy, n * (a.steps + 5))
            pos, q = pos.reshape(-1, n, 3), q.reshape(-1, n, 4)
            for i in range(5):
                r.render(pos[i], q[i])
            t = time.time()
            for i in range(5, a.steps + 5):
                r.render(pos[i], q[i])
            fps_r = n * a.steps / (time.time() - t)
            T = 64
            pt, qt = approach_poses(rng, gp, gy, n * T)
            r.render_trajectories(np.arange(n), pt.reshape(n, T, 3), qt.reshape(n, T, 4))
            t = time.time()
            reps = max(1, a.steps // T)
            for _ in range(reps):
                r.render_trajectories(np.arange(n), pt.reshape(n, T, 3), qt.reshape(n, T, 4))
            fps_t = n * T * reps / (time.time() - t)
            t = time.time()
            import subprocess

            subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=60)
            row = dict(gpus=gpus, instances_per_gpu=int(ipg), slots=n, look=a.look, startup_s=round(start, 1),
                       render_frames_per_s=round(fps_r, 2), render_frames_per_s_per_gpu=round(fps_r / len(gpus), 2),
                       traj_frames_per_s=round(fps_t, 2), traj_frames_per_s_per_gpu=round(fps_t / len(gpus), 2),
                       closed_loop_steps_per_s_per_flight=round(fps_r / n, 2), nvidia_smi_s=round(time.time() - t, 2),
                       restarts=r.restarts, cpus=len(os.sched_getaffinity(0)))  # fmt: skip
        finally:
            r.close()
        log(json.dumps(row))
        res.append(row)
        (out / "bench_fps.json").write_text(json.dumps(res, indent=1))


# ============================================================================ frames
def load_poses(a):
    from fpv.perception import labels as L

    if a.bank:
        z = np.load(a.bank)
        b = {k: z[k] for k in z.files if k != "meta"}
    else:
        kind, seed, n_envs, steps = a.flights.split(":")
        b = L.collect_flights(a.teacher, int(n_envs), int(steps), int(seed), None if kind in ("random", "teacher") else kind,
                              wide_spawn_p=a.wide_p, teacher_noise=a.teacher_noise, log=log)  # fmt: skip
    rng = np.random.default_rng(a.seed)
    off = (b["episode"] * 7919 + a.seed) % a.every
    keep = np.flatnonzero((b["t"] % a.every) == off)
    if a.max_frames and len(keep) > a.max_frames:
        keep = np.sort(rng.choice(keep, a.max_frames, replace=False))
    b = {k: v[keep] for k, v in b.items()}
    if a.perturb > 0:
        import jax
        import jax.numpy as jnp

        p, q = L.perturb_poses(jax.random.key(a.seed), jnp.asarray(b["pos"]), jnp.asarray(b["quat"]), p=a.perturb)
        b["pos"], b["quat"] = np.asarray(p, np.float32), np.asarray(q, np.float32)
    o = np.lexsort((b["t"], b["episode"]))
    return {k: v[o] for k, v in b.items()}


def segments(b, seg=32):
    """(start, end) index ranges of <= seg consecutive frames of one episode (one layout each)."""
    out = []
    ep = b["episode"]
    starts = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1, len(ep)]
    for s, e in zip(starts[:-1], starts[1:]):
        for i in range(s, e, seg):
            out.append((i, min(e, i + seg)))
    return out


def render_poses(r, b, log_every=50):
    """Unreal frames of every pose of b (grouped by episode, layout set per slot)."""
    from fpv.render_airsim import CHUNK

    segs = segments(b, CHUNK)
    frames = np.zeros((len(b["pos"]), r.height, r.width, 3), np.uint8)
    cur = np.full(r.n, -1)
    t0 = time.time()
    for k, i in enumerate(range(0, len(segs), r.n)):
        batch = segs[i : i + r.n]
        slots = np.arange(len(batch))
        eps = np.array([b["episode"][s] for s, _ in batch])
        ch = slots[eps != cur[slots]]
        if len(ch):
            r.set_gates(ch, np.stack([b["gates_pos"][batch[j][0]] for j in ch]),
                        np.stack([b["gates_yaw"][batch[j][0]] for j in ch]))  # fmt: skip
            cur[ch] = eps[ch]
        T = max(e - s for s, e in batch)
        idx = np.stack([np.minimum(np.arange(s, s + T), e - 1) for s, e in batch])
        fr = r.render_trajectories(slots, b["pos"][idx], b["quat"][idx])
        for j, (s, e) in enumerate(batch):
            frames[s:e] = fr[j, : e - s]
        if (k + 1) % log_every == 0:
            done = batch[-1][1]
            log(f"  rendered {done}/{len(frames)} ({done / (time.time() - t0):.1f} frames/s)")
    return frames


def mode_frames(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = load_poses(a)
    log(f"{len(b['pos'])} poses, {len(np.unique(b['episode']))} episodes")
    n_sh = math.ceil(len(b["pos"]) / a.shard)
    todo = [i for i in range(n_sh) if not (out / f"shard_{i:04d}.npz").exists()]
    if not todo:
        log("all shards exist")
        return
    r = renderer(a.ue)
    try:
        for i in todo:
            sl = slice(i * a.shard, (i + 1) * a.shard)
            bi = {k: v[sl] for k, v in b.items()}
            t = time.time()
            fr = render_poses(r, bi)
            tmp = out / f"shard_{i:04d}.tmp.npz"
            (np.savez_compressed if a.compress else np.savez)(
                tmp, frames=fr, **{k: bi[k] for k in ("pos", "quat", "gates_pos", "gates_yaw", "target", "episode", "t")})
            tmp.rename(out / f"shard_{i:04d}.npz")
            log(f"shard {i + 1}/{n_sh}: {len(fr)} frames in {time.time() - t:.0f} s ({len(fr) / (time.time() - t):.1f}/s)")
    finally:
        r.close()
    (out / "meta.json").write_text(json.dumps(dict(vars(a), frames=int(len(b["pos"])), restarts=r.restarts), indent=1))


def load_shards(dirs, max_frames=0):
    parts = []
    for d in dirs:
        for i, p in enumerate(sorted(Path(d).glob("shard_*.npz"))):
            z = np.load(p)
            s = {k: z[k] for k in z.files}
            s["episode"] = s["episode"].astype(np.int64) + 10_000_000 * len(parts)  # episodes of shards/dirs apart
            parts.append(s)
    b = {k: np.concatenate([s[k] for s in parts]) for k in parts[0]}
    if max_frames and len(b["pos"]) > max_frames:
        b = {k: v[:max_frames] for k, v in b.items()}
    return b


# ============================================================================ label check
def gate_mask(fr):
    """Magenta gate-frame pixels (stock look: GATE_RGB (1, 0, 0.8) lit by the sun / sky)."""
    f = fr.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return (r > 60) & (b > 45) & (g * 2 < np.minimum(r, b)) & (r + b > 140)


def mode_labels(a):
    from fpv.perception import labels as L
    from fpv.render_airsim import GATE_DEPTH

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.bank)
    bank = {k: z[k] for k in z.files if k != "meta"}
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(bank["pos"]), a.n, replace=False))
    b = {k: v[idx] for k, v in bank.items()}
    o = np.lexsort((b["t"], b["episode"]))
    b = {k: v[o] for k, v in b.items()}
    S = 4
    r = renderer(a.ue, width=L.W * S, height=L.H * S)
    try:
        hi = render_poses(r, b)
    finally:
        r.close()
    r = renderer(a.ue)
    try:
        lo = render_poses(r, b)
    finally:
        r.close()
    np.savez_compressed(out / "label_frames.npz", frames=lo, **{k: b[k] for k in ("pos", "quat", "gates_pos", "gates_yaw", "target", "episode", "t")})
    sel = lambda k: b[k].astype(np.float64)  # noqa: E731
    lab = L.gate_labels(sel("pos"), sel("quat"), sel("gates_pos"), sel("gates_yaw"))
    nrm_w, _ = track.gate_axes(sel("gates_yaw"))
    faces = [L.gate_labels(sel("pos"), sel("quat"), sel("gates_pos") + s * GATE_DEPTH / 2 * nrm_w, sel("gates_yaw"))["corners"]
             for s in (-1, 1)]  # fmt: skip
    masks = gate_mask(hi)
    ts = np.linspace(0.2, 0.8, 7)
    ss = np.arange(-3.0, 4.0 + 1e-9, 0.125)
    full = lab["visible"] & (lab["n_inside"] == 4) & (lab["side_px"] >= 8)
    trans, resid, dist_of, nofr, bad_inside = [], [], [], 0, 0
    for f, g in zip(*np.nonzero(full)):
        q = lab["corners"][f, g]
        c = q.mean(0)
        # the opening itself must be free of gate pixels near the centre (no nearer gate in front)
        for e in range(4):
            a_, b_ = q[e], q[(e + 1) % 4]
            d = (b_ - a_) / np.linalg.norm(b_ - a_)
            nrm = np.array([d[1], -d[0]])
            if np.dot(nrm, (a_ + b_) / 2 - c) < 0:
                nrm = -nrm
            pf = []
            for fq in faces:
                qq = fq[f, g]
                a2, b2 = qq[np.argmin(np.linalg.norm(qq - a_, axis=1))], qq[np.argmin(np.linalg.norm(qq - b_, axis=1))]
                d2 = (b2 - a2) / max(np.linalg.norm(b2 - a2), 1e-9)
                pf.append((a2, np.array([d2[1], -d2[0]])))
            for t in ts:
                p = a_ + t * (b_ - a_)
                pts = p[None] + ss[:, None] * nrm[None]
                fx = np.round((pts[:, 0] + 0.5) * S - 0.5).astype(int)
                fy = np.round((pts[:, 1] + 0.5) * S - 0.5).astype(int)
                okp = (fx >= 0) & (fx < L.W * S) & (fy >= 0) & (fy < L.H * S)
                hit = np.zeros(len(ss), bool)
                hit[okp] = masks[f, fy[okp], fx[okp]]
                if not hit.any():
                    nofr += 1
                    continue
                k = int(np.argmax(hit))
                if k == 0:  # gate pixels already inside the opening: another gate in front, skip
                    bad_inside += 1
                    continue
                trans.append(ss[k] - 0.0625)
                dist_of.append(lab["dist"][f, g])
                exp = min(np.dot(a2 - p, n2) / np.dot(nrm, n2) for a2, n2 in pf)
                resid.append(trans[-1] - exp)
    trans, resid, dist_of = np.asarray(trans), np.asarray(resid), np.asarray(dist_of)
    res = dict(frames=int(a.n), gates_checked=int(full.sum()), edge_samples=int(len(trans)), no_frame_found=int(nofr),
               gate_pixels_inside_opening=int(bad_inside), median=float(np.median(trans)), mean=float(trans.mean()),
               p10=float(np.percentile(trans, 10)), p90=float(np.percentile(trans, 90)),
               frac_abs_le_0p25=float((np.abs(trans) <= 0.25).mean()), frac_abs_le_0p5=float((np.abs(trans) <= 0.5).mean()),
               frac_abs_le_1=float((np.abs(trans) <= 1).mean()))  # fmt: skip
    for lo_, hi_ in ((0, 2), (2, 4), (4, 8), (8, 12)):
        m = (dist_of >= lo_) & (dist_of < hi_)
        if m.any():
            res[f"median_{lo_}_{hi_}m"] = float(np.median(trans[m]))
            res[f"face_resid_median_{lo_}_{hi_}m"] = float(np.median(resid[m]))
            res[f"frac_abs_le_0p5_{lo_}_{hi_}m"] = float((np.abs(trans[m]) <= 0.5).mean())
    res["face_aware_residual"] = dict(median=float(np.median(resid)), mean=float(resid.mean()), p1=float(np.percentile(resid, 1)),
                                      p99=float(np.percentile(resid, 99)), frac_abs_le_0p25=float((np.abs(resid) <= 0.25).mean()),
                                      frac_abs_le_0p5=float((np.abs(resid) <= 0.5).mean()))  # fmt: skip
    res["note"] = ("Unreal frames rendered at 1280x720 (4x); magenta mask; transition = signed px at 320x180 from the projected "
                   "opening edge (gate mid-plane) to the first gate pixel, outward; residual = minus the nearer projected face edge")  # fmt: skip
    (out / "label_check_unreal.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-(lab["visible"].sum(1) + 0.1 * rng.random(len(lo))))
    pick = np.r_[order[:6], np.flatnonzero((lab["visible"] & (lab["n_inside"] < 4)).any(1))[:2]][:8]
    fig, axs = plt.subplots(2, 4, figsize=(20, 5.9))
    cols = ["yellow", "cyan", "lime", "orange"]
    for ax, f in zip(axs.ravel(), pick):
        ax.imshow(lo[f], extent=(-0.5, L.W - 0.5, L.H - 0.5, -0.5))
        for g in range(6):
            if not (lab["visible"][f, g] or lab["dontcare"][f, g]):
                continue
            q = lab["corners"][f, g]
            ax.plot(*np.r_[q, q[:1]].T, "-" if lab["visible"][f, g] else ":", color="white", lw=0.6)
            for c in range(4):
                ax.plot(*q[c], "o", ms=3, color=cols[c])
            ax.text(*q.mean(0), f"G{g} {lab['dist'][f, g]:.1f}m", color="white", fontsize=6)
        ax.set_xlim(-20.5, L.W + 19.5)
        ax.set_ylim(L.H + 19.5, -20.5)
        ax.set_title(f"frame {f}: {int(lab['visible'][f].sum())} visible", fontsize=8)
        ax.axis("off")
    fig.suptitle("labels on UNREAL frames (320x180): corner 1 yellow, 2 cyan, 3 lime, 4 orange; solid visible, dotted don't-care", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "label_sheet_unreal.png", dpi=110)
    # zoomed crops at 4x: one near gate per frame, label quad drawn on the 1280x720 frame
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for ax, f in zip(axs.ravel(), pick):
        g = int(np.argmin(np.where(full[f], lab["dist"][f], np.inf)))
        if not full[f, g]:
            ax.axis("off")
            continue
        q = (lab["corners"][f, g] + 0.5) * S - 0.5
        lo_xy, hi_xy = q.min(0) - 0.6 * (q.max(0) - q.min(0)), q.max(0) + 0.6 * (q.max(0) - q.min(0))
        ax.imshow(hi[f])
        ax.plot(*np.r_[q, q[:1]].T, "-", color="cyan", lw=0.8)
        ax.set_xlim(lo_xy[0], hi_xy[0])
        ax.set_ylim(hi_xy[1], lo_xy[1])
        ax.set_title(f"frame {f} G{g} {lab['dist'][f, g]:.1f} m (1280x720)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "label_zoom_unreal.png", dpi=90)
    log(f"wrote {out}")


# ============================================================================ detector evaluation
def detect_frames(det, frames, chunk=64):
    import jax
    import jax.numpy as jnp

    from fpv.perception.detector import decode

    f = jax.jit(lambda p, x: decode(det.cfg, det.net.apply(p, x), snap=det.snap))
    cs, cf, cr = [], [], []
    for i in range(0, len(frames), chunk):
        x = frames[i : i + chunk]
        m = len(x)
        if m < chunk:
            x = np.concatenate([x, np.repeat(x[-1:], chunk - m, 0)])
        o = f(det.params, jnp.asarray(x))
        cs.append(np.asarray(o["corners"])[:m])
        cf.append(np.asarray(o["conf"])[:m])
        cr.append(np.asarray(o["corners_reg"])[:m])
    return dict(corners=np.concatenate(cs), conf=np.concatenate(cf), corners_reg=np.concatenate(cr))


def hidden_frac(b, chunk=64):
    """labels.hidden_fraction from the synthetic hit map of the same poses (occlusion by nearer gates)."""
    import jax
    import jax.numpy as jnp

    from fpv.perception import labels as L

    render = L.make_renderer()
    sc = {k: jnp.asarray(v[:1].repeat(chunk, 0)) for k, v in L.scene_pool(1, 0).items()}

    @jax.jit
    def go(p, q, gp, gy):
        _, ids = render(p, q, gp, gy, sc, jax.random.key(0))
        return L.hidden_fraction(p, q, gp, gy, ids, L.gate_labels(p, q, gp, gy)["side_px"])

    out = []
    for i in range(0, len(b["pos"]), chunk):
        j = np.arange(i, min(i + chunk, len(b["pos"])))
        jp = np.r_[j, np.full(chunk - len(j), j[-1])]
        out.append(np.asarray(go(*(jnp.asarray(b[k][jp]) for k in ("pos", "quat", "gates_pos", "gates_yaw"))))[: len(j)])
    return np.concatenate(out)


def synth_frames(b, seed, chunk=64):
    import jax
    import jax.numpy as jnp

    from fpv.perception import labels as L

    render = jax.jit(L.make_renderer())
    scenes = L.scene_pool(1024, seed)
    ns = len(scenes["tgt"])
    key = jax.random.key(seed)
    out = []
    for i in range(0, len(b["pos"]), chunk):
        j = np.arange(i, min(i + chunk, len(b["pos"])))
        jp = np.r_[j, np.full(chunk - len(j), j[-1])]
        key, k = jax.random.split(key)
        si = b["episode"][jp] % ns
        img, _ = render(*(jnp.asarray(b[x][jp]) for x in ("pos", "quat", "gates_pos", "gates_yaw")),
                        {x: jnp.asarray(v[si]) for x, v in scenes.items()}, k)  # fmt: skip
        out.append(np.asarray(img)[: len(j)])
    return np.concatenate(out)


def short(R):
    ng = R["next_gate"]
    return dict(frames=R["frames"], visible_gates=R["visible_gates"], miss=R["miss_rate"], miss_excl_hidden70=R["miss_rate_excl_hidden_70pct"],
                fp_per_frame=R["fp_per_frame"], err_median=R["corner_err_px"].get("median"), err_p90=R["corner_err_px"].get("p90"),
                next_visible_frac=ng["visible_frac"], next_detected=ng["detection_rate"], next_err_median=ng["corner_err_px"].get("median"),
                next_err_p90=ng["corner_err_px"].get("p90"),
                by_distance={k: dict(miss=v["miss"], err_median=v["err"].get("median")) for k, v in R["by_distance"].items()})  # fmt: skip


def mode_detect(a):
    from eval_detector import evaluate_set

    from fpv.perception.detector import load_detector

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = load_detector(a.detector)
    report = dict(detector=a.detector, threshold=a.thr, sets={})
    fn = out / "detect.json"
    if fn.exists():
        report = json.loads(fn.read_text())
    for spec in a.sets:
        name, d = spec.split("=")
        b = load_shards(d.split(","), a.max_frames)
        t = time.time()
        res = detect_frames(det, b["frames"])
        res["hidden"] = hidden_frac(b)
        R = evaluate_set(name, b, res, a.thr, out)
        report["sets"][name] = dict(unreal=short(R), unreal_full=R)
        log(f"{name} unreal: {json.dumps(short(R))} ({time.time() - t:.0f} s)")
        if a.synth:
            fs = synth_frames(b, 9001)
            rs = detect_frames(det, fs)
            rs["hidden"] = res["hidden"]
            Rs = evaluate_set(name + "_synth", b, rs, a.thr, out)
            report["sets"][name]["synth_same_poses"] = short(Rs)
            log(f"{name} synth same poses: {json.dumps(short(Rs))}")
        if a.tilt_split:  # upright vs inverted (body z below the horizon, tilt > 90 deg)
            q = b["quat"].astype(np.float64)
            cz = 1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2)
            for part, mk in (("upright", cz >= 0), ("inverted", cz < 0)):
                if mk.sum() < 20:
                    continue
                bs = {k: v[mk] for k, v in b.items()}
                rs_ = {k: v[mk] for k, v in res.items()}
                Rp = evaluate_set(f"{name}_{part}", bs, rs_, a.thr, out)
                report["sets"][name][part] = short(Rp)
                log(f"{name} {part}: {json.dumps(short(Rp))}")
        fn.write_text(json.dumps(report, indent=1, default=float))
        if a.sheet:
            failure_sheet_ue(b, res, a.thr, out / f"failures_{name}.png")


def failure_sheet_ue(b, res, thr, path, seed=0):
    import matplotlib

    from fpv.perception import detector as D
    from fpv.perception import labels as L

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = L.gate_labels(*(b[k].astype(np.float64) for k in ("pos", "quat", "gates_pos", "gates_yaw")))
    mt = D.match(res["corners"], res["conf"], lab, thr)
    rng = np.random.default_rng(seed)
    miss = np.flatnonzero((lab["visible"] & (mt["det_of_gate"] < 0) & (res["hidden"] < 0.7)).any(1))
    fp = np.flatnonzero(mt["fp"].any(1))
    ok = np.flatnonzero(~mt["fp"].any(1) & ~(lab["visible"] & (mt["det_of_gate"] < 0)).any(1) & (lab["visible"].sum(1) >= 2))
    pick = [("miss", f) for f in rng.permutation(miss)[:4]] + [("false det", f) for f in rng.permutation(fp)[:4]]
    pick += [("correct", f) for f in rng.permutation(ok)[: 12 - len(pick)]]
    fig, axs = plt.subplots(3, 4, figsize=(20, 8.9))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (kind, f) in zip(axs.ravel(), pick):
        ax.imshow(b["frames"][f], extent=(-0.5, L.W - 0.5, L.H - 0.5, -0.5))
        for g in range(6):
            if lab["visible"][f, g] or lab["dontcare"][f, g]:
                q = lab["corners"][f, g]
                ax.plot(*np.r_[q, q[:1]].T, "-" if lab["visible"][f, g] else ":", color="white", lw=0.8)
        for s in range(res["conf"].shape[1]):
            if res["conf"][f, s] >= thr:
                q = res["corners"][f, s]
                ax.plot(*np.r_[q, q[:1]].T, "-", color="red", lw=0.8)
                ax.text(*q.min(0), f"{res['conf'][f, s]:.2f}", color="red", fontsize=6)
        ax.set_xlim(-20.5, L.W + 19.5)
        ax.set_ylim(L.H + 19.5, -20.5)
        ax.set_title(f"{kind} (frame {f})", fontsize=8)
    fig.suptitle("Unreal frames: white = labels (dotted don't-care), red = detections >= threshold", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ============================================================================ closed loop
def make_parts(env, act, det, thr):
    """Jitted pieces of one control step: pre (policy action + physics + IMU + records), detect, post."""
    import jax
    import jax.numpy as jnp

    from eval_stage5 import _DIAG_DT
    from fpv.env import quat_to_mat
    from fpv.env_est import EstEnvState
    from fpv.perception import estimator, imu
    from fpv.perception.detector import decode

    n = env.cfg.n_envs

    @jax.jit
    def pre(state, full, alive):
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
            vel=s.vel[:, 0], ang_vel=s.ang_vel[:, 0], action=action, target=race.target.astype(jnp.int8), alive=alive,
            wrong=(est.nxt != race.target) & alive,
            pose_wrong=(jnp.linalg.norm(c_est - c_true, axis=-1) > jnp.maximum(0.5, 0.15 * d_true)) & alive,
            next_age=jnp.take_along_axis(est.gage, est.nxt[:, None], 1)[:, 0].astype(jnp.int16),
            locked=est.locked & alive, dist_target=d_true, est_nxt=est.nxt.astype(jnp.int8),
            pos=s.pos[:, 0], quat=s.quat[:, 0],
        )  # fmt: skip
        key, kr, ki, kd, ks, kn = jax.random.split(state.key, 6)
        race2, reward, ev = env.race.step_core(race, action)
        s1 = race2.data.states
        imu_st, gyro, acc = imu.imu_measure(state.imu, ki, s.quat[:, 0], s1.quat[:, 0], s.vel[:, 0], s1.vel[:, 0],
                                            s.ang_vel[:, 0], s1.ang_vel[:, 0], env.dt, env.ecfg.imu)  # fmt: skip
        ended = alive & (ev["terminated"] | ev["truncated"])
        alive2 = alive & ~(ev["terminated"] | ev["truncated"])
        outcome = jnp.where(ev["crash_gate"], 1, jnp.where(ev["ground"], 2, jnp.where(ev["oob"], 3, 0)))
        rec |= dict(passed=ev["passed"] & alive, in_fov=ev["in_fov"], reward=reward, outcome=outcome.astype(jnp.int8),
                    ended=ended, d_valid=alive2)  # fmt: skip
        return race2, imu_st, gyro, acc, key, kd, alive2, rec, s1.pos[:, 0], s1.quat[:, 0]

    @jax.jit
    def detect(frames):
        o = decode(det.cfg, det.net.apply(det.params, frames), snap=det.snap)
        return o["corners"], o["conf"]

    @jax.jit
    def post(state, race2, imu_st, gyro, acc, key, kd, corners, conf):
        env._inj = (corners, conf)
        det_st, est_st, obs, pinfo = env._perceive(race2, state.det, state.est, state.latency, gyro, acc,
                                                   jnp.full(n, env.dt), kd, state.start_idx)  # fmt: skip
        full = jnp.concatenate([obs, env.race.obs(race2)], -1)
        st2 = EstEnvState(race=race2, det=det_st, imu=imu_st, est=est_st, latency=state.latency, est_obs=obs,
                          start_idx=state.start_idx, key=key)  # fmt: skip
        diag = {"d_" + k: v.astype(_DIAG_DT.get(k, jnp.float32)) for k, v in pinfo["diag"].items()}
        nxt = est_st.nxt
        Pc, Py, _, T2 = estimator.gate_estimates(est_st, race2.track_pos, race2.track_yaw, env.ecfg.est)
        c = jnp.take_along_axis(Pc, nxt[:, None, None], 1)[:, 0]
        y = jnp.take_along_axis(Py, nxt[:, None], 1)[:, 0]
        belief = estimator._apply_T(T2[:, None], track.gate_corners(c, y))
        return st2, full, diag, dict(corners=pinfo["corners"], conf=pinfo["conf"], est_nxt=nxt, belief_corners_b=belief)

    return pre, detect, post


def make_env(env_cfg, est_cfg, m, thr, known_start, circuit=None):
    import jax.numpy as jnp

    from eval_stage5 import Stage5Env
    from fpv.env_est import condition
    from fpv.perception import estimator

    class Stage6Env(Stage5Env):
        """Stage5Env whose detections are injected (`_inj`, computed on the host from Unreal frames + detector)."""

        _inj = None

        def _perceive(self, race, det_st, est_st, latency, gyro, acc, dt, key, start_idx):
            corners, conf = self._inj
            conf = jnp.where(conf >= self.thr, conf, 0.0)
            est_st, einfo = estimator.est_update(est_st, corners, conf, gyro, acc, race.track_pos, race.track_yaw, dt,
                                                 self.ecfg.est, start_idx)  # fmt: skip
            obs = estimator.observation(est_st, race.track_pos, race.track_yaw, race.prev_action, self.ecfg.est)
            from eval_stage5 import det_diag

            s = race.data.states
            diag = det_diag(s.pos[:, 0], s.quat[:, 0], race.gates_pos, race.gates_yaw, race.target, corners, conf)
            return det_st, est_st, obs, dict(est=einfo, corners=corners, conf=conf, diag=diag)

    ecfg = dataclasses.replace(condition("x1"), known_start=known_start, est=dataclasses.replace(est_cfg, conf_min=thr))
    env = Stage6Env(dataclasses.replace(env_cfg, n_envs=m), ecfg, "true", thr=thr, circuit=circuit)
    return env, ecfg


class SynthFrames:
    """Plumbing check of the host loop: stage 5's synthetic frames (ray caster, the flight's look from the stage-5
    scene pool, the same per-step render key as `Stage5Env._frames_detect` for batches <= 256) instead of Unreal."""

    restarts = 0

    def __init__(self, n):
        import jax

        from fpv.perception import labels as L

        self.n, self.scene = n, None
        render = L.make_renderer()
        self._f = jax.jit(lambda p, q, gp, gy, sc, k: render(p, q, gp, gy, sc, jax.random.split(k, 1)[0])[0])

    def set_scene(self, scene):
        import jax.numpy as jnp

        self.scene = {k: jnp.asarray(v) for k, v in scene.items()}

    def set_gates(self, slots, gp, gy):
        import jax.numpy as jnp

        self.gp, self.gy = jnp.asarray(gp, jnp.float32), jnp.asarray(gy, jnp.float32)

    def frames(self, pos, quat, key):
        return self._f(pos, quat, self.gp, self.gy, self.scene, key)

    def close(self):
        pass


class Pipeline:
    """The chain on one batch of m flights, one Unreal slot per flight."""

    def __init__(self, env, act, det, thr, r):
        self.env, self.r, self.thr = env, r, thr
        self.pre, self.detect, self.post = make_parts(env, act, det, thr)
        self.t = dict(physics=0.0, render=0.0, detect=0.0, estimator_policy=0.0, steps=0)

    def detect_np(self, frames):
        import jax

        c, f = self.detect(frames)
        return jax.device_get((c, f))

    def start(self, race, det_st, imu_st, lat, sidx, key):
        from eval_stage5 import finish_reset

        s = race.data.states
        pos, quat = np.asarray(s.pos[:, 0], np.float64), np.asarray(s.quat[:, 0], np.float64)
        self.r.set_gates(np.arange(self.r.n), np.asarray(race.gates_pos), np.asarray(race.gates_yaw))
        if isinstance(self.r, SynthFrames):
            import jax

            k4 = jax.random.split(jax.random.fold_in(key, 0xE57), 5)[3]
            fr = self.r.frames(s.pos[:, 0], s.quat[:, 0], k4)
        else:
            fr = self.r.render(pos, quat)
        c, f = self.detect_np(fr)
        self.env._inj = (c, f)
        return finish_reset(self.env, race, det_st, imu_st, lat, sidx, key), fr

    def fly(self, state, full, steps, keep=(), log_every=250):
        import jax
        import jax.numpy as jnp

        n = self.env.cfg.n_envs
        alive = jnp.ones(n, bool)
        recs, kept = [], {k: [] for k in ("frames", "corners", "conf", "est_nxt", "belief_corners_b")}
        tt = time.time()
        for t in range(steps):
            t0 = time.perf_counter()
            race2, imu_st, gyro, acc, key, kd, alive2, rec, pos_d, quat_d = self.pre(state, full, alive)
            pos, quat, al = jax.device_get((pos_d, quat_d, alive2))
            t1 = time.perf_counter()
            if isinstance(self.r, SynthFrames):
                fr = self.r.frames(pos_d, quat_d, kd)
            else:
                fr = self.r.render(np.asarray(pos, np.float64), np.asarray(quat, np.float64), active=al)
            t2 = time.perf_counter()
            c, f = self.detect(jnp.asarray(fr))
            jax.block_until_ready(f)
            t3 = time.perf_counter()
            state, full, diag, extra = self.post(state, race2, imu_st, gyro, acc, key, kd, c, f)
            rec = jax.device_get(rec | diag)
            t4 = time.perf_counter()
            if al.any() and t >= 3:  # the first steps include compilation
                self.t["physics"] += t1 - t0
                self.t["render"] += t2 - t1
                self.t["detect"] += t3 - t2
                self.t["estimator_policy"] += t4 - t3
                self.t["steps"] += 1
            recs.append(rec)
            if keep:
                ex = jax.device_get(extra)
                kept["frames"].append(fr[list(keep)])
                for k in ("corners", "conf", "est_nxt", "belief_corners_b"):
                    kept[k].append(np.asarray(ex[k])[list(keep)])
            alive = alive2
            if not al.any():
                # every flight ended: the remaining steps only repeat the frozen record (alive False everywhere)
                for _ in range(t + 1, steps):
                    recs.append({k: (np.zeros_like(v) if v.dtype == bool else v) for k, v in rec.items()})
                break
            if log_every and (t + 1) % log_every == 0:
                log(f"    step {t + 1}/{steps}: {int(al.sum())}/{n} flying, {(t + 1) / (time.time() - tt):.1f} steps/s")
        rec = {k: np.stack([r_[k] for r_ in recs]) for k in recs[0]}
        rec = {k: (v.astype(np.float32) if v.dtype in (np.float16, np.float64) else v) for k, v in rec.items()}
        kept = {k: np.stack(v, 1) for k, v in kept.items() if v} if keep else None
        return rec, kept


def per_flight(rec, dt):
    """Per-flight outcome (for pairing with stage 5 on the same flights)."""
    alive, passed, ended = rec["alive"], rec["passed"], rec["ended"]
    return dict(gates=passed.sum(0).astype(int).tolist(), steps=alive.sum(0).astype(int).tolist(),
                outcome=(rec["outcome"] * ended).sum(0).astype(int).tolist())  # fmt: skip


def mode_fly(a):
    import jax

    from eval_est import make_policy  # noqa: F401  (CUDA_VISIBLE_DEVICES handling)
    from eval_stage5 import analyse, policy_and_est, slice_tree
    from fpv.env_est import EstEnv, condition
    from fpv.perception.detector import load_detector

    a.est, a.source, a.cond, a.imu_scale = [], "detector", "x1", None
    act, env_cfg, est_cfg = policy_and_est(a)
    dt = 1 / env_cfg.policy_hz
    det = load_detector(a.detector)
    out = Path(a.out)
    (out / "rec").mkdir(parents=True, exist_ok=True)
    blocks = a.block.split(",")
    names = list(track.SCENARIOS)
    B = a.batch
    r = SynthFrames(a.synth_batch) if a.synth_batch else renderer(a.ue)
    m = r.n
    tag = ("_nominal" if a.nominal else "") + ("_known" if a.known_start else "")
    try:
        cache = {}
        for block in blocks:
            reset_ecfg = dataclasses.replace(condition("x1"), known_start=a.known_start,
                                             est=dataclasses.replace(est_cfg, conf_min=a.thr))  # fmt: skip
            a0, a1 = (int(x) for x in a.idx.split(":"))
            if a.nominal:
                # the camera students' nominal protocol (eval_student / student_flight_video): FlightBatch with nominal
                # gates and physics on the layout, seed --fb-seed; perception state drawn as state_from_race does
                from fpv.closed_loop import FlightBatch
                from fpv.perception import detnoise, imu

                n_full, off = a1, 0
                fb = FlightBatch(n_full, a.teacher, seed=a.fb_seed, randomize=False, scenario=track.resolve(block))
                fb.reset()
                env, _ = make_env(fb.cfg, est_cfg, m, a.thr, a.known_start)
                key = jax.random.key(a.fb_seed * 100 + 1)
                k1, k2, k3, _, _ = jax.random.split(jax.random.fold_in(key, 0xE57), 5)
                ds, is_, lat = env._scales(k1, n_full)
                race_full = fb._state
                det_full = detnoise.det_init(k2, ds, env.race.n_gates)
                imu_full = imu.imu_init(k3, is_, env.ecfg.imu)
                sidx_full = jax.numpy.where(a.known_start, race_full.target, -1)
                tmpl, _ = jax.jit(env.race.reset)(key)
                idx_all = np.arange(a0, a1)
            else:
                if block == "random":
                    n_full, key, circ_full, off = B, jax.random.key(a.seed), None, 0
                else:
                    layouts = [track.circuit_arrays(track.resolve(nm)) for nm in names]
                    circ_full = (np.repeat(np.stack([p for p, _ in layouts]), B, 0), np.repeat(np.stack([y for _, y in layouts]), B, 0))
                    n_full, key, off = len(names) * B, jax.random.key(a.seed + 20), names.index(block) * B
                ck = (block == "random", a.known_start)
                if ck not in cache:
                    full_env = EstEnv(dataclasses.replace(env_cfg, n_envs=n_full), reset_ecfg, circuit=circ_full)
                    cache = {ck: jax.jit(full_env.reset)(key)[0]}
                    del full_env
                st_full = cache[ck]
                race_full, det_full, imu_full, lat, sidx_full = st_full.race, st_full.det, st_full.imu, st_full.latency, st_full.start_idx
                idx_all = np.arange(a0, a1) + off
                circ = None if circ_full is None else (circ_full[0][idx_all[:m]], circ_full[1][idx_all[:m]])
                env, _ = make_env(env_cfg, est_cfg, m, a.thr, a.known_start, circ)
                tmpl, _ = jax.jit(env.race.reset)(key)
            pipe = Pipeline(env, act, det, a.thr, r)
            for bi, s0 in enumerate(range(0, len(idx_all), m)):
                fn = out / "rec" / f"{block}{tag}_{a0 + s0:05d}.npz"
                if fn.exists():
                    continue
                idx = idx_all[s0 : s0 + m]
                pad = np.r_[idx, np.full(m - len(idx), idx[-1])]
                if isinstance(r, SynthFrames):
                    from fpv.perception import labels as L

                    r.set_scene({k: v[pad] for k, v in L.scene_pool(n_full, a.seed + (0 if block == "random" else 20)).items()})
                race = slice_tree(race_full, tmpl, pad, n_full)
                sl = lambda t: jax.tree.map(lambda x: x[pad], t)  # noqa: E731
                t0 = time.time()
                (state, full), _ = pipe.start(race, sl(det_full), sl(imu_full), lat[pad], sidx_full[pad], key)
                rec, _ = pipe.fly(state, full, a.steps)
                rec = {k: v[:, : len(idx)] for k, v in rec.items()}
                np.savez_compressed(fn, **rec, idx=idx - off)
                g = rec["passed"].sum(0)
                log(f"[{block}{tag}] flights {a0 + s0}-{a0 + s0 + len(idx) - 1}: gates {g.tolist()} ({time.time() - t0:.0f} s, "
                    f"restarts {r.restarts})")  # fmt: skip
                timing = {k: (v / max(pipe.t["steps"], 1) * 1000 if k != "steps" else v) for k, v in pipe.t.items()}
                (out / f"timing{tag}_{block}.json").write_text(json.dumps(dict(ms_per_step=timing, slots=m), indent=1))
                if a.stop_at and time.time() > a.stop_at:
                    log("stop-at reached")
                    break
            report_block(out, block, tag, dt, a, analyse)
            if a.stop_at and time.time() > a.stop_at:
                break
    finally:
        r.close()


def report_block(out, block, tag, dt, a, analyse):
    files = sorted((out / "rec").glob(f"{block}{tag}_[0-9]*.npz"))
    if not files:
        return
    zs = [np.load(f) for f in files]
    rec = {k: np.concatenate([z[k] for z in zs], 1) for k in zs[0].files if k != "idx"}
    idx = np.concatenate([z["idx"] for z in zs])
    o = np.argsort(idx)
    rec = {k: v[:, o] for k, v in rec.items()}
    res = dict(mode="fly_unreal", block=block, known_start="_known" in tag, nominal="_nominal" in tag,
               fb_seed=a.fb_seed if "_nominal" in tag else None, detector=a.detector, thr=a.thr, policy=a.policy,
               seed=a.seed if block == "random" else a.seed + 20, batch=a.batch, idx=idx[o].tolist(), flights=int(len(idx)),
               steps=a.steps, renderer=json.loads(a.ue))  # fmt: skip
    res |= analyse(rec, dt)
    res["per_flight"] = per_flight(rec, dt)
    (out / f"{block}{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"[{block}{tag}] {len(idx)} flights: 3 laps {res['frac_3_laps']:.4f} 1 lap {res['frac_1_lap']:.4f} crash {res['crash_rate']:.4f} "
        f"next det {res['det_stats']['next_detected_frac_when_visible']:.4f} err p50 {res['det_stats']['next_err_px'].get('p50')}")  # fmt: skip


def mode_report(a):
    from eval_stage5 import analyse

    out = Path(a.out)
    for block in a.block.split(","):
        report_block(out, block, ("_nominal" if a.nominal else "") + ("_known" if a.known_start else ""), 1 / 50, a, analyse)


# ============================================================================ video
def mode_video(a):
    import jax

    import flight_video
    from eval_stage5 import policy_and_est, state_from_race
    from fpv.closed_loop import FlightBatch
    from fpv.perception.detector import load_detector

    a.est, a.source, a.cond, a.imu_scale = [], "detector", "x1", None
    act, _, est_cfg = policy_and_est(a)
    det = load_detector(a.detector)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.scenario
    gates = track.resolve(name)
    gp, gy = track.circuit_arrays(gates)
    r = renderer(a.ue)
    n = r.n
    try:
        fb = FlightBatch(n, a.teacher, seed=a.fb_seed, randomize=False, scenario=gates)
        fb.reset()
        race = fb._state
        env, _ = make_env(fb.cfg, est_cfg, n, a.thr, a.known_start)
        pipe = Pipeline(env, act, det, a.thr, r)

        # state_from_race with the injected first frame
        s = race.data.states
        r.set_gates(np.arange(n), np.asarray(race.gates_pos), np.asarray(race.gates_yaw))
        fr0 = r.render(np.asarray(s.pos[:, 0], np.float64), np.asarray(s.quat[:, 0], np.float64))
        env._inj = pipe.detect_np(fr0)
        state, full = state_from_race(env, race, jax.random.key(a.fb_seed * 100 + 1))
        rec, kept = pipe.fly(state, full, a.steps, keep=(0,))
    finally:
        r.close()
    dt = 1 / 50
    slot = 0
    T = int(rec["alive"][:, slot].sum())
    pos, quat = rec["pos"][:T, slot], rec["quat"][:T, slot]
    passed = rec["passed"][:T, slot]
    outcome = int((rec["outcome"][:, slot] * rec["ended"][:, slot]).sum())
    OUT = {0: "time_limit", 1: "gate_crash", 2: "ground", 3: "out_of_bounds"}
    kind = "known first gate" if a.known_start else "unknown first gate"
    rec_path = flight_video.save_record(out / "flight.npz", pos=pos, quat=quat, action=rec["action"][:T, slot],
                                        gates_pos=gp, gates_yaw=gy, passed=passed, target=rec["target"][:T, slot],
                                        name=f"{name}: perception pipeline (detector + estimator + policy)", hz=50)  # fmt: skip
    fr = kept["frames"][0, :T]
    green = ((fr[..., 1].astype(np.int16) > 100) & (fr[..., 1] > 1.5 * np.maximum(fr[..., 0], fr[..., 2]))).sum((1, 2))
    np.savez_compressed(out / "pipeline_fpv.npz", frames=fr, corners=kept["corners"][0, :T], conf=kept["conf"][0, :T],
                        est_nxt=kept["est_nxt"][0, :T], target=rec["target"][:T, slot])  # fmt: skip
    # teacher from the same spawn (same FlightBatch seed, slot 0)
    tb = FlightBatch(n, a.teacher, seed=a.fb_seed, randomize=False, scenario=gates)
    st = tb.reset()
    tpos = [st["pos"][slot]]
    tg = 0
    for _ in range(a.steps):
        st = tb.step(st["teacher_action"])
        if st["done"][slot]:
            break
        tpos.append(st["pos"][slot])
        tg += int(st["passed"][slot])
    tpos = np.stack(tpos)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(tpos[:, 0], tpos[:, 1], color="0.35", lw=1.0, ls="--", label=f"teacher (state based), {tg} gates")
    ax.plot(pos[:, 0], pos[:, 1], color="tab:blue", lw=1.0, alpha=0.9, label=f"pipeline (camera + IMU), {int(passed.sum())} gates in {T / 50:.1f} s")
    ax.plot(*pos[0, :2], "o", color="tab:green", label="start")
    if outcome > 0:
        ax.plot(*pos[-1, :2], "x", color="tab:red", ms=10, mew=2, label=f"crash ({OUT[outcome]})")
    for g in range(len(gy)):
        _, left = track.gate_axes(gy[g])
        for lo, hi in ((track.GATE_OPENING / 2, track.GATE_OUTER / 2), (-track.GATE_OUTER / 2, -track.GATE_OPENING / 2)):
            ax.plot([gp[g, 0] + lo * left[0], gp[g, 0] + hi * left[0]], [gp[g, 1] + lo * left[1], gp[g, 1] + hi * left[1]], color="m", lw=4)
        ax.annotate(f"G{g + 1}", gp[g, :2], textcoords="offset points", xytext=(8, 8), color="m")
    ax.set_aspect("equal")
    ax.set_title(f"{name}, top-down: perception pipeline vs teacher")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.savefig(out / "trajectory_pipeline_vs_teacher.png", dpi=130, bbox_inches="tight")
    video = None
    if not a.record_only:
        det_overlay_video(out, fr, kept, rec, T, slot, name, a.thr)
        gpus = json.loads(a.ue)["gpus"]
        video = flight_video.make_videos([rec_path], out, ["fpv", "chase", "overview"], "grid", gpus, a.video_ipg, "stock",
                                         sheet_name="sheet.jpg", highlight=())[0]  # fmt: skip
    info = dict(pipeline=dict(detector=a.detector, thr=a.thr, policy=a.policy, first_gate=kind), teacher=a.teacher, scenario=name,
                fb_seed=a.fb_seed, fb_slots=n, slot=slot, steps=T, seconds=T / 50, gates=int(passed.sum()), outcome=OUT[outcome],
                laps=int(passed.sum()) // 6, record=str(rec_path), video=video, pick="first", highlight_target=False,
                pipeline_frames_with_green_gate=float((green >= 4).mean()), teacher_gates_same_spawn=tg,
                next_gate_detected_when_visible=float(rec["d_next_det"][:T, slot][rec["d_next_vis"][:T, slot]].mean()),
                all_slots_gates=rec["passed"].sum(0).astype(int).tolist())  # fmt: skip
    (out / "flight.json").write_text(json.dumps(info, indent=1, default=str))
    log(json.dumps({k: info[k] for k in ("scenario", "steps", "gates", "laps", "outcome", "pipeline_frames_with_green_gate")}))


def det_overlay_video(out, fr, kept, rec, T, slot, name, thr):
    """FPV frames the detector saw (Unreal, 320x180, no highlight), upscaled 4x, with the detections above the
    threshold drawn in red with their confidence; caption: gates passed, true next gate, estimator's next gate."""
    from PIL import Image, ImageDraw

    import flight_video

    S = 4
    font, small = flight_video._font(22), flight_video._font(16)
    w = flight_video.Writer(str(out / "flight_fpv_detections.mp4"), 50)
    for t in range(T):
        img = Image.fromarray(fr[t]).resize((fr.shape[2] * S, fr.shape[1] * S), Image.NEAREST)
        d = ImageDraw.Draw(img)
        for k in range(kept["conf"].shape[2]):
            c = float(kept["conf"][0, t, k])
            if c < thr:
                continue
            q = (kept["corners"][0, t, k] + 0.5) * S - 0.5
            d.line([tuple(p) for p in np.r_[q, q[:1]]], fill=(255, 40, 40), width=3)
            d.text((float(q[:, 0].min()), float(q[:, 1].min()) - 20), f"{c:.2f}", fill=(255, 40, 40), font=small)
        txt = (f"{name}  t {t / 50:5.2f} s  gates {int(rec['passed'][: t + 1, slot].sum())}  next gate G{int(rec['target'][t, slot]) + 1}"
               f" (estimator: G{int(kept['est_nxt'][0, t]) + 1})")  # fmt: skip
        d.text((12, 8), txt, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        d.text((12, img.height - 28), "detector input: Unreal frame, 320x180, no highlight; red = detected gates (confidence)",
               fill=(255, 255, 255), font=small, stroke_width=2, stroke_fill=(0, 0, 0))  # fmt: skip
        w.add(np.asarray(img))
    w.close()

# ============================================================================ side view / record videos
SIDE_SIZE = (1280, 560)


def side_video(rec_path, out, gpu, caption, dist=8.5):
    """Unreal SIDE view of a flight record (fixed camera on the -y side looking +y, level, hfov 80, the quadrotor
    model 2x size, gates plain), with the last 1.5 s of the path and the tilt drawn on top -> <out>/flight_side.mp4;
    then <out>/flight_combined_side.mp4 = flight_combined.mp4 (published layout) above the side view."""
    from PIL import Image, ImageDraw

    import flight_video
    from fpv import render_airsim as ra

    rec = flight_video.load_record(rec_path)
    pos, quat = rec["pos"].astype(np.float64), rec["quat"].astype(np.float64)
    T = len(pos)
    gp = rec["gates_pos"].astype(np.float64)
    pts = np.concatenate([pos, gp])
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    cam = np.array([c[0], lo[1] - dist, max(c[2], 1.0)])
    cq = ra.look_at_quat(cam, cam + np.array([0.0, 1.0, 0.0]))
    W, H = SIDE_SIZE
    r = ra.AirsimRenderer([gpu], 1, W, H, "stock", gates_pos=gp, gates_yaw=rec["gates_yaw"].astype(np.float64),
                          hfov_deg=80.0, mount_pitch_deg=0.0, drone_scale=ra.DRONE_SCALE, startup_timeout=400)  # fmt: skip
    try:
        fr = r.render_trajectories([0], np.repeat(cam[None, None], T, 1), np.repeat(cq[None, None], T, 1),
                                   drone_pos=pos[None], drone_quat=quat[None])[0]  # fmt: skip
    finally:
        r.close()
    uv = np.stack([ra.project_points(p[None], cam, cq, W, H, 80.0)[0][0] for p in pos])
    cz = 1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
    tilt = np.degrees(np.arccos(np.clip(cz, -1, 1)))
    font = flight_video._font(18)
    w = flight_video.Writer(str(out / "flight_side.mp4"), int(rec["hz"]))
    hz = int(rec["hz"])
    for t in range(T):
        img = Image.fromarray(fr[t])
        d = ImageDraw.Draw(img)
        tr = uv[max(0, t - int(1.5 * hz)) : t + 1]
        if len(tr) > 1:
            d.line([tuple(p) for p in tr], fill=(255, 220, 0), width=2)
        x, y = uv[t]
        d.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 220, 0), width=2)
        d.text((12, 8), f"SIDE VIEW  {caption}  t {t / hz:5.2f} s  gates {int(rec['gates_cum'][t])}  tilt {tilt[t]:5.1f} deg"
               + ("  INVERTED" if cz[t] < 0 else ""), fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))  # fmt: skip
        w.add(np.asarray(img))
    w.close()
    import imageio.v2 as iio

    comb = iio.get_reader(str(out / "flight_combined.mp4"))
    side = iio.get_reader(str(out / "flight_side.mp4"))
    w = flight_video.Writer(str(out / "flight_combined_side.mp4"), hz)
    for a_, b_ in zip(comb, side):
        if a_.shape[1] != W:
            a_ = np.asarray(Image.fromarray(a_).resize((W, int(a_.shape[0] * W / a_.shape[1]))))
        w.add(np.concatenate([a_, b_], 0))
    w.close()
    return dict(side_camera=dict(pos=cam.round(3).tolist(), quat_xyzw=np.asarray(cq).round(4).tolist(), hfov_deg=80.0),
                frames=T, max_tilt_deg=float(tilt.max()), inverted_frames=int((cz < 0).sum()))


def mode_side(a):
    """Side view for an existing video directory (flight.npz + flight_combined.mp4 of `video` / `record-video`)."""
    out = Path(a.out)
    info = side_video(out / "flight.npz", out, json.loads(a.ue)["gpus"][0], a.caption)
    (out / "flight_side.json").write_text(json.dumps(info, indent=1))
    log(json.dumps(info))


def mode_record_video(a):
    """Published-layout Unreal video of an existing flight record (e.g. the state-based acro teacher's), FPV plain,
    captioned with --caption, + side view. --record: npz with pos, quat (xyzw), action, gates_pos, gates_yaw."""
    import flight_video

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.record)
    T = int(z["alive"].sum()) if "alive" in z.files else len(z["pos"])
    rec_path = flight_video.save_record(out / "flight.npz", pos=z["pos"][:T], quat=z["quat"][:T], action=z["action"][:T],
                                        gates_pos=z["gates_pos"], gates_yaw=z["gates_yaw"], name=a.caption, hz=50)  # fmt: skip
    gpus = json.loads(a.ue)["gpus"]
    video = flight_video.make_videos([rec_path], out, ["fpv", "chase", "overview"], "grid", gpus, a.video_ipg, "stock",
                                     sheet_name="sheet.jpg", highlight=())[0]  # fmt: skip
    side = side_video(rec_path, out, gpus[0], a.caption)
    r = flight_video.load_record(rec_path)
    info = dict(flown_by=a.caption, source_record=a.record, steps=T, seconds=T / 50, gates=int(r["passed"].sum()),
                laps=int(r["passed"].sum()) // 6, video=video, side=side, highlight_target=False)  # fmt: skip
    (out / "flight.json").write_text(json.dumps(info, indent=1, default=str))
    log(json.dumps({k: info[k] for k in ("flown_by", "steps", "gates", "laps")}))


def mode_replay_video(a):
    """The `video` outputs of a flight flown with `video --record-only`, WITHOUT re-flying: chase / FPV / overview /
    combined + sheet rendered from <out>/flight.npz, and the detection video from <out>/pipeline_fpv.npz (the Unreal
    frames, detections and estimator next gate recorded in flight). Adds the video info to flight.json."""
    import flight_video

    out = Path(a.out)
    info = json.loads((out / "flight.json").read_text())
    rec_path = out / "flight.npz"
    rec = flight_video.load_record(rec_path)
    z = np.load(out / "pipeline_fpv.npz")
    T = len(rec["pos"])
    if len(z["frames"]) != T or info["steps"] != T:
        raise ValueError(f"{out}: record {T} steps, pipeline_fpv {len(z['frames'])}, flight.json {info['steps']}")
    kept = {k: z[k][None] for k in ("corners", "conf", "est_nxt")}
    det_overlay_video(out, z["frames"], kept, dict(passed=rec["passed"][:, None], target=rec["target"][:, None]), T, 0,
                      info["scenario"], info["pipeline"]["thr"])  # fmt: skip
    gpus = json.loads(a.ue)["gpus"]
    info["video"] = flight_video.make_videos([rec_path], out, ["fpv", "chase", "overview"], "grid", gpus, a.video_ipg,
                                             "stock", sheet_name="sheet.jpg", highlight=())[0]  # fmt: skip
    info["video_rendered"] = "after the flight from flight.npz + pipeline_fpv.npz (replay, not re-flown)"
    (out / "flight.json").write_text(json.dumps(info, indent=1, default=str))
    log(json.dumps({k: info[k] for k in ("scenario", "steps", "gates", "laps", "outcome")}))


# ============================================================================ acrobatic chase camera
def _project(cam, q, pts, W, H, hfov_deg):
    """Pinhole of `render_airsim.project_points` (mount pitch 0), broadcast over [...] -> uv [..., 2], depth [...]."""
    from fpv import render_airsim as ra

    R = ra.quat_to_mat(q)
    d = np.asarray(pts, np.float64) - cam
    f, l, u = ((d * R[..., :, i]).sum(-1) for i in range(3))
    th = math.tan(math.radians(hfov_deg) / 2)
    tv = th * H / W
    z = np.where(np.abs(f) < 1e-9, 1e-9, f)
    return np.stack([W / 2 - 0.5 - (W / 2) / th * l / z, H / 2 - 0.5 - (H / 2) / tv * u / z], -1), f


def _inside(uv, depth, W, H, m=0.0, dmin=0.05):
    return (depth > dmin) & (uv[..., 0] >= m * W - 0.5) & (uv[..., 0] <= (1 - m) * W - 0.5) & (uv[..., 1] >= m * H - 0.5) & (
        uv[..., 1] <= (1 - m) * H - 0.5)


def _relevant_gate(rec, hold_s=0.25):
    """Gate of interest per frame: the next gate, or the gate just passed for ``hold_s`` after its pass."""
    T, hz = len(rec["pos"]), rec["hz"]
    tgt = np.asarray(rec["target"] if rec["target"] is not None else np.zeros(T), np.int64).copy()
    tgt[tgt < 0] = 0
    rel = tgt.copy()
    for k in np.flatnonzero(rec["passed"]):
        rel[k : min(T, k + 1 + int(hold_s * hz))] = tgt[k]
    return rel


def acro_chase_camera(rec, W, H, hfov_deg, dt_s=0.1, n_az=48, dists=np.arange(2.0, 4.51, 0.25),
                      heights=np.arange(0.4, 2.41, 0.2), base_s=0.35, look_s=0.3, smooth_s=0.08, h_pref=1.0):
    """Continuous chase camera for acrobatic flights, in the WORLD frame (not tied to the drone's attitude or
    instantaneous velocity). Camera = base + offset: base = the drone position low-passed (``base_s``); offset on a
    grid of world azimuths (``n_az``) x distances x heights. The path is chosen globally by dynamic programming over
    keyframes every ``dt_s``, where each step may only move to a neighbouring grid cell (<= 7.5 deg of azimuth,
    0.25 m of distance, 0.2 m of height per keyframe) and every move costs, so the camera orbits / dollies slowly and
    only when it must. Per keyframe: the drone must be inside the image (6 % margin) and not hidden by a gate bar, the
    camera >= 0.35 m from every gate frame, >= 0.3 m above ground and inside `render_airsim.CAMERA_BOUNDS` (clear of
    the level's blocks); the opening of the gate being passed must be in the image at every pass (hard); otherwise
    the next gate's opening is wanted (soft), and near / behind the slowly filtered heading is mildly preferred.
    The keyframes are interpolated linearly and Gaussian-smoothed (``smooth_s``); the camera looks (level) at a point
    40 % of the way from the drone to its gate of interest, low-passed (``look_s``). Returns cam_pos, cam_quat."""
    from fpv import render_airsim as ra

    pos = rec["pos"].astype(np.float64)
    T, hz = len(pos), float(rec["hz"])
    gp, gy = rec["gates_pos"].astype(np.float64), rec["gates_yaw"].astype(np.float64)
    base = ra._smooth(pos, base_s * hz)
    v = np.gradient(ra._smooth(pos, 0.12 * hz), axis=0) * hz
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    ok = np.linalg.norm(v[:, :2], axis=1) > 1.0
    if ok.any():
        ang = np.interp(np.arange(T), np.flatnonzero(ok), ang[ok])
    head = ra._smooth(ang, 0.6 * hz)
    rel = _relevant_gate(rec)
    dg = gp[rel] - pos
    L = ra._smooth(pos + 0.4 * dg * np.minimum(1.0, 3.0 / np.maximum(np.linalg.norm(dg, axis=1), 1e-6))[:, None], look_s * hz)
    lo, hi = ra.CAMERA_BOUNDS
    az = np.arange(n_az) * 2 * np.pi / n_az
    dists, heights = np.asarray(dists, np.float64), np.asarray(heights, np.float64)
    A, Dn, Hn = len(az), len(dists), len(heights)
    AZ, D_, H_ = (x.ravel() for x in np.meshgrid(az, dists, heights, indexing="ij"))
    off = np.stack([D_ * np.cos(AZ), D_ * np.sin(AZ), H_], -1)
    step = int(round(dt_s * hz))
    keys = np.arange(0, T, step)
    passes = np.flatnonzero(rec["passed"])
    tgt = np.asarray(rec["target"], np.int64)
    def penalty(tt, hard):
        """Cost [states] of the constraints at frame tt (hard: the gate being passed must be in the image)."""
        C = base[tt] + off
        q = ra.look_at_quat(C, np.broadcast_to(L[tt], C.shape))
        uv, dep = _project(C, q, pos[tt], W, H, hfov_deg)
        u = 1e4 * ~_inside(uv, dep, W, H, 0.06, 0.3) + 1e4 * ra.gate_frame_blocks(C, pos[tt], gp, gy, margin=0.02)
        u = u + 1e4 * ((ra.gate_frame_distance(C, gp, gy) < 0.35) | (C[:, 2] < 0.3) | (C < lo).any(-1) | (C > hi).any(-1))
        g = int(tgt[tt]) if hard else int(rel[tt])
        cuv, cdep = _project(C[:, None], q[:, None], track.gate_corners(gp[g], gy[g])[None], W, H, hfov_deg)
        return u + (1e4 if hard else 7.0) * ~_inside(cuv, cdep, W, H, 0.0, 0.2).all(-1)

    unary = np.zeros((len(keys), len(off)))
    for i, t in enumerate(keys):
        near = [k for k in passes if t - step / 2 <= k < t + step / 2]  # pass frames of this keyframe's window
        for tt in [x for x in (t - 2, t, t + 2) if 0 <= x < T]:  # three frames around the keyframe
            unary[i] += penalty(tt, False)
        for tt in near:
            unary[i] += penalty(tt, True)
        unary[i] += 3.0 * 0.5 * (1 + np.cos(AZ - head[t])) + 4.0 * (D_ - dists[0]) + 1.0 * np.abs(H_ - h_pref)
    moves = [(da, dd, dh) for da in (-2, -1, 0, 1, 2) for dd in (-1, 0, 1) for dh in (-1, 0, 1)]

    def solve(unary):
        un = unary.reshape(len(keys), A, Dn, Hn)
        cost = un[0].copy()
        back = np.zeros((len(keys), A, Dn, Hn), np.int8)
        for i in range(1, len(keys)):
            best = np.full(cost.shape, np.inf)
            arg = np.zeros(cost.shape, np.int8)
            for m, (da, dd, dh) in enumerate(moves):
                c = np.roll(cost, da, 0)  # state x reached from x - (da, dd, dh)
                if dd:
                    c = np.roll(c, dd, 1)
                    c[:, 0 if dd > 0 else -1] = np.inf
                if dh:
                    c = np.roll(c, dh, 2)
                    c[:, :, 0 if dh > 0 else -1] = np.inf
                c = c + 2.0 * abs(da) + 6.0 * (abs(da) == 2) + 3.0 * abs(dd) + 3.0 * abs(dh)
                better = c < best
                best = np.where(better, c, best)
                arg = np.where(better, m, arg)
            cost = best + un[i]
            back[i] = arg
        st = np.zeros((len(keys), 3), np.int64)
        st[-1] = np.unravel_index(int(cost.argmin()), cost.shape)
        for i in range(len(keys) - 1, 0, -1):
            da, dd, dh = moves[back[i][tuple(st[i])]]
            st[i - 1] = ((st[i][0] - da) % A, st[i][1] - dd, st[i][2] - dh)
        tt = np.arange(T)
        a_f = ra._smooth(np.interp(tt, keys, np.unwrap(az[st[:, 0]])), smooth_s * hz)
        d_f = ra._smooth(np.interp(tt, keys, dists[st[:, 1]]), smooth_s * hz)
        h_f = ra._smooth(np.interp(tt, keys, heights[st[:, 2]]), smooth_s * hz)
        cam = base + np.stack([d_f * np.cos(a_f), d_f * np.sin(a_f), h_f], -1)
        cam = np.minimum(np.maximum(cam, lo), hi)
        cam[:, 2] = np.maximum(cam[:, 2], 0.3)
        return cam, ra.look_at_quat(cam, L)

    cam, q = solve(unary)
    uv1, dp1 = _project(cam, q, pos, W, H, hfov_deg)
    bad = ra.gate_frame_blocks(cam, pos, gp, gy) | ~_inside(uv1, dp1, W, H)
    log(f"acro_chase_camera: drone hidden or outside the image in {int(bad.sum())} of {T} frames")
    return cam, q


def chase_metrics(rec, cam, q, W, H, hfov_deg):
    from fpv import render_airsim as ra

    pos = rec["pos"].astype(np.float64)
    gp, gy = rec["gates_pos"].astype(np.float64), rec["gates_yaw"].astype(np.float64)
    uv, dep = _project(cam, q, pos, W, H, hfov_deg)
    din = _inside(uv, dep, W, H)
    occ = ra.gate_frame_blocks(cam, pos, gp, gy)
    tgt = np.asarray(rec["target"], np.int64)
    passes = {}
    for k in np.flatnonzero(rec["passed"]):
        g = int(tgt[k])
        cuv, cdep = _project(cam[k], q[k], track.gate_corners(gp[g], gy[g]), W, H, hfov_deg)
        okp = bool(din[k] and not occ[k] and _inside(cuv, cdep, W, H).all())
        passes.setdefault(f"G{g + 1}", []).append(okp)
    fwd = ra.quat_to_mat(q)[:, :, 0]
    ang = np.degrees(np.arccos(np.clip((fwd[1:] * fwd[:-1]).sum(-1), -1, 1))) * rec["hz"]
    spd = np.linalg.norm(np.diff(cam, axis=0), axis=1) * rec["hz"]
    return dict(drone_in_image_frac=round(float(din.mean()), 4), drone_in_and_unoccluded_frac=round(float((din & ~occ).mean()), 4),
                min_gate_frame_distance_m=round(float(ra.gate_frame_distance(cam, gp, gy).min()), 3),
                min_camera_z_m=round(float(cam[:, 2].min()), 3),
                camera_speed_m_s_p50_p99_max=[round(float(x), 2) for x in (np.median(spd), np.percentile(spd, 99), spd.max())],
                look_angular_speed_deg_s_p50_p99_max=[round(float(x), 1) for x in (np.median(ang), np.percentile(ang, 99), ang.max())],
                camera_to_drone_m_p10_p50_p90=[round(float(x), 2) for x in np.percentile(np.linalg.norm(cam - pos, axis=1), [10, 50, 90])],
                passes_ok={g: f"{sum(v)} of {len(v)}" for g, v in sorted(passes.items())},
                passes_ok_all=f"{sum(sum(v) for v in passes.values())} of {sum(len(v) for v in passes.values())}")  # fmt: skip


def mode_rechase(a):
    """Re-render an existing video directory with `acro_chase_camera` (same flight record, nothing re-flown); the
    previous chase / combined / combined_side videos are kept as *_oldcam.mp4. --dry: metrics only."""
    import flight_video
    from fpv import render_airsim as ra

    out = Path(a.out)
    rec_path = out / "flight.npz"
    rec = flight_video.load_record(rec_path)
    W, H = flight_video.PANEL
    hf = ra.TP_HFOV_DEG
    old = ra.chase_camera(rec["pos"], rec["quat"], rec["gates_pos"], rec["gates_yaw"], hz=rec["hz"])
    cam, q = acro_chase_camera(rec, W, H, hf)
    res = dict(before=chase_metrics(rec, old[0], old[1], W, H, hf), after=chase_metrics(rec, cam, q, W, H, hf))
    log(json.dumps(res))
    if a.dry:
        return
    if not a.sheet_only:
        render_rechase(a, out, rec_path, cam, q)
    dive_sheet(out, rec, res)


def render_rechase(a, out, rec_path, cam, q):
    import flight_video
    from fpv import render_airsim as ra

    for s in ("flight_chase", "flight_combined", "flight_combined_side"):
        if (out / f"{s}.mp4").exists() and not (out / f"{s}_{a.keep_suffix}.mp4").exists():
            (out / f"{s}.mp4").rename(out / f"{s}_{a.keep_suffix}.mp4")
    ra.chase_camera = lambda *args, **kw: (cam, q, {"camera": "acro_chase_camera (scripts/stage6_unreal.py)"})
    gpus = json.loads(a.ue)["gpus"]
    flight_video.make_videos([rec_path], out, ["fpv", "chase", "overview"], "grid", gpus, a.video_ipg, "stock",
                             sheet_name="sheet.jpg", highlight=())  # fmt: skip
    side_video(rec_path, out, gpus[0], a.caption)


def dive_sheet(out, rec, res):
    import flight_video

    # dive sheet: chase frames around the first three dives (0.1 s before the G1 pass -> 0.3 s after the G3 pass)
    import imageio.v2 as iio
    from PIL import Image

    rd = iio.get_reader(str(out / "flight_chase.mp4"))
    tgt = np.asarray(rec["target"], np.int64)
    pk = np.flatnonzero(rec["passed"])
    g1 = [k for k in pk if tgt[k] == 0][:3]
    rows = []
    for k in g1:
        k3 = next((j for j in pk if j > k and tgt[j] == 2), k + 60)
        ts = np.linspace(k - 5, min(len(rec["pos"]) - 1, k3 + 15), 8).round().astype(int)
        rows.append(np.concatenate([flight_video.annotate(flight_video._resize(rd.get_data(int(t)), (400, 225)),
                                                          f"t {t / rec['hz']:.2f} s gates {int(rec['gates_cum'][t])}", font_size=13)
                                    for t in ts], 1))  # fmt: skip
    Image.fromarray(np.concatenate(rows, 0)).save(out / "dive_chase_sheet.jpg", quality=85)
    res["dive_sheet_g1_pass_steps"] = [int(k) for k in g1]
    (out / "rechase.json").write_text(json.dumps(res, indent=1))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "frames", "labels", "detect", "fly", "report", "video", "side", "record-video",
                                     "replay-video", "rechase"])  # fmt: skip
    ap.add_argument("--ue", default='{"gpus":[0,1],"instances_per_gpu":2,"look":"stock"}')
    ap.add_argument("--out", required=True)
    # bench
    ap.add_argument("--configs", default="0:1;0:2;0:4")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--look", default="stock")
    # frames / labels
    ap.add_argument("--bank", default=None)
    ap.add_argument("--flights", default=None, help="random|<layout>:seed:n_envs:steps (labels.collect_flights)")
    ap.add_argument("--wide-p", type=float, default=0.5)
    ap.add_argument("--teacher-noise", type=float, default=0.3)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--n", type=int, default=256)
    # detect
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--tilt-split", action="store_true", help="detect: also report upright / inverted frames")
    ap.add_argument("--teacher", default=TEACHER, help="frames (flights teacher:...), video / nominal FlightBatch")
    ap.add_argument("--acro", action="store_true", help="register the acrobatic layouts (fpv.acro) first")
    # fly / video
    ap.add_argument("--detector", default=DET)
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--known-start", action="store_true")
    ap.add_argument("--nominal", action="store_true", help="fly: FlightBatch nominal flights (seed --fb-seed) on the layout")
    ap.add_argument("--synth-batch", type=int, default=0,
                    help="fly: plumbing check, stage 5's synthetic frames instead of Unreal, this many flights per batch")
    ap.add_argument("--block", default="figure8")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--idx", default="0:48")
    ap.add_argument("--stop-at", type=float, default=0.0)
    ap.add_argument("--scenario", default="figure8")
    ap.add_argument("--fb-seed", type=int, default=900002)
    ap.add_argument("--video-ipg", type=int, default=2)
    ap.add_argument("--caption", default="", help="side / record-video: who flies (shown on the video)")
    ap.add_argument("--record", default=None, help="record-video: flight record npz")
    ap.add_argument("--dry", action="store_true", help="rechase: camera metrics only, no rendering")
    ap.add_argument("--keep-suffix", default="oldcam", help="rechase: suffix of the replaced videos")
    ap.add_argument("--sheet-only", action="store_true", help="rechase: metrics + dive sheet from the rendered chase video")
    ap.add_argument("--record-only", action="store_true", help="video: flight record + flight.json only, no videos")
    a = ap.parse_args()
    if a.acro:
        from fpv import acro

        acro.register()
    {"bench": mode_bench, "frames": mode_frames, "labels": mode_labels, "detect": mode_detect, "fly": mode_fly,
     "report": mode_report, "video": mode_video, "side": mode_side, "record-video": mode_record_video,
     "replay-video": mode_replay_video, "rechase": mode_rechase}[a.mode](a)  # fmt: skip


if __name__ == "__main__":
    main()
