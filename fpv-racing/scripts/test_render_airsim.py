"""Tests of fpv.render_airsim.AirsimRenderer on the node (run with the AirSim venv's python).

    /tmp/fpv_airsim/venv/bin/python scripts/test_render_airsim.py --gpus 4 5

Prints and stores (checks/out/render_airsim_test.json):
  * sync under load: every slot active, random per-slot views classified by image content, for
    `render` and for `render_trajectories`; random `active` masks;
  * rollout geometry: teacher rollout poses rendered with their episode's gate layout, magenta gate
    pixels compared with the projection of the target gate (camera model of fpv/track.py);
  * set_gates cost, watchdog recovery, start-up time, throughput table for both looks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpv import track  # noqa: E402
from fpv.render_airsim import GATE_DEPTH, AirsimRenderer  # noqa: E402

OUT = ROOT / "checks" / "out"


# ----------------------------------------------------------------------------- camera model
def quat_to_mat(q):
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def project(points, pos, quat, width, height):
    """Contract-frame world points [N,3] -> pixel coordinates [N,2] (integer = pixel centre), depth [N].

    Camera of fpv/track.py: at the body origin, optical axis = body +x pitched up, x right, y down.
    """
    p = math.radians(track.CAMERA["pitch_up_deg"])
    body_from_cam = np.array([[0, math.sin(p), math.cos(p)], [-1, 0, 0], [0, -math.cos(p), math.sin(p)]])
    pc = (np.asarray(points, float) - np.asarray(pos, float)) @ (quat_to_mat(np.asarray(quat, float)) @ body_from_cam)
    th, tv = track.camera_half_fov_tan()
    z = np.where(np.abs(pc[:, 2]) < 1e-9, 1e-9, pc[:, 2])
    u = width / 2 - 0.5 + (width / 2) / th * pc[:, 0] / z
    v = height / 2 - 0.5 + (height / 2) / tv * pc[:, 1] / z
    return np.stack([u, v], -1), pc[:, 2]


def gate_bars(pos, yaw):
    """The four bar boxes of a gate as [8,3] vertex arrays."""
    n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    l, u = np.array([-n[1], n[0], 0.0]), np.array([0.0, 0.0, 1.0])
    opening, bar = track.GATE_OPENING, (track.GATE_OUTER - track.GATE_OPENING) / 2
    c = opening / 2 + bar / 2
    return [np.array([np.asarray(pos, float) + (cl + a * sl / 2) * l + (cu + b * su / 2) * u + d * GATE_DEPTH / 2 * n
                      for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
            for cl, cu, sl, su in ((0, c, track.GATE_OUTER, bar), (0, -c, track.GATE_OUTER, bar),
                                   (c, 0, bar, opening), (-c, 0, bar, opening))]


def expected_mask(gates_pos, gates_yaw, which, pos, quat, width, height):
    """Silhouette of the listed gates predicted by the camera model."""
    m = np.zeros((height, width), np.uint8)
    for g in which:
        for bar in gate_bars(gates_pos[g], gates_yaw[g]):
            uv, z = project(bar, pos, quat, width, height)
            if z.min() > 0.05 and np.abs(uv).max() < 1e5:
                cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return m.astype(bool)


def gate_mask(img_rgb):
    r, g, b = (img_rgb[..., i].astype(np.int32) for i in range(3))
    return (r > 120) & (b > 90) & (g < 0.75 * np.minimum(r, b))


def bbox(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.min() - 0.5, ys.min() - 0.5, xs.max() + 0.5, ys.max() + 0.5])


class Load(threading.Thread):
    """Samples GPU utilisation and the CPU time of all sim/client processes of a renderer."""

    def __init__(self, r):
        super().__init__(daemon=True)
        self.r, self.gpu, self.halt = r, [], threading.Event()
        self.pids = list(r._sim_pids.values()) + [p.pid for p in r._procs.values()]
        self.t0, self.cpu0 = time.time(), self.cpu()

    def cpu(self):
        tot = 0.0
        for pid in self.pids:
            try:
                f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
                tot += (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
            except OSError:
                pass
        return tot

    def run(self):
        while not self.halt.is_set():
            out = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, self.r.gpus)), "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            try:
                self.gpu.append([[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()])
            except ValueError:
                pass
            self.halt.wait(0.5)

    def summary(self):
        self.halt.set()
        g = np.array(self.gpu) if self.gpu else np.zeros((1, len(self.r.gpus), 2))
        return {"gpu_util_mean_per_gpu": g[..., 0].mean(0).round(1).tolist(),
                "gpu_mem_mib_per_gpu": g[..., 1].max(0).tolist(),
                "cpu_cores_per_slot": round((self.cpu() - self.cpu0) / (time.time() - self.t0) / self.r.n, 2)}


# ----------------------------------------------------------------------------- 1. sync under load
def make_views(k, rng):
    """k body poses whose images differ strongly (positions around the circuit, various headings)."""
    centre = track.circuit_arrays()[0].mean(0)
    pos, quat = [], []
    for i in range(k):
        ang, rad, h = 2 * math.pi * i / k, 3.0 + 4.0 * (i % 3), 0.6 + 1.7 * (i % 4)
        p = centre + np.array([rad * math.cos(ang), rad * math.sin(ang), 0.0])
        p[2] = h
        yaw = ang + math.pi + (i % 2) * math.pi * 0.75 + rng.uniform(-0.3, 0.3)  # towards / away from the centre
        pitch, roll = rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.6)
        cy, sy, cp, sp, cr, sr = (f(a / 2) for a in (yaw, pitch, roll) for f in (math.cos, math.sin))
        quat.append([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy])
        pos.append(p)
    return np.array(pos), np.array(quat)


def small(frames):
    return frames[..., ::4, ::4, :].astype(np.float32)


def classify(frames, refs):
    """frames [n,h,w,3], refs [n,K,h,w,3] (already `small`) -> (view index [n], distance to it [n])."""
    d = np.abs(small(frames)[:, None] - refs).mean((2, 3, 4))
    return d.argmin(1), d.min(1)


def test_sync(r, steps, traj_steps, rng, n_views=8):
    n = r.n
    vpos, vquat = make_views(n_views, rng)
    refs = np.zeros((n, n_views, *small(np.zeros((1, r.height, r.width, 3), np.uint8)).shape[1:]), np.float32)
    full = []
    for k in range(n_views):  # settled reference of each view in each slot
        for _ in range(3):
            f = r.render(np.repeat(vpos[k][None], n, 0), np.repeat(vquat[k][None], n, 0))
        refs[:, k] = small(f)
        full.append(f.astype(np.int16))
    gap = np.array([[np.abs(refs[:, a] - refs[:, b]).mean() for b in range(n_views)] for a in range(n_views)])
    res = {"slots": n, "views": n_views,
           "min_reference_gap_between_views": round(float((gap + 1e9 * np.eye(n_views)).min()), 2)}

    # history independence: a frame right after a jump from another view vs the settled frame of
    # the same view, per pixel, compared with the frame-to-frame noise of a repeated view
    jump, rep = [], []
    for k in range(n_views):
        a = (k + 3) % n_views
        r.render(np.repeat(vpos[a][None], n, 0), np.repeat(vquat[a][None], n, 0))
        first = r.render(np.repeat(vpos[k][None], n, 0), np.repeat(vquat[k][None], n, 0)).astype(np.int16)
        again = r.render(np.repeat(vpos[k][None], n, 0), np.repeat(vquat[k][None], n, 0)).astype(np.int16)
        jump.append(np.abs(first - full[k]))
        rep.append(np.abs(again - first))
    jump, rep = np.stack(jump), np.stack(rep)
    res["history_independence"] = {
        name: {"mean_abs_diff": round(float(x.mean()), 3), "p99.9_abs_diff": float(np.percentile(x, 99.9)),
               "fraction_of_pixels_differing_by_more_than_16": float((x > 16).mean()),
               "worst_frame_mean_abs_diff": round(float(x.mean((2, 3, 4)).max()), 3)}
        for name, x in (("first_frame_after_jump_vs_settled", jump), ("repeated_frame_noise", rep))}
    print("HISTORY", json.dumps(res["history_independence"]), flush=True)

    # render(): every slot active, independent random view per slot per step
    wrong, stale, worst, t0, detail = 0, 0, 0.0, time.time(), []
    idx = rng.integers(0, n_views, (steps, n))
    for s in range(steps):
        got, dist = classify(r.render(vpos[idx[s]], vquat[idx[s]]), refs)
        bad = (got != idx[s]) | (dist > 5.0)  # wrong view, or an image that is none of the views
        wrong += int(bad.sum())
        worst = max(worst, float(dist[~bad].max()) if (~bad).any() else 0.0)
        for i in np.flatnonzero(bad):
            was_prev = bool(s > 0 and got[i] == idx[s - 1, i] and dist[i] <= 5.0)
            stale += was_prev
            if len(detail) < 20:
                detail.append({"step": s, "slot": int(i), "commanded": int(idx[s, i]), "seen": int(got[i]),
                               "dist": round(float(dist[i]), 2), "seen_is_previous_pose": was_prev,
                               "worker_ms": r.last_ms.get(int(i))})
    res["render"] = {"steps_per_slot": steps, "frames": steps * n, "mismatched_frames": wrong,
                     "mismatches_showing_previous_pose": int(stale), "first_mismatches": detail,
                     "max_abs_diff_to_own_reference": round(worst, 3), "restarts": r.restarts,
                     "seconds": round(time.time() - t0, 1)}
    print("SYNC render", json.dumps(res["render"]), flush=True)

    # render() with random active masks: inactive slots must be zeros, active ones still correct
    wrong = nonzero = 0
    m_steps = max(50, steps // 10)
    for s in range(m_steps):
        act, ix = rng.random(n) < 0.6, rng.integers(0, n_views, n)
        f = r.render(vpos[ix], vquat[ix], act)
        got, _ = classify(f, refs)
        wrong += int((got != ix)[act].sum())
        nonzero += int(f[~act].any())
    res["render_masked"] = {"steps": m_steps, "mismatched_active_frames": wrong, "inactive_slots_not_zero": nonzero}
    print("SYNC render_masked", json.dumps(res["render_masked"]), flush=True)

    # render_trajectories(): all slots at once, random view per frame
    wrong, worst, t0, part = 0, 0.0, time.time(), 200
    for t in range(0, traj_steps, part):
        ix = rng.integers(0, n_views, (n, min(part, traj_steps - t)))
        f = r.render_trajectories(np.arange(n), vpos[ix], vquat[ix])
        for j in range(ix.shape[1]):
            got, dist = classify(f[:, j], refs)
            wrong += int((got != ix[:, j]).sum())
            worst = max(worst, float(dist.max()))
    res["render_trajectories"] = {"steps_per_slot": traj_steps, "frames": traj_steps * n, "mismatched_frames": wrong,
                                  "max_abs_diff_to_own_reference": round(worst, 3), "restarts": r.restarts,
                                  "seconds": round(time.time() - t0, 1)}
    print("SYNC render_trajectories", json.dumps(res["render_trajectories"]), flush=True)
    return res


# ----------------------------------------------------------------------------- 2. rollout geometry
def test_rollouts(r, d, path, rng, rounds=2, per_episode=40):
    n, W, H = r.n, r.width, r.height
    rows, tiles, gate_ms = [], [], []
    for rd in range(rounds):
        eps = rng.choice(np.flatnonzero(d["length"] > per_episode + 5), n, replace=False)
        t0 = time.perf_counter()
        r.set_gates(np.arange(n), d["gates_pos"][eps], d["gates_yaw"][eps])
        gate_ms.append({"wall_ms_all_slots": round(1e3 * (time.perf_counter() - t0), 1),
                        "worker_ms_per_slot_mean": round(float(np.mean(list(r.last_ms.values()))), 1),
                        "worker_ms_per_slot_max": round(float(np.max(list(r.last_ms.values()))), 1)})
        half = per_episode // 2
        # closed-loop API: random steps; first call directly after set_gates
        ts = np.stack([np.sort(rng.choice(int(d["length"][e]), half, replace=False)) for e in eps])  # [n, half]
        frames = np.stack([r.render(d["pos"][eps, ts[:, j]], d["quat"][eps, ts[:, j]]) for j in range(half)], 1)
        # offline API: one contiguous segment per episode
        st = np.array([rng.integers(0, int(d["length"][e]) - half) for e in eps])
        tt = st[:, None] + np.arange(half)[None]
        frames_t = r.render_trajectories(np.arange(n), d["pos"][eps[:, None], tt], d["quat"][eps[:, None], tt])
        for api, T_idx, F in (("render", ts, frames), ("render_trajectories", tt, frames_t)):
            for i, e in enumerate(eps):
                for j in range(half):
                    t = int(T_idx[i, j])
                    row = check_frame(F[i, j], d, int(e), t, W, H)
                    row.update(api=api, episode=int(e), step=t, slot=i, round=rd, index_in_call_sequence=j)
                    rows.append(row)
                    if row["predicted_visible"] and row["unoccluded"] and not row["detected"]:
                        # diagnose: what is in the frame, and does rendering the same pose again find the gate?
                        pos = np.repeat(d["pos"][e, t][None], n, 0)
                        quat = np.repeat(d["quat"][e, t][None], n, 0)
                        act = np.zeros(n, bool)
                        act[i] = True
                        again = check_frame(r.render(pos, quat, act)[i], d, int(e), t, W, H)
                        row.update(
                            gate_px_in_frame=int(gate_mask(F[i, j]).sum()), frame_mean=round(float(F[i, j].mean()), 1),
                            diff_to_previous_frame_of_slot=round(float(np.abs(
                                F[i, j].astype(float) - F[i, j - 1]).mean()), 2) if j else None,
                            rerender_detected=again.get("detected"), rerender_gate_px=again["expected_px"])
                        k = sum(1 for x in rows if "rerender_detected" in x)
                        if k <= 6:
                            np.save(OUT / f"render_airsim_fail_{k}.npy", F[i, max(0, j - 1):j + 2])
                    if row["predicted_visible"] and len(tiles) < 15 and rng.random() < 0.03:
                        tiles.append((f"ep {e} t {t} target G{row['target'] + 1} err {row.get('bbox_err_px', -1):.1f}px",
                                      overlay(F[i, j], d, int(e), t, W, H)))
    vis = [x for x in rows if x["predicted_visible"]]
    clear = [x for x in vis if x["unoccluded"]]
    det = [x for x in clear if x["detected"]]
    err = np.array([x["bbox_err_px"] for x in det]) if det else np.zeros(1)
    cen = np.array([x["centroid_err_px"] for x in det]) if det else np.zeros(1)
    res = {
        "rollouts": str(path), "frames": len(rows), "episodes": len({x["episode"] for x in rows}),
        "frames_target_gate_predicted_in_view": len(vis),
        "of_those_not_overlapped_by_other_gates": len(clear),
        "detected_where_predicted": len(det),
        "detected_fraction": round(len(det) / max(1, len(clear)), 4),
        "bbox_edge_err_px_median_p95_max": [round(float(v), 2) for v in (np.median(err), np.percentile(err, 95), err.max())],
        "centroid_err_px_median_p95_max": [round(float(v), 2) for v in (np.median(cen), np.percentile(cen, 95), cen.max())],
        "silhouette_iou_median": round(float(np.median([x["iou"] for x in det])) if det else 0.0, 3),
        "all_gates_iou_median_all_frames_with_gates": round(float(np.median([x["iou_all"] for x in rows if x["iou_all"] is not None])), 3),
        "by_api_detected_fraction": {a: round(sum(x["detected"] for x in clear if x["api"] == a)
                                              / max(1, sum(x["api"] == a for x in clear)), 4)
                                     for a in ("render", "render_trajectories")},
        # gate colour outside the predicted silhouettes: expected when a gate bar crosses the camera
        # plane (the test's silhouette predictor skips such bars instead of clipping them)
        "frames_with_gate_colour_outside_prediction": sum(x["stray_px"] > 30 for x in rows),
        "of_those_with_a_gate_bar_crossing_the_camera_plane": sum(x["stray_px"] > 30 and x["bar_crosses_camera_plane"]
                                                                  for x in rows),
        "set_gates": gate_ms,
    }
    worst = sorted(det, key=lambda x: -x["bbox_err_px"])[:3]
    res["worst_cases"] = [{k: x[k] for k in ("api", "episode", "step", "target", "bbox_err_px", "iou", "expected_px")
                           if k in x} for x in worst]
    res["undetected"] = [{k: v for k, v in x.items() if k not in ("predicted_visible", "unoccluded", "iou_all")}
                         for x in clear if not x["detected"]][:20]
    if tiles:
        cols = 5
        sheet = np.zeros((-(-len(tiles) // cols) * H, cols * W, 3), np.uint8)
        for k, (label, img) in enumerate(tiles):
            cv2.putText(img, label, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 2)
            cv2.putText(img, label, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
            sheet[(k // cols) * H:(k // cols + 1) * H, (k % cols) * W:(k % cols + 1) * W] = img
        cv2.imwrite(str(OUT / "render_airsim_rollout_sheet.png"), sheet[..., ::-1])
        res["contact_sheet"] = str(OUT / "render_airsim_rollout_sheet.png")
    print("ROLLOUT", json.dumps(res), flush=True)
    return res


def check_frame(img, d, e, t, W, H):
    gp, gy, pos, quat, g = d["gates_pos"][e], d["gates_yaw"][e], d["pos"][e, t], d["quat"][e, t], int(d["target"][e, t])
    mask = gate_mask(img)
    exp_all = expected_mask(gp, gy, range(len(gy)), pos, quat, W, H)
    own = expected_mask(gp, gy, [g], pos, quat, W, H)
    others = expected_mask(gp, gy, [k for k in range(len(gy)) if k != g], pos, quat, W, H)
    uv, z = project(track.gate_corners(gp[g].astype(np.float64), np.float64(gy[g])), pos, quat, W, H)
    inside = bool((z > 0.2).all() and (uv[:, 0] > 2).all() and (uv[:, 0] < W - 3).all()
                  and (uv[:, 1] > 2).all() and (uv[:, 1] < H - 3).all())
    union = (mask | exp_all).sum()
    row = {"target": g, "predicted_visible": bool(inside and own.sum() >= 60), "expected_px": int(own.sum()),
           "iou_all": round(float((mask & exp_all).sum() / union), 3) if exp_all.sum() >= 60 else None,
           "stray_px": int((mask & ~cv2.dilate(exp_all.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)).sum()),
           "bar_crosses_camera_plane": any(
               (z := project(bar, pos, quat, W, H)[1]).min() <= 0.05 and z.max() > 0.0
               for k in range(len(gy)) for bar in gate_bars(gp[k], gy[k]))}
    if not row["predicted_visible"]:
        return row
    near = cv2.dilate(own.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    row["unoccluded"] = not bool((near & cv2.dilate(others.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).any())
    m = mask & near
    row["detected"] = bool(m.sum() > 0.5 * own.sum())
    if m.any():
        row["bbox_err_px"] = float(np.abs(bbox(m) - bbox(own)).max())
        row["centroid_err_px"] = float(np.linalg.norm(np.argwhere(m).mean(0) - np.argwhere(own).mean(0)))
        row["iou"] = float((m & own).sum() / (m | own).sum())
    return row


def overlay(img, d, e, t, W, H):
    g = int(d["target"][e, t])
    uv, _ = project(track.gate_corners(d["gates_pos"][e, g].astype(np.float64), np.float64(d["gates_yaw"][e, g])),
                    d["pos"][e, t], d["quat"][e, t], W, H)
    vis = np.ascontiguousarray(img)
    cv2.polylines(vis, [np.round(uv).astype(np.int32)], True, (0, 255, 0), 1)
    return vis


# ----------------------------------------------------------------------------- 3. watchdog, throughput
def test_watchdog(r, d):
    n = r.n
    pos, quat = d["pos"][:n, 5], d["quat"][:n, 5]
    before = r.render(pos, quat)
    victim = n // 2
    os.killpg(r._sim_pids[victim], signal.SIGKILL)  # the sim of one slot dies
    t0 = time.time()
    after = r.render(pos, quat)
    res = {"killed_slot": victim, "restarts": r.restarts, "recovery_s": round(time.time() - t0, 1),
           "frame_mean_abs_diff_after_restart": round(float(np.abs(after[victim].astype(float) - before[victim]).mean()), 3),
           "other_slots_mean_abs_diff": round(float(np.abs(after.astype(float) - before).mean()), 3)}
    print("WATCHDOG", json.dumps(res), flush=True)
    return res


def throughput(r, d, steps, traj_len):
    n = r.n
    E = d["pos"].shape[0]
    eps = np.arange(n) % E
    r.set_gates(np.arange(n), d["gates_pos"][eps], d["gates_yaw"][eps])
    L = int(d["length"][eps].min())
    for t in range(3):
        r.render(d["pos"][eps, t], d["quat"][eps, t])
    # timing without GPU polling (kept apart in case nvidia-smi queries disturb rendering); CPU from /proc
    load, times = Load(r), []
    for s in range(steps):
        t0 = time.perf_counter()
        r.render(d["pos"][eps, s % L], d["quat"][eps, s % L])
        times.append(time.perf_counter() - t0)
    cpu = round((load.cpu() - load.cpu0) / (time.time() - load.t0) / n, 2)
    times = np.array(times)
    load = Load(r)  # GPU utilisation: same loop again with the sampler running
    load.start()
    t0 = time.perf_counter()
    for s in range(steps // 2):
        r.render(d["pos"][eps, s % L], d["quat"][eps, s % L])
    sampled_ms = 1e3 * (time.perf_counter() - t0) / max(1, steps // 2)
    stats = load.summary()
    stats["cpu_cores_per_slot"] = cpu
    half = np.zeros(n, bool)
    half[: n // 2] = True
    t0 = time.perf_counter()
    for s in range(max(10, steps // 4)):
        r.render(d["pos"][eps, s % L], d["quat"][eps, s % L], half)
    half_ms = 1e3 * (time.perf_counter() - t0) / max(10, steps // 4)
    tl = min(traj_len, L)
    t0 = time.perf_counter()
    r.render_trajectories(np.arange(n), d["pos"][eps, :tl], d["quat"][eps, :tl])
    traj_fps = n * tl / (time.perf_counter() - t0)
    row = {"look": r.look, "gpus": r.gpus, "instances_per_gpu": n // len(r.gpus), "slots": n,
           "res": f"{r.width}x{r.height}",
           "render_step_ms_mean": round(1e3 * float(times.mean()), 1),
           "render_step_ms_median": round(1e3 * float(np.median(times)), 1),
           "render_step_ms_mean_while_polling_nvidia_smi": round(sampled_ms, 1),
           "render_step_ms_p95": round(1e3 * float(np.percentile(times, 95)), 1),
           "render_frames_per_s": round(n / float(times.mean()), 1),
           "render_step_ms_half_slots_active": round(half_ms, 1),
           "render_trajectories_frames_per_s": round(traj_fps, 1), **stats, "restarts": r.restarts}
    print("THROUGHPUT", json.dumps(row), flush=True)
    return row


def our_sims():
    out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "[B]locks-Linux-Shipping"], capture_output=True, text=True)
    return len(out.stdout.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--ipg", type=int, default=8, help="instances per GPU for the sync / geometry tests")
    ap.add_argument("--res", default="320x180")
    ap.add_argument("--sync-steps", type=int, default=2000)
    ap.add_argument("--traj-steps", type=int, default=2000)
    ap.add_argument("--rollouts", default=str(ROOT / "runs/teacher/cruise_s1/rollouts.npz"))
    ap.add_argument("--throughput-steps", type=int, default=100)
    ap.add_argument("--rollout-rounds", type=int, default=2, help="16 episodes x 40 frames per round")
    ap.add_argument("--throughput-ipg", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--looks", nargs="*", default=["stock", "fast"])
    ap.add_argument("--skip", nargs="*", default=[], choices=["sync", "rollouts", "watchdog", "throughput"])
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.lower().split("x"))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with np.load(a.rollouts) as z:  # read each array once: indexing an NpzFile re-reads it every time
        d = {k: z[k] for k in ("pos", "quat", "target", "gates_pos", "gates_yaw", "length")}
    res = {"args": vars(a)}

    try:  # a GPU that does not exist must give a clear error
        AirsimRenderer(gpus=[99], instances_per_gpu=1)
        res["bad_gpu_error"] = "NO ERROR RAISED"
    except RuntimeError as e:
        res["bad_gpu_error"] = str(e)
    print("BAD_GPU", res["bad_gpu_error"], flush=True)

    t0 = time.time()
    with AirsimRenderer(gpus=a.gpus, instances_per_gpu=a.ipg, width=W, height=H, look="stock") as r:
        res["startup_s"] = {"slots": r.n, "seconds": round(time.time() - t0, 1)}
        print("STARTUP", json.dumps(res["startup_s"]), flush=True)
        if "rollouts" not in a.skip:
            res["rollout_geometry"] = test_rollouts(r, d, a.rollouts, rng, a.rollout_rounds)
        if "sync" not in a.skip:
            nominal = track.circuit_arrays()
            r.set_gates(np.arange(r.n), np.repeat(nominal[0][None], r.n, 0), np.repeat(nominal[1][None], r.n, 0))
            res["sync"] = test_sync(r, a.sync_steps, a.traj_steps, rng)
        if "watchdog" not in a.skip:
            res["watchdog"] = test_watchdog(r, d)
        rows = []
        if "throughput" not in a.skip and a.ipg in a.throughput_ipg and "stock" in a.looks:
            rows.append(throughput(r, d, a.throughput_steps, 96))
    if "throughput" not in a.skip:
        for look in a.looks:
            for ipg in a.throughput_ipg:
                if look == "stock" and ipg == a.ipg:
                    continue
                with AirsimRenderer(gpus=a.gpus, instances_per_gpu=ipg, width=W, height=H, look=look) as r:
                    rows.append(throughput(r, d, a.throughput_steps, 96))
        res["throughput"] = rows
    time.sleep(2)
    res["sim_processes_left_after_close"] = our_sims()
    mine = f"_{os.getpid()}_"  # scratch of this process only (other renderers may be running)
    res["scratch_left_after_close"] = len([p for p in list(Path("/tmp").glob("fpv_airsim_run_*"))
                                           + list(Path("/dev/shm").glob("fpv_airsim_*")) if mine in p.name])
    print("LEFTOVER_SIMS", res["sim_processes_left_after_close"], "LEFTOVER_SCRATCH", res["scratch_left_after_close"],
          flush=True)
    (OUT / "render_airsim_test.json").write_text(json.dumps(res, indent=1, default=str))
    print("WROTE", OUT / "render_airsim_test.json")


if __name__ == "__main__":
    main()
