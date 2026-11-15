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
from fpv import render_airsim as ra  # noqa: E402
from fpv.render_airsim import GATE_DEPTH, AirsimRenderer  # noqa: E402

OUT = ROOT / "checks" / "out"


def _rel(text):
    """Paths in reports relative to the project directory (the reports are published)."""
    return text.replace(str(ROOT) + "/", "")


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
    print("ROLLOUT", _rel(json.dumps(res)), flush=True)
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


# ----------------------------------------------------------------------------- 4. third person
def drone_silhouette(dpos, dquat, cam_pos, cam_quat, W, H, hfov, scale):
    """Pixels covered by the quadrotor model (all parts) predicted by the pinhole model."""
    m = np.zeros((H, W), np.uint8)
    R = ra.quat_to_mat(np.asarray(dquat, float))
    for _, V, _, I, _ in ra.drone_parts(scale):
        uv, z = ra.project_points(np.asarray(dpos, float) + V @ R.T, cam_pos, cam_quat, W, H, hfov)
        if z.min() < 0.05 or np.abs(uv).max() > 1e5:
            continue
        for tri in np.round(uv[I.reshape(-1, 3)] * 8).astype(np.int32):  # 3 fractional bits
            cv2.fillConvexPoly(m, tri, 1, cv2.LINE_8, 3)
    return m.astype(bool)


def gates_in_front(gp, gy, depth_limit, cam_pos, cam_quat, W, H, hfov):
    """Silhouette of the gate bars that are nearer to the camera than ``depth_limit``."""
    m = np.zeros((H, W), np.uint8)
    for g in range(len(gy)):
        for bar in gate_bars(gp[g], gy[g]):
            uv, z = ra.project_points(bar, cam_pos, cam_quat, W, H, hfov)
            if z.min() > 0.05 and z.min() < depth_limit and np.abs(uv).max() < 1e5:
                cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return m.astype(bool)


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def score_frames(with_drone, without, dpos, dquat, cam_pos, cam_quat, gp, gy, hfov, scale):
    """Compare the model's pixels (frame with model minus frame without) with the predicted silhouette
    of the model at THIS step and at the neighbouring steps. -> one row per frame."""
    T, H, W = with_drone.shape[:3]
    sil = [drone_silhouette(dpos[t], dquat[t], cam_pos[t], cam_quat[t], W, H, hfov, scale) for t in range(T)]
    nb = lambda t, k: drone_silhouette(dpos[k], dquat[k], cam_pos[t], cam_quat[t], W, H, hfov, scale)  # noqa: E731
    rows = []
    for t in range(T):
        pred = sil[t]
        prev, nxt = (nb(t, t - 1) if t else None), (nb(t, t + 1) if t + 1 < T else None)
        row = {"t": t, "pred_px": int(pred.sum())}
        uv, z = ra.project_points(dpos[t], cam_pos[t], cam_quat[t], W, H, hfov)
        if pred.sum() < 20 or z[0] <= 0:
            rows.append(row | {"skipped": "model under 20 px or behind the camera"})
            continue
        occl = gates_in_front(gp, gy, z[0], cam_pos[t], cam_quat[t], W, H, hfov)
        win = pred.copy()
        for o in (prev, nxt):
            if o is not None:
                win |= o
        win = cv2.dilate(win.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        diff = np.abs(with_drone[t].astype(np.int16) - without[t]).max(-1) > 25
        det = diff & win
        row.update(occluded_frac=round(float((pred & occl).sum() / pred.sum()), 3), det_px=int(det.sum()),
                   iou=round(_iou(det, pred), 3),
                   iou_prev=None if prev is None else round(_iou(det, prev), 3),
                   iou_next=None if nxt is None else round(_iou(det, nxt), 3),
                   neighbour_overlap=max(_iou(pred, o) for o in (prev, nxt) if o is not None))
        if det.any():
            row["centroid_err_px"] = round(float(np.linalg.norm(np.argwhere(det).mean(0) - np.argwhere(pred).mean(0))), 2)
        # detected: at least 30% of the predicted pixels changed (thin, distant views of the dark frame
        # against dark ground can stay below the difference threshold). A detected frame belongs to
        # its own step if its model pixels sit where this step's pose puts them (centroid within
        # 5 px) and match this step better than the neighbouring steps' poses (judged only where
        # those differ: silhouette overlap < 0.7)
        row["detected"] = bool(det.sum() >= 0.3 * pred.sum())
        own = row.get("centroid_err_px", 1e9) <= 5.0
        if row["neighbour_overlap"] < 0.7:
            own = own and row["iou"] > max(v for v in (row["iou_prev"], row["iou_next"]) if v is not None)
        row["matches_own_step"] = bool(own)
        rows.append(row)
    return rows


def summarize_rows(rows, name):
    scored = [x for x in rows if "skipped" not in x]
    clear = [x for x in scored if x["occluded_frac"] < 0.05]
    discriminating = [x for x in clear if x["neighbour_overlap"] < 0.7]
    bad = [x for x in clear if x["detected"] and not x["matches_own_step"]]
    err = np.array([x["centroid_err_px"] for x in clear if "centroid_err_px" in x] or [0.0])
    out = {"frames": len(rows), "scored": len(scored), "model_not_occluded_by_gates": len(clear),
           "of_those_where_neighbour_steps_differ": len(discriminating),
           "not_detected": sum(1 for x in clear if not x["detected"]),
           "frames_not_matching_own_step": len(bad),
           "frames_matching_a_neighbour_step_better": sum(1 for x in clear if x["iou"] < max(
               [v for v in (x["iou_prev"], x["iou_next"]) if v is not None] or [0])),
           "iou_median_p5": [round(float(np.median([x["iou"] for x in clear] or [0])), 3),
                             round(float(np.percentile([x["iou"] for x in clear] or [0], 5)), 3)],
           "centroid_err_px_median_p95_max": [round(float(v), 2) for v in (np.median(err), np.percentile(err, 95), err.max())],
           "model_px_median": int(np.median([x["pred_px"] for x in scored] or [0])),
           "first_bad": bad[:10]}
    print("THIRD_PERSON", name, json.dumps({k: v for k, v in out.items() if k != "first_bad"}), flush=True)
    return out


def random_jumps(rng, k, steps):
    """Per slot: 4 fixed cameras looking at the track area and a drone pose per frame, random in front
    of the camera of that frame (1 to 3.5 m away, random attitude). -> cam [k,T,3], cq, dp, dq."""
    centre = np.array([3.5, 2.75, 1.2])
    cams = []
    for s in range(k):
        c = []
        for j in range(4):
            ang = rng.uniform(0, 2 * math.pi)
            p = centre + np.array([4 * math.cos(ang), 4 * math.sin(ang), rng.uniform(0.5, 2.5)])
            c.append((p, ra.look_at_quat(p, centre + rng.uniform(-1, 1, 3) * [1, 1, 0.3])))
        cams.append(c)
    idx = rng.integers(0, 4, (k, steps))
    cp = np.array([[cams[s][j][0] for j in idx[s]] for s in range(k)])
    cq = np.array([[cams[s][j][1] for j in idx[s]] for s in range(k)])
    R = ra.quat_to_mat(cq)
    d = rng.uniform(1.0, 3.5, (k, steps))
    local = np.stack([d, rng.uniform(-0.55, 0.55, (k, steps)) * d, rng.uniform(-0.3, 0.3, (k, steps)) * d], -1)
    dp = cp + np.einsum("ktij,ktj->kti", R, local)
    dp[..., 2] = np.maximum(dp[..., 2], 0.25)
    r_, p_, y_ = rng.uniform(-0.6, 0.6, (k, steps)), rng.uniform(-0.6, 0.6, (k, steps)), rng.uniform(-math.pi, math.pi, (k, steps))
    cy, sy, cpi, spi, cr, sr = (f(a / 2) for a in (y_, p_, r_) for f in (np.cos, np.sin))
    dq = np.stack([sr * cpi * cy - cr * spi * sy, cr * spi * cy + sr * cpi * sy, cr * cpi * sy - sr * spi * cy,
                   cr * cpi * cy + sr * spi * sy], -1)
    return cp, cq, dp, dq


def test_third_person(gpus, ipg, d, rng, frames_per_view=300, jump_steps=300, render_steps=60, size=(640, 360)):
    W, H = size
    hfov, scale = ra.TP_HFOV_DEG, ra.DRONE_SCALE
    hidden = np.array(ra.DRONE_HIDDEN_POS, float)
    res = {"renderer": {"gpus": gpus, "instances_per_gpu": ipg, "size": [W, H], "hfov_deg": hfov, "drone_scale": scale}}
    parts = ra.drone_parts(scale)
    allv = np.concatenate([V for _, V, _, _, _ in parts])
    res["drone_model"] = {"scale_vs_true_size": scale, "motor_to_motor_m": round(0.092 * scale, 3),
                          "span_incl_rotor_discs_m": round(float(allv[:, 0].max() - allv[:, 0].min()), 3),
                          "height_m": round(float(allv[:, 2].max() - allv[:, 2].min()), 3)}
    t0 = time.time()
    with AirsimRenderer(gpus=gpus, instances_per_gpu=ipg, width=W, height=H, look="stock", hfov_deg=hfov,
                        mount_pitch_deg=0.0, drone_scale=scale) as r:
        res["startup_s"] = round(time.time() - t0, 1)
        k = r.n
        # (a) recorded flights: teacher rollouts (own gate layouts), chase and overview cameras
        eps = rng.choice(np.flatnonzero(d["length"] >= frames_per_view + 50), k, replace=False)
        r.set_gates(np.arange(k), d["gates_pos"][eps], d["gates_yaw"][eps])
        dp = np.stack([d["pos"][e, 50:50 + frames_per_view] for e in eps]).astype(float)
        dq = np.stack([d["quat"][e, 50:50 + frames_per_view] for e in eps]).astype(float)
        views, chase_info = {"chase": ([], []), "overview": ([], [])}, []
        for i, e in enumerate(eps):
            full_p, full_q = d["pos"][e, : d["length"][e]].astype(float), d["quat"][e, : d["length"][e]].astype(float)
            cp, cq, info = ra.chase_camera(full_p, full_q, d["gates_pos"][e], d["gates_yaw"][e])
            chase_info.append(info)
            views["chase"][0].append(cp[50:50 + frames_per_view])
            views["chase"][1].append(cq[50:50 + frames_per_view])
            pts = np.concatenate([track.gate_corners(d["gates_pos"][e].astype(float), d["gates_yaw"][e].astype(float),
                                                     track.GATE_OUTER).reshape(-1, 3), full_p])
            op, oq = ra.overview_camera(pts, W, H, hfov)
            views["overview"][0].append(np.repeat(op[None], frames_per_view, 0))
            views["overview"][1].append(np.repeat(oq[None], frames_per_view, 0))
        res["chase_camera"] = {"drone_centre_hidden_by_gate_frac": [x["occluded_frac"] for x in chase_info],
                               "min_gate_frame_distance_m": min(x["min_gate_frame_distance_m"] for x in chase_info)}
        rates = {}
        for v, (cp, cq) in views.items():
            cp, cq = np.stack(cp), np.stack(cq)
            t1 = time.time()
            a = r.render_trajectories(np.arange(k), cp, cq, dp, dq)
            rates[v] = round(k * frames_per_view / (time.time() - t1) / len(gpus), 1)
            b = r.render_trajectories(np.arange(k), cp, cq, np.broadcast_to(hidden, dp.shape), dq)
            rows = []
            for i, e in enumerate(eps):
                rows += [x | {"episode": int(e)} for x in score_frames(a[i], b[i], dp[i], dq[i], cp[i], cq[i],
                                                                       d["gates_pos"][e], d["gates_yaw"][e], hfov, scale)]
            res[f"flight_{v}"] = summarize_rows(rows, v)
            if v == "chase":
                inside = [x for x in rows if "skipped" not in x]
                res["chase_camera"]["frames_model_visible_frac"] = round(sum(
                    1 for x in inside if x["det_px"] > 0) / max(1, len(rows)), 4)
        res["render_frames_per_s_per_gpu"] = {v: {"frames_per_s_per_gpu": f, "instances_per_gpu": ipg, "size": [W, H]}
                                              for v, f in rates.items()}
        # (b) random jumps: every frame a new random model pose (and one of 4 cameras), gates out of the way
        far = np.repeat(np.array([[0.0, -40.0, 1.0]] * 6)[None], k, 0)
        r.set_gates(np.arange(k), far, np.zeros((k, 6)))
        gp_far, gy_far = far[0], np.zeros(6)
        cp, cq, jp, jq = random_jumps(rng, k, jump_steps)
        a = r.render_trajectories(np.arange(k), cp, cq, jp, jq)
        b = r.render_trajectories(np.arange(k), cp, cq, np.broadcast_to(hidden, jp.shape), jq)
        rows = []
        for i in range(k):
            rows += score_frames(a[i], b[i], jp[i], jq[i], cp[i], cq[i], gp_far, gy_far, hfov, scale)
        res["random_jumps_render_trajectories"] = summarize_rows(rows, "jumps render_trajectories")
        fa, fb = [], []
        for t in range(render_steps):  # closed-loop API: model pose and camera pose in the same call
            fa.append(r.render(cp[:, t], cq[:, t], drone_pos=jp[:, t], drone_quat=jq[:, t]))
        for t in range(render_steps):
            fb.append(r.render(cp[:, t], cq[:, t], drone_pos=np.broadcast_to(hidden, (k, 3)), drone_quat=jq[:, t]))
        fa, fb = np.stack(fa, 1), np.stack(fb, 1)
        rows = []
        for i in range(k):
            rows += score_frames(fa[i], fb[i], jp[i, :render_steps], jq[i, :render_steps], cp[i, :render_steps],
                                 cq[i, :render_steps], gp_far, gy_far, hfov, scale)
        res["random_jumps_render"] = summarize_rows(rows, "jumps render")
        res["restarts"], res["failed_launches"] = r.restarts, [x["error"] for x in r.failed_launches]
    res["sync_pass"] = sync_pass(res)
    res["pass"] = res["sync_pass"]
    return res


def sync_pass(res):
    keys = ("flight_chase", "flight_overview", "random_jumps_render_trajectories", "random_jumps_render")
    return bool(all(res[k]["frames_not_matching_own_step"] == 0
                    and res[k]["not_detected"] <= 0.02 * res[k]["model_not_occluded_by_gates"] for k in keys))


# ----------------------------------------------------------------------------- 5. target highlight
def highlight_mask(img, rgb):
    """Pixels of the highlight colour: its dominant channel(s) clearly above the others (like
    `gate_mask` for magenta). For (0, 1, 0): green > 1.4 x max(red, blue) and green > 80."""
    f = img.astype(np.int32)
    rgb = np.asarray(rgb, float)
    on, off = rgb >= 0.5 * rgb.max(), rgb < 0.5 * rgb.max()
    lo = f[..., on].min(-1)
    hi = f[..., off].max(-1) if off.any() else np.zeros(lo.shape, np.int32)
    return (lo > 80) & (lo > 1.4 * hi)


def score_highlight(frames, targets, d, eps, ts, rgb, W, H):
    """Per frame: on which gates the highlight colour appears, versus the commanded target of that
    frame and of the previous one. -> rows.

    A frame is wrong if the highlight shows on any gate other than its own target, or if its target
    is clearly visible (>= 60 predicted pixels not hidden by a nearer gate) and is not highlighted
    or still shows magenta."""
    rows = []
    for i, e in enumerate(eps):
        gp, gy = d["gates_pos"][e], d["gates_yaw"][e]
        for j in range(frames.shape[1]):
            t, k = int(ts[i, j]), int(targets[i, j])
            pos, quat = d["pos"][e, t], d["quat"][e, t]
            preds = [expected_mask(gp, gy, [g], pos, quat, W, H) for g in range(6)]
            dist = np.linalg.norm(gp - pos, axis=1)
            hl = highlight_mask(frames[i, j], rgb)
            near = [cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for m in preds]
            hits = [int((hl & m).sum()) for m in near]
            row = {"slot": i, "frame": j, "target": k, "prev_target": int(targets[i, j - 1]) if j else None,
                   "highlight_px": int(hl.sum()), "highlight_px_per_gate": hits,
                   "on_other_gate": [g for g in range(6) if g != k and hits[g] >= 20
                                     and not (k >= 0 and (near[g] & near[k]).any())]}  # overlapping silhouettes are ambiguous
            vis = 0
            if k >= 0:
                nearer = [preds[g] for g in range(6) if g != k and dist[g] < dist[k]]
                own = preds[k] & ~np.logical_or.reduce(nearer) if nearer else preds[k]
                vis = int(own.sum())
                row["target_unhidden_px"] = vis
                if vis >= 60:
                    row["target_px_highlight_frac"] = round(float((hl & own).sum() / vis), 3)
                    row["target_px_magenta_frac"] = round(float((gate_mask(frames[i, j]) & own).sum() / vis), 3)
            row["target_visible"] = vis >= 60
            row["ok"] = not row["on_other_gate"] and (not row["target_visible"] or (
                hits[k] >= 20 and row["target_px_magenta_frac"] < 0.05))
            if k < 0:
                row["ok"] = row["ok"] and row["highlight_px"] < 20
            rows.append(row)
    return rows


def summarize_highlight(rows, name):
    vis = [x for x in rows if x["target_visible"]]
    out = {"frames": len(rows), "target_clearly_visible": len(vis), "frames_wrong": sum(not x["ok"] for x in rows),
           "frames_highlight_on_previous_target": sum(1 for x in rows if x["prev_target"] is not None and x["prev_target"] >= 0
                                                      and x["prev_target"] in x["on_other_gate"]),
           "target_highlight_coverage_median_p5": [round(float(np.median([x["target_px_highlight_frac"] for x in vis] or [0])), 3),
                                                   round(float(np.percentile([x["target_px_highlight_frac"] for x in vis] or [0], 5)), 3)],
           "target_magenta_frac_max": max([x["target_px_magenta_frac"] for x in vis] or [0]),
           "first_wrong": [x for x in rows if not x["ok"]][:10]}
    print("HIGHLIGHT", name, json.dumps({k: v for k, v in out.items() if k != "first_wrong"}), flush=True)
    return out


def test_highlight(gpus, ipg, d, rng, rgb, steps=300, render_steps=80, size=(320, 180)):
    """Lock-step of the target highlight: random target every frame (worst case) on teacher rollout
    poses with their own layouts, for render_trajectories and render; cost of target changes."""
    W, H = size
    res = {"renderer": {"gpus": gpus, "instances_per_gpu": ipg, "size": [W, H], "highlight_rgb": list(rgb)}}
    with AirsimRenderer(gpus=gpus, instances_per_gpu=ipg, width=W, height=H, highlight_target=True,
                        highlight_rgb=rgb) as r:
        k = r.n
        eps = rng.choice(np.flatnonzero(d["length"] > steps + 20), k, replace=False)
        r.set_gates(np.arange(k), d["gates_pos"][eps], d["gates_yaw"][eps])
        ts = np.stack([np.arange(steps) + 10] * k)
        P, Q = d["pos"][eps[:, None], ts], d["quat"][eps[:, None], ts]
        tg = rng.integers(-1, 6, (k, steps))
        f = r.render_trajectories(np.arange(k), P, Q, target=tg)
        res["render_trajectories_random_target"] = summarize_highlight(score_highlight(f, tg, d, eps, ts, rgb, W, H),
                                                                       "render_trajectories")
        fr = np.stack([r.render(P[:, j], Q[:, j], target=tg[:, j]) for j in range(render_steps)], 1)
        res["render_random_target"] = summarize_highlight(score_highlight(fr, tg[:, :render_steps], d, eps, ts[:, :render_steps],
                                                                          rgb, W, H), "render")
        # realistic: the teacher's own target sequence (changes when a gate is passed)
        tr = d["target"][eps[:, None], ts].astype(np.int64)
        f = r.render_trajectories(np.arange(k), P, Q, target=tr)
        res["render_trajectories_flight_target"] = summarize_highlight(score_highlight(f, tr, d, eps, ts, rgb, W, H),
                                                                       "flight target")
        # cost (shared GPU: pessimistic)
        timing = {}
        for name, kw in (("no_target", {}), ("constant_target", {"target": np.zeros((k, steps), np.int64)}),
                         ("target_every_frame", {"target": tg}), ("flight_target", {"target": tr})):
            t0 = time.perf_counter()
            r.render_trajectories(np.arange(k), P, Q, **kw)
            timing[name] = round(k * steps / (time.perf_counter() - t0), 1)
        res["render_trajectories_frames_per_s"] = timing
        timing = {}
        for name, kw in (("no_target", lambda j: {}), ("target_every_frame", lambda j: {"target": tg[:, j]})):
            t0 = time.perf_counter()
            for j in range(render_steps):
                r.render(P[:, j], Q[:, j], **kw(j))
            timing[name] = round(1e3 * (time.perf_counter() - t0) / render_steps, 1)
        res["render_ms_per_step"] = timing
        res["changes_per_frame"] = {"target_every_frame": round(float((np.diff(tg, axis=1) != 0).mean()), 3),
                                    "flight_target": round(float((np.diff(tr, axis=1) != 0).mean()), 4)}
        res["restarts"] = r.restarts
    keys = ("render_trajectories_random_target", "render_random_target", "render_trajectories_flight_target")
    res["pass"] = bool(all(res[k_]["frames_wrong"] == 0 for k_ in keys))
    return res


def startup_trials(gpus, ipg, trials):
    """Full start-ups of gpus x ipg slots: how often a slot needed a relaunch, and why."""
    rows = []
    for i in range(trials):
        t0 = time.time()
        try:
            with AirsimRenderer(gpus=gpus, instances_per_gpu=ipg, width=320, height=180) as r:
                row = {"trial": i, "ok": True, "seconds": round(time.time() - t0, 1), "slots": r.n,
                       "slot_relaunches": sum(v - 1 for v in r.launches.values()),
                       "errors": [x["error"] for x in r.failed_launches]}
                r.render(np.repeat([[2.0, 0.0, 1.0]], r.n, 0), np.repeat([[0, 0, 0, 1.0]], r.n, 0))
        except RuntimeError as e:
            row = {"trial": i, "ok": False, "seconds": round(time.time() - t0, 1), "error": str(e)[:500]}
        print("STARTUP_TRIAL", json.dumps(row), flush=True)
        rows.append(row)
    return {"gpus": gpus, "instances_per_gpu": ipg, "trials": rows,
            "slot_launches": sum(x.get("slots", 0) + x.get("slot_relaunches", 0) for x in rows),
            "slot_relaunches": sum(x.get("slot_relaunches", 0) for x in rows),
            "trials_ok": sum(x["ok"] for x in rows)}


class _FlakyRenderer(AirsimRenderer):
    """AirsimRenderer whose listed slots get a worker with a non-existent AirSim root for their first
    ``n`` launches: the worker starts, cannot launch its sim and reports an error, as a real failed
    start-up does."""

    def __init__(self, *args, fail=None, **kw):
        self._fail = dict(fail or {})
        super().__init__(*args, **kw)

    def _spawn(self, s):
        if self._fail.get(s, 0) <= 0:
            return super()._spawn(s)
        self._fail[s] -= 1
        popen = ra.subprocess.Popen

        def bad_root(cmd, *a, **k):
            cmd = list(cmd)
            cmd[cmd.index("--root") + 1] = "/nonexistent_fpv_airsim_root"
            return popen(cmd, *a, **k)

        ra.subprocess.Popen = bad_root
        try:
            super()._spawn(s)
        finally:
            ra.subprocess.Popen = popen


def startup_injected(gpus, ipg):
    """The relaunch path, exercised: (a) one slot fails its first launch -> relaunched alone, frames
    of all slots agree; (b) one slot always fails -> clear error after startup_retries + 1 launches and
    no sim left running."""
    res = {}
    victim = ipg * len(gpus) // 2 + 1
    pos, quat = np.array([[-0.5, 0.2, 1.1]]), np.array([[0.0, 0.0, 0.05, 0.9987]])
    t0 = time.time()
    with _FlakyRenderer(gpus=gpus, instances_per_gpu=ipg, width=320, height=180, fail={victim: 1}) as r:
        f = r.render(np.repeat(pos, r.n, 0), np.repeat(quat, r.n, 0)).astype(np.int16)
        same_gpu = [s for s in range(r.n) if s != victim and s % len(gpus) == victim % len(gpus)]
        res["one_failed_first_launch"] = {
            "slots": r.n, "victim_slot": victim, "seconds": round(time.time() - t0, 1), "launches": r.launches,
            "failed_launches": [{"slot": x["slot"], "error": x["error"]} for x in r.failed_launches],
            "relaunched_alone": r.launches[victim] == 2 and all(v == 1 for s, v in r.launches.items() if s != victim),
            "victim_frame_mean_abs_diff_to_same_gpu_slots": round(float(np.abs(f[victim] - f[same_gpu]).mean()), 3),
            "other_slots_mean_abs_diff_among_themselves": round(float(np.abs(f[same_gpu[0]] - f[same_gpu[1:]]).mean()), 3)}
    before = our_sims(gpus)
    t0 = time.time()
    try:
        _FlakyRenderer(gpus=gpus[:1], instances_per_gpu=2, width=320, height=180, startup_retries=2, fail={1: 99})
        err = "NO ERROR RAISED"
    except RuntimeError as e:
        err = str(e)
    time.sleep(3)
    res["always_failing_slot"] = {"startup_retries": 2, "seconds": round(time.time() - t0, 1), "error": err[:400],
                                  "sims_running_before_on_these_gpus": before, "sims_running_after_on_these_gpus": our_sims(gpus)}
    res["pass"] = bool(res["one_failed_first_launch"]["relaunched_alone"]
                       and res["one_failed_first_launch"]["victim_frame_mean_abs_diff_to_same_gpu_slots"] < 2.0
                       and "failed to start 3 times" in err and res["always_failing_slot"]["sims_running_after_on_these_gpus"] <= before)
    print("STARTUP_INJECTED", json.dumps(res), flush=True)
    return res


def our_sims(gpus=None):
    """Sim processes of this user (only those rendering on ``gpus`` if given: other jobs of the same
    user may run sims on other GPUs)."""
    out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-af", "[B]locks-Linux-Shipping"], capture_output=True, text=True)
    lines = out.stdout.splitlines()
    if gpus is not None:
        lines = [ln for ln in lines if any(f"-graphicsadapter={g} " in ln + " " for g in gpus)]
    return len(lines)


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
    ap.add_argument("--third-person", action="store_true",
                    help="only the third-person tests (+ --startup-trials); writes checks/out/third_person_test.json")
    ap.add_argument("--tp-ipg", type=int, default=2)
    ap.add_argument("--startup-trials", type=int, default=0)
    ap.add_argument("--startup-ipg", type=int, default=6)
    ap.add_argument("--highlight", type=float, nargs=3, default=None, metavar=("R", "G", "B"),
                    help="only the target-highlight test with this colour; writes checks/out/highlight_test.json")
    ap.add_argument("--startup-inject-only", action="store_true",
                    help="only the injected start-up failure test, merged into checks/out/third_person_test.json")
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.lower().split("x"))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with np.load(a.rollouts) as z:  # read each array once: indexing an NpzFile re-reads it every time
        d = {k: z[k] for k in ("pos", "quat", "target", "gates_pos", "gates_yaw", "length")}
    res = {"args": vars(a)}
    if a.highlight:
        res |= test_highlight(a.gpus, a.tp_ipg, d, rng, tuple(a.highlight))
        time.sleep(2)
        res["sim_processes_left_after_close"] = our_sims(a.gpus)
        (OUT / "highlight_test.json").write_text(_rel(json.dumps(res, indent=1, default=str)))
        print("WROTE", _rel(str(OUT / "highlight_test.json")), "PASS", res["pass"])
        return
    if a.startup_inject_only:
        path = OUT / "third_person_test.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        res["startup_injected_failure"] = startup_injected(a.gpus, a.startup_ipg)
        if "flight_chase" in res:
            res["sync_pass"] = sync_pass(res)
            res["pass"] = res["sync_pass"] and res["startup_injected_failure"]["pass"]
        path.write_text(_rel(json.dumps(res, indent=1, default=str)))
        print("WROTE", _rel(str(path)))
        return
    if a.third_person:
        if a.startup_trials:
            res["startup_reliability"] = startup_trials(a.gpus, a.startup_ipg, a.startup_trials)
            res["startup_injected_failure"] = startup_injected(a.gpus, a.startup_ipg)
        res |= test_third_person(a.gpus, a.tp_ipg, d, rng)
        if "startup_injected_failure" in res:
            res["pass"] = res["sync_pass"] and res["startup_injected_failure"]["pass"]
        time.sleep(2)
        res["sim_processes_left_after_close"] = our_sims(a.gpus)
        (OUT / "third_person_test.json").write_text(_rel(json.dumps(res, indent=1, default=str)))
        print("WROTE", _rel(str(OUT / "third_person_test.json")), "PASS", res["pass"])
        return

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
    (OUT / "render_airsim_test.json").write_text(_rel(json.dumps(res, indent=1, default=str)))
    print("WROTE", _rel(str(OUT / "render_airsim_test.json")))


if __name__ == "__main__":
    main()
