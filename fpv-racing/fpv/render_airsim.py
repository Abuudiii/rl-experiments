"""FPV frames from Project AirSim (v1.0.1 packaged Blocks, Unreal 5.2), used purely as a renderer.

    from fpv.render_airsim import AirsimRenderer
    r = AirsimRenderer(gpus=[4, 5], instances_per_gpu=8, width=320, height=180, look="stock")
    r.set_gates(slots, gates_pos, gates_yaw)             # per-slot gate layout (every episode reset)
    frames = r.render(pos, quat_xyzw, active=None)       # [n, H, W, 3] uint8 RGB, closed loop
    frames = r.render_trajectories(slots, pos, quat)     # [k, T, H, W, 3] uint8 RGB, offline
    r.close()

Everything a caller passes is in the contract frame (world: x forward, y left, z up, metres; body
pose = body->world, quaternion xyzw). The renderer applies the frame conversion to AirSim's NED
frame, the placement of the track in the Blocks level (`SCENE_ORIGIN`) and the camera mount
(`fpv.track.CAMERA`: at the body origin, looking along body +x, pitched up, pinhole, square pixels).

One slot = one sim process (one Unreal instance) + one client process. Slot i runs on
``gpus[i % len(gpus)]``. The client processes run with the AirSim venv's interpreter, so the calling
process only needs numpy. Frames come back through a file in /dev/shm.

Protocols (both lock-step: the sim clock is paused except while the client steps it, and every
frame is matched to its clock step by the message time stamp, never by arrival order):

* AirSim publishes, for each clock step, a message with the camera pose of that step but the PIXELS
  rendered in the previous step (UnrealCamera::TickComponent enqueues the read-back before the
  deferred capture of the same frame runs).
* `render`: set pose, step, step, take the second message -> two captures per frame.
* `render_trajectories`: set pose k, step, pair message k+1 with pose k -> one capture per frame
  plus one flush step per chunk of `CHUNK` frames.

Third person (optional, constructor ``drone_scale > 0``): every instance also holds a quadrotor model
(`drone_glb`; X frame, red front rotors, blue rear rotors), and `render` / `render_trajectories` take
its pose per frame (``drone_pos``, ``drone_quat``, body->world) next to the camera pose; the model is
moved in the same clock step as the camera, so every frame shows both poses of its own step.
``hfov_deg`` / ``mount_pitch_deg`` replace the FPV camera's field of view and mount pitch; with
``mount_pitch_deg=0`` the pose passed to `render*` is the camera pose itself (optical axis = +x, no
roll = level horizon). `chase_camera` and `overview_camera` compute such poses for a flight and
`project_points` is the matching pinhole model. Renderers built without these arguments are the
FPV renderer above, unchanged (no model in the scene).

Target highlight (optional, constructor ``highlight_target=True``, colour ``highlight_rgb``): every
instance also holds one gate in the highlight colour; ``target`` (per slot for `render`, per frame
for `render_trajectories`) names the gate shown in that colour: its magenta copy moves under the
floor and the highlight gate takes its pose, before the clock step of that frame.

Needs `scripts/setup_airsim.sh` to have been run on the machine (Blocks build + client venv under
`$FPV_AIRSIM_ROOT`, default /tmp/fpv_airsim).
"""

from __future__ import annotations

import atexit
import json
import math
import os
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

try:
    from fpv import track
except ImportError:  # run as a script by the worker interpreter
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fpv import track

AIRSIM_ROOT = os.environ.get("FPV_AIRSIM_ROOT", "/tmp/fpv_airsim")
# Where the contract-frame origin (ground level) sits in the Blocks level, in contract axes relative
# to AirSim's origin: the Blocks floor is 1.0 m above it and a 2.5 m high slab covers
# x in [-2, 10), y in [-10, 12) there, so the track is placed on open floor.
SCENE_ORIGIN = (-20.0, -3.0, 1.0)
GATE_DEPTH = 0.05  # thickness of the gate frame along the travel direction [m]
GATE_RGB = (1.0, 0.0, 0.8)  # magenta
HIGHLIGHT_RGB = (0.0, 1.0, 0.0)  # target gate with highlight_target=True: on screen median (0, 226, 9); no
# bright pixel of the scene around the volume (floor, sky, blocks, magenta gates) has a chroma within 0.5 of it
GATE_MESH_YAW = math.pi / 2  # the imported glTF frame (normal +Z) comes out with its normal along +y
STEP_NS = 10_000_000
CHUNK = 32  # frames per worker command in render_trajectories
# Third-person quadrotor model (Crazyflie-class: 92 mm motor to motor, 47 mm propellers), drawn
# DRONE_SCALE times true size by default so it reads clearly next to the 0.72 m gate frames.
DRONE_SCALE = 2.0
DRONE_FRAME_RGB = (0.09, 0.09, 0.10)  # body, arms, motors
DRONE_FRONT_RGB = (1.0, 0.12, 0.04)  # front rotor discs + nose
DRONE_REAR_RGB = (0.04, 0.32, 1.0)  # rear rotor discs
DRONE_HIDDEN_POS = (0.0, 0.0, -50.0)  # where the model waits until it gets a pose (under the floor)
# Axis map of the glTF importer (measured: boxes on the glTF +X/+Y/+Z axes of an object at identity
# pose appear along body +x/+z/-y): glTF vector = GLTF_FROM_BODY @ body-frame vector (body frame =
# contract frame of the object pose: x forward, y left, z up). The gate's GATE_MESH_YAW follows from it.
GLTF_FROM_BODY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# Where a third-person camera may go (contract frame): the nearest level geometry outside the floor
# is a block row from x = 18.1 m (up to z = 1.5 m); everything else is more than 30 m away.
CAMERA_BOUNDS = (np.array([-15.0, -15.0, 0.3]), np.array([16.0, 18.0, 15.0]))
TP_HFOV_DEG = 80.0  # default horizontal field of view of the third-person views
# Unreal console variables. r.AllowOcclusionQueries=0: occlusion culling uses the visibility found in
# the PREVIOUS render of the view, so after a jump from a view where a gate bar hid the scene, gates
# were missing for one frame; without it every frame depends only on its own pose.
# r.OneFrameThreadLag=0 does not change the image.
_BASE_CVARS = "r.OneFrameThreadLag=0,r.AllowOcclusionQueries=0"
LOOKS = {
    "stock": _BASE_CVARS,
    "fast": _BASE_CVARS + ",r.ShadowQuality=0",
}


# ----------------------------------------------------------------------------- frame conversion
def to_ned(pos, quat_xyzw):
    """Contract-frame body pose -> AirSim NED pose: (xyz, quaternion wxyz).

    The two frames differ by a half turn about x plus the SCENE_ORIGIN shift:
    p -> (px + ox, -(py + oy), -(pz + oz)),  q = (x, y, z, w) -> (w, x, -y, -z).
    """
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    ox, oy, oz = SCENE_ORIGIN
    return ((float(pos[0]) + ox, -(float(pos[1]) + oy), -(float(pos[2]) + oz)),
            (w / n, x / n, -y / n, -z / n))


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _quat_angle_deg(a, b):
    return math.degrees(2 * math.acos(min(1.0, abs(sum(p * q for p, q in zip(a, b))))))


def _yaw_quat_xyzw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


# ----------------------------------------------------------------------------- view geometry (numpy, batched)
def quat_to_mat(q):
    """xyzw quaternions [..., 4] -> rotation matrices [..., 3, 3] (columns = rotated x, y, z axes)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(R):
    """Rotation matrices [..., 3, 3] -> xyzw quaternions [..., 4] with w >= 0."""
    R = np.asarray(R, np.float64)
    m = R.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    for i, r in enumerate(m):
        t = np.trace(r)
        if t > 0:
            s = 2 * math.sqrt(1 + t)
            q[i] = ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4)
        else:
            k = int(np.argmax(np.diag(r)))
            j, l = (k + 1) % 3, (k + 2) % 3
            s = 2 * math.sqrt(max(1e-12, 1 + r[k, k] - r[j, j] - r[l, l]))
            v = np.zeros(3)
            v[k], v[j], v[l] = s / 4, (r[j, k] + r[k, j]) / s, (r[l, k] + r[k, l]) / s
            q[i] = (*v, (r[l, j] - r[j, l]) / s)
        if q[i, 3] < 0:
            q[i] = -q[i]
    return q.reshape(*R.shape[:-2], 4)


def look_at_quat(cam, target):
    """Camera orientation(s) [..., 4] xyzw: +x from ``cam`` towards ``target``, no roll (y level)."""
    f = np.asarray(target, np.float64) - np.asarray(cam, np.float64)
    f = f / np.linalg.norm(f, axis=-1, keepdims=True)
    left = np.cross(np.array([0.0, 0.0, 1.0]), f)
    left = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-9)
    return mat_to_quat(np.stack([f, left, np.cross(f, left)], -1))


def project_points(points, cam_pos, cam_quat, width, height, hfov_deg, pitch_up_deg=0.0):
    """World points [N,3] -> pixel coordinates [N,2] (integer = pixel centre) and depth [N].

    Camera at ``cam_pos``, carrier orientation ``cam_quat`` (xyzw, contract frame), optical axis =
    carrier +x pitched up by ``pitch_up_deg``, square pixels, horizontal field of view ``hfov_deg``.
    The FPV camera is ``hfov_deg=track.CAMERA["hfov_deg"], pitch_up_deg=track.CAMERA["pitch_up_deg"]``.
    """
    p = math.radians(pitch_up_deg)
    body_from_cam = np.array([[0, math.sin(p), math.cos(p)], [-1, 0, 0], [0, -math.cos(p), math.sin(p)]])
    pc = (np.asarray(points, float).reshape(-1, 3) - np.asarray(cam_pos, float)) @ (quat_to_mat(cam_quat) @ body_from_cam)
    th = math.tan(math.radians(hfov_deg) / 2)
    tv = th * height / width
    z = np.where(np.abs(pc[:, 2]) < 1e-9, 1e-9, pc[:, 2])
    u = width / 2 - 0.5 + (width / 2) / th * pc[:, 0] / z
    v = height / 2 - 0.5 + (height / 2) / tv * pc[:, 1] / z
    return np.stack([u, v], -1), pc[:, 2]


def _smooth(x, sigma):
    """Gaussian smoothing along axis 0 (sigma in samples), edges extended with the end values."""
    x = np.asarray(x, np.float64)
    if sigma <= 0:
        return x.copy()
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x.reshape(len(x), -1), ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(c, k, mode="valid") for c in xp.T], -1).reshape(x.shape)


def _gate_bars_local(margin=0.0):
    """The four bar boxes of a gate in its own frame (travel direction, left, up): (centre, half size)."""
    opening, bar = track.GATE_OPENING, (track.GATE_OUTER - track.GATE_OPENING) / 2
    c = opening / 2 + bar / 2
    return [(np.array([0.0, cl, cu]), np.array([GATE_DEPTH, sl, su]) / 2 + margin)
            for cl, cu, sl, su in ((0, c, track.GATE_OUTER, bar), (0, -c, track.GATE_OUTER, bar),
                                   (c, 0, bar, opening), (-c, 0, bar, opening))]


def _gate_bases(gates_pos, gates_yaw):
    for g, yaw in zip(np.asarray(gates_pos, np.float64), np.asarray(gates_yaw, np.float64)):
        yield g, np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])


def gate_frame_distance(points, gates_pos, gates_yaw):
    """Distance [...] from each point [..., 3] to the nearest gate frame (four bar boxes per gate) [m]."""
    P = np.asarray(points, np.float64)
    best = np.full(P.shape[:-1], np.inf)
    for g, B in _gate_bases(gates_pos, gates_yaw):
        local = (P - g) @ B
        for c, h in _gate_bars_local():
            best = np.minimum(best, np.linalg.norm(np.maximum(np.abs(local - c) - h, 0), axis=-1))
    return best


def gate_frame_blocks(p0, p1, gates_pos, gates_yaw, margin=0.0):
    """bool [...]: the straight segment p0 -> p1 ([..., 3], broadcast) passes through a gate bar
    (box grown by ``margin``). With p0 = camera and p1 = drone: the drone's centre is hidden."""
    p0, p1 = np.broadcast_arrays(np.asarray(p0, np.float64), np.asarray(p1, np.float64))
    hit = np.zeros(p0.shape[:-1], bool)
    for g, B in _gate_bases(gates_pos, gates_yaw):
        a, b = (p0 - g) @ B, (p1 - g) @ B
        d = b - a
        par = np.abs(d) < 1e-12
        for c, h in _gate_bars_local(margin):
            lo, hi = c - h, c + h
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - a) / d, (hi - a) / d
            inside = (a >= lo) & (a <= hi)
            tn = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2)).max(-1)
            tx = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2)).min(-1)
            hit |= (tx >= np.maximum(tn, 0.0)) & (tn <= 1.0)
    return hit


CHASE_HEIGHTS = np.round(np.arange(0.0, 0.451, 0.025), 3)  # camera heights above the flown path [m]


def chase_camera(pos, quat, gates_pos=None, gates_yaw=None, hz=50.0, follow=0.9, height=0.3, ahead=1.0,
                 yaw_sigma_s=0.3, path_sigma_s=0.06, clearance=0.1, max_rate=1.25, w_change=20.0,
                 height_sigma_s=0.1, min_z=0.15):
    """Chase-camera poses for a recorded flight -> (cam_pos [T,3], cam_quat [T,4], info).

    The camera flies along the drone's own (smoothed, ``path_sigma_s``) path, ``follow`` metres of
    path length behind it (before the start the path is extended straight back along the initial
    heading), at a height above that path, and looks at the point ``ahead`` m in front of the drone
    along its heading (body yaw smoothed non-causally with ``yaw_sigma_s``), level horizon. The drone
    passed every gate through its opening, so a camera on its path does too, and the line of sight
    (a short chord of that path) stays inside the opening as well.
    The height is chosen per frame from CHASE_HEIGHTS by dynamic programming: a height is excluded
    where a gate bar lies on the line from the camera to the drone or the camera is closer than
    ``clearance`` to a frame (penalised within a few frames of that, to leave room for the final
    smoothing); among the rest the path prefers ``height``, pays ``w_change`` per metre of change
    and changes by at most ``max_rate`` m/s; then it is smoothed (Gaussian, ``height_sigma_s``).
    So the camera rides ``height`` above the path between gates and sinks to the drone's line where
    it follows it through an opening. Use with ``mount_pitch_deg=0``.
    info: ``occluded_frac`` = fraction of frames whose drone centre is hidden by a gate bar
    (line of sight through a bar), ``min_gate_frame_distance_m``, height statistics.
    """
    pos = np.asarray(pos, np.float64)
    T = len(pos)
    fx = quat_to_mat(quat)[:, :, 0]
    yaw = _smooth(np.unwrap(np.arctan2(fx[:, 1], fx[:, 0])), yaw_sigma_s * hz)
    fwd = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(T)], -1)
    path = _smooth(pos, path_sigma_s * hz)
    back = path[0] - fwd[0] * (follow + 0.1) * np.linspace(1, 0, 12)[:, None]  # straight run-up before the start
    ext = np.concatenate([back[:-1], path])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ext, axis=0), axis=1))])
    s_cam = arc[len(back) - 1:] - follow
    base = np.stack([np.interp(s_cam, arc, ext[:, i]) for i in range(3)], -1)
    hs = CHASE_HEIGHTS
    info = {}
    if gates_pos is None:
        h = np.full(T, float(height))
    else:
        cam_c = base[:, None] + hs[None, :, None] * np.array([0.0, 0.0, 1.0])
        bad = gate_frame_blocks(cam_c, pos[:, None], gates_pos, gates_yaw) | (
            gate_frame_distance(cam_c, gates_pos, gates_yaw) < clearance)
        w = int(round(2.5 * height_sigma_s * hz))
        cs = np.concatenate([np.zeros((1, len(hs))), np.cumsum(bad, 0)])
        t_ = np.arange(T)
        near_bad = (cs[np.clip(t_ + w + 1, 0, T)] - cs[np.clip(t_ - w, 0, T)]) > 0
        unary = 1e4 * bad + 50.0 * near_bad + 10.0 * np.abs(hs - height)
        step = np.abs(hs[:, None] - hs[None])
        trans = np.where(step <= max_rate / hz + 1e-9, w_change * step, np.inf)
        cost, arg = unary[0].copy(), np.zeros((T, len(hs)), np.int32)
        for t in range(1, T):
            tot = cost[:, None] + trans
            arg[t] = tot.argmin(0)
            cost = tot[arg[t], np.arange(len(hs))] + unary[t]
        idx = np.empty(T, np.int64)
        idx[-1] = int(cost.argmin())
        for t in range(T - 1, 0, -1):
            idx[t - 1] = arg[t, idx[t]]
        h = _smooth(hs[idx], height_sigma_s * hz)
        info["frames_with_no_clear_height"] = round(float(bad.all(1).mean()), 4)
    cam = base + h[:, None] * np.array([0.0, 0.0, 1.0])
    cam[:, 2] = np.maximum(cam[:, 2], min_z)
    q = look_at_quat(cam, pos + ahead * fwd)
    if gates_pos is not None:
        info["occluded_frac"] = round(float(gate_frame_blocks(cam, pos, gates_pos, gates_yaw).mean()), 4)
        info["min_gate_frame_distance_m"] = round(float(gate_frame_distance(cam, gates_pos, gates_yaw).min()), 3)
    info["height_m_p10_median_p90"] = [round(float(v), 3) for v in np.percentile(h, [10, 50, 90])]
    info["distance_to_drone_m_min_median_max"] = [round(float(v), 3) for v in np.percentile(np.linalg.norm(pos - cam, axis=1), [0, 50, 100])]
    return cam, q, info


def overview_camera(points, width, height, hfov_deg=TP_HFOV_DEG, margin=0.05, elevations_deg=(40, 50, 60),
                    bounds=CAMERA_BOUNDS):
    """A fixed elevated camera that sees all ``points`` [N,3] (e.g. gate frames + flight path).

    Tries azimuths every 10 degrees and the listed elevations, puts the camera as close to the
    centre of the points' bounding box as it can while every point stays inside the image with a
    ``margin`` (fraction of the image size), and keeps the closest such camera (largest track in the
    image). ``bounds`` = (lo [3], hi [3]): allowed camera positions. -> (cam_pos [3], cam_quat [4]).
    """
    P = np.asarray(points, np.float64).reshape(-1, 3)
    centre = (P.min(0) + P.max(0)) / 2
    lo_px, hi_px = np.array([margin * width, margin * height]), np.array([(1 - margin) * width, (1 - margin) * height])

    def fits(cam):
        uv, z = project_points(P, cam, look_at_quat(cam, centre), width, height, hfov_deg)
        return z.min() > 0.3 and (uv >= lo_px).all() and (uv <= hi_px).all()

    best = None
    for el in np.radians(elevations_deg):
        for az in np.radians(np.arange(0, 360, 10)):
            u = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            lo, hi = 0.5, 200.0
            if not fits(centre + hi * u):
                continue
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if fits(centre + mid * u) else (mid, hi)
            cam = centre + hi * u
            if bounds is not None and not ((cam >= bounds[0]).all() and (cam <= bounds[1]).all()):
                continue
            if best is None or hi < best[0]:
                best = (hi, cam)
    if best is None:
        raise ValueError("no overview camera inside the bounds sees all points")
    return best[1], look_at_quat(best[1], centre)


# ----------------------------------------------------------------------------- gate mesh
def _box(cx, cy, cz, sx, sy, sz):
    v, n, idx = [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            u, w = (axis + 1) % 3, (axis + 2) % 3
            base = len(v)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[u], p[w] = sign * 0.5, a * 0.5, b * 0.5
                v.append([cx + p[0] * sx, cy + p[1] * sy, cz + p[2] * sz])
                nn = [0.0, 0.0, 0.0]
                nn[axis] = float(sign)
                n.append(nn)
            quad = [0, 1, 2, 0, 2, 3] if sign > 0 else [0, 2, 1, 0, 3, 2]
            idx += [base + q for q in quad]
    return v, n, idx


def _glb(prims) -> bytes:
    """Binary glTF with one mesh. prims: list of (V [n,3], N [n,3], I [m], glTF material dict)."""
    blob, views, accessors, primitives, materials = b"", [], [], [], []
    for V, N, I, material in prims:
        V, N, I = np.asarray(V, np.float32), np.asarray(N, np.float32), np.asarray(I, np.uint16)
        k = len(accessors)
        for arr, target in ((V, 34962), (N, 34962), (I, 34963)):
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes, "target": target})
            blob += arr.tobytes()
        blob += b"\0" * (-len(blob) % 4)
        accessors += [{"bufferView": k, "componentType": 5126, "count": len(V), "type": "VEC3",
                       "min": V.min(0).tolist(), "max": V.max(0).tolist()},
                      {"bufferView": k + 1, "componentType": 5126, "count": len(N), "type": "VEC3"},
                      {"bufferView": k + 2, "componentType": 5123, "count": len(I), "type": "SCALAR"}]
        primitives.append({"attributes": {"POSITION": k, "NORMAL": k + 1}, "indices": k + 2, "material": len(materials)})
        materials.append(material)
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": primitives}], "materials": materials, "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views, "accessors": accessors,
    }
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(blob)) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(blob), b"BIN\0") + blob)


def _material(rgb, emissive, roughness=0.8):
    return {"pbrMetallicRoughness": {"baseColorFactor": [*rgb, 1.0], "metallicFactor": 0.0, "roughnessFactor": roughness},
            "emissiveFactor": [emissive * c for c in rgb], "doubleSided": True}


def _merge(parts):
    V, N, I = [], [], []
    for v, n, i in parts:
        I += [len(V) + k for k in i]
        V += list(v)
        N += list(n)
    return np.array(V, np.float64), np.array(N, np.float64), np.array(I, np.int64)


def gate_glb(rgb=GATE_RGB) -> bytes:
    """Binary glTF of one gate: four bars around a square opening, in the X-Y plane, normal +Z."""
    opening, bar = track.GATE_OPENING, (track.GATE_OUTER - track.GATE_OPENING) / 2
    c, outer = opening / 2 + bar / 2, track.GATE_OUTER
    V, N, I = _merge(_box(cx, cy, 0, sx, sy, GATE_DEPTH)
                     for cx, cy, sx, sy in ((0, c, outer, bar), (0, -c, outer, bar), (c, 0, bar, opening), (-c, 0, bar, opening)))
    return _glb([(V, N, I, _material(tuple(rgb), 0.4))])


def _cylinder(cx, cy, cz, r, h, seg=20):
    """Closed cylinder along z: centre (cx, cy, cz), radius r, height h."""
    v, n, idx = [], [], []
    ang = 2 * np.pi * np.arange(seg) / seg
    for sign in (-1, 1):
        base = len(v)
        v.append([cx, cy, cz + sign * h / 2])
        n.append([0.0, 0.0, float(sign)])
        for a in ang:
            v.append([cx + r * math.cos(a), cy + r * math.sin(a), cz + sign * h / 2])
            n.append([0.0, 0.0, float(sign)])
        for k in range(seg):
            i1, i2 = base + 1 + k, base + 1 + (k + 1) % seg
            idx += [base, i1, i2] if sign > 0 else [base, i2, i1]
    base = len(v)
    for a in ang:
        for z in (cz - h / 2, cz + h / 2):
            v.append([cx + r * math.cos(a), cy + r * math.sin(a), z])
            n.append([math.cos(a), math.sin(a), 0.0])
    for k in range(seg):
        a0, b0 = base + 2 * k, base + 2 * ((k + 1) % seg)
        idx += [a0, b0, b0 + 1, a0, b0 + 1, a0 + 1]
    return v, n, idx


def drone_parts(scale=DRONE_SCALE):
    """Quadrotor model in the body frame (x forward, y left, z up, metres, origin = body origin).

    Returns [(name, V [n,3], N [n,3], I [m], rgb)] for the dark frame (body, arms, motors), the red
    front (front rotor discs + nose) and the blue rear (rear rotor discs). X layout, true size
    times ``scale``: 92 mm motor to motor, 47 mm discs.
    """
    a = 0.092 / 2 / math.sqrt(2)  # motor x and y offsets
    frame, front, rear = [_box(0, 0, 0.002, 0.036, 0.030, 0.012)], [_box(0.024, 0, 0.003, 0.012, 0.010, 0.008)], []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        v, n, i = _box(0, 0, 0, a * math.sqrt(2) + 0.004, 0.008, 0.005)
        ang = math.atan2(sy, sx)
        R = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
        frame.append(((np.array(v) @ R.T + [sx * a / 2, sy * a / 2, 0]).tolist(), (np.array(n) @ R.T).tolist(), i))
        frame.append(_cylinder(sx * a, sy * a, 0.003, 0.0045, 0.012))
        (front if sx > 0 else rear).append(_cylinder(sx * a, sy * a, 0.0105, 0.0235, 0.002, seg=24))
    out = []
    for name, parts, rgb in (("frame", frame, DRONE_FRAME_RGB), ("front", front, DRONE_FRONT_RGB), ("rear", rear, DRONE_REAR_RGB)):
        V, N, I = _merge(parts)
        out.append((name, V * scale, N, I, rgb))
    return out


def drone_glb(scale=DRONE_SCALE) -> bytes:
    """Binary glTF of the quadrotor model (`drone_parts`), axes mapped for the importer."""
    M = np.asarray(GLTF_FROM_BODY, np.float64)
    prims = []
    for name, V, N, I, rgb in drone_parts(scale):
        if np.linalg.det(M) < 0:  # a mirror flips the winding
            I = I.reshape(-1, 3)[:, ::-1].reshape(-1)
        prims.append((V @ M.T, N @ M.T, I, _material(rgb, 0.0 if name == "frame" else 0.35, 0.6)))
    return _glb(prims)


def _sim_configs(width, height, hfov_deg=None, pitch_up_deg=None):
    hfov_deg = track.CAMERA["hfov_deg"] if hfov_deg is None else hfov_deg
    pitch_up_deg = track.CAMERA["pitch_up_deg"] if pitch_up_deg is None else pitch_up_deg
    robot = {
        "physics-type": "non-physics",  # AirSim never integrates dynamics; the robot only moves on SetPose
        "links": [{"name": "Frame", "collision": {"enabled": False}}],
        "sensors": [{
            "id": "Fpv", "type": "camera", "enabled": True, "parent-link": "Frame",
            "capture-interval": 0.001,  # below the clock step, so every step captures
            "capture-settings": [{
                "image-type": 0, "width": width, "height": height, "fov-degrees": hfov_deg,
                "capture-enabled": True, "streaming-enabled": False, "pixels-as-float": False,
                "compress": False, "target-gamma": 2.5, "motion-blur-amount": 0.0,
            }],
            # NED body frame: +pitch = nose up
            "origin": {"xyz": "0.0 0.0 0.0", "rpy-deg": f"0 {pitch_up_deg} 0"},
        }],
    }
    scene = {
        "id": "SceneFpvCamera",
        "actors": [{"type": "robot", "name": "Cam", "origin": {"xyz": "-20.0 3.0 -3.0", "rpy-deg": "0 0 0"},
                    "robot-config": "robot.jsonc"}],
        "clock": {"type": "steppable", "step-ns": STEP_NS, "real-time-update-rate": 100000, "pause-on-start": True},
        "home-geo-point": {"latitude": 47.641468, "longitude": -122.140165, "altitude": 122.0},
        "scene-type": "UnrealNative",
    }
    return scene, robot


def _port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_ports(k):
    """k TCP ports that are free right now (picked by the kernel, so instances never collide on a
    fixed numbering with each other or with unrelated services on the machine)."""
    socks = [socket.socket() for _ in range(k)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ============================================================================= worker process
class _Worker:
    """One sim instance and its client. Runs in the AirSim venv; talks JSON lines over two pipes."""

    def __init__(self, a):
        self.a = a
        self.sim = None
        self.out = os.fdopen(a.fd_out, "w", buffering=1)
        self.inp = os.fdopen(a.fd_in, "r")
        self.frames = np.memmap(a.shm, np.uint8, "r+", shape=(a.n_slots, CHUNK, a.height, a.width, 3))[a.slot]
        self.cv = threading.Condition()
        self.want, self.msg = -1, None
        self.tilt = (math.cos(math.radians(a.mount_pitch) / 2), 0.0,
                     math.sin(math.radians(a.mount_pitch) / 2), 0.0)  # NED: +pitch = nose up
        self.drone_name = None

    # ---- sim process
    def launch_sim(self, attempts=3):
        a = self.a
        ini_dir = os.path.join(a.home, ".config/Epic/Blocks/Saved/Config/Linux")
        os.makedirs(ini_dir, exist_ok=True)
        # The Shipping build ignores -dpcvars/-ini:/XDG_CONFIG_HOME; it does read a user Engine.ini,
        # so each instance gets a private HOME.
        with open(os.path.join(ini_dir, "Engine.ini"), "w") as f:
            f.write("[SystemSettings]\n" + "\n".join(c for c in a.cvars.split(",") if c) + "\n")
        exe = os.path.join(a.root, "blocks/Blocks/Binaries/Linux/Blocks-Linux-Shipping")

        def die_with_parent():
            import ctypes
            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG

        for attempt in range(attempts):
            # The ports are free now but only bound by the sim a moment later; if something else
            # takes one in between, the sim exits and we try again with new ports.
            self.port_topics, self.port_services = _free_ports(2)
            cmd = [exe, "Blocks", "-RenderOffScreen", "-nosound", "-ResX=64", "-ResY=64",
                   f"-graphicsadapter={a.gpu}", f"-topicsport={self.port_topics}",
                   f"-servicesport={self.port_services}"]
            log = open(os.path.join(a.home, "sim.log"), "ab")
            self.sim = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        env=dict(os.environ, HOME=a.home), preexec_fn=die_with_parent)
            t0 = time.time()
            while self.sim.poll() is None and not _port_in_use(self.port_services):
                if time.time() - t0 > a.startup_timeout:
                    raise TimeoutError("sim did not open its service port")
                time.sleep(0.5)
            time.sleep(3)  # let the default level finish loading
            if self.sim.poll() is None:
                return
        raise RuntimeError(f"sim exited with code {self.sim.returncode} {attempts} times (GPU {a.gpu} unusable?)")

    def sim_gpu(self):
        """nvidia-smi index of the GPU the sim renders on (it may also be listed, idle, on others)."""
        out = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "m"], capture_output=True, text=True).stdout
        best = (-1, None)
        for line in out.splitlines():
            f = line.split()
            if len(f) > 3 and f[1] == str(self.sim.pid):
                best = max(best, (int(f[3]) if f[3].isdigit() else 0, int(f[0])))
        return best[1]

    # ---- client
    def connect(self, gates):
        from projectairsim import Drone, ProjectAirSimClient, World
        a = self.a
        scene, robot = _sim_configs(a.width, a.height, a.hfov, a.mount_pitch)
        json.dump(scene, open(os.path.join(a.home, "scene.jsonc"), "w"))
        json.dump(robot, open(os.path.join(a.home, "robot.jsonc"), "w"))
        self.client = ProjectAirSimClient(port_topics=self.port_topics, port_services=self.port_services)
        self.client.connect()
        self.world = World(self.client, "scene.jsonc", delay_after_load_sec=2, sim_config_path=a.home)
        self.robot = Drone(self.client, self.world, "Cam")
        self.client.subscribe(self.robot.sensors["Fpv"]["scene_camera"], self._on_image)
        self.world.pause()
        self.t = self.world.get_sim_time()
        glb = gate_glb()
        self.gates = [(list(p), float(y)) for p, y in zip(gates["pos"], gates["yaw"])]
        self.gate_names = [
            self.world.spawn_object_from_file(f"Gate{i}", "gltf", glb, True, self._gate_pose(p, y), [1.0, 1.0, 1.0], False)
            for i, (p, y) in enumerate(self.gates)]
        self.target, self.hi_name = -1, None
        if a.highlight:
            rgb = [float(v) for v in a.highlight.split(",")]
            self.hi_name = self.world.spawn_object_from_file("GateTarget", "gltf", gate_glb(rgb), True,
                                                             self._gate_pose(DRONE_HIDDEN_POS, 0.0), [1.0, 1.0, 1.0], False)
        if a.drone_scale > 0:
            self.drone_name = self.world.spawn_object_from_file(
                "Drone", "gltf", drone_glb(a.drone_scale), True, self._pose(*to_ned(DRONE_HIDDEN_POS, (0, 0, 0, 1))),
                [1.0, 1.0, 1.0], False)
        for _ in range(4):  # the first messages after a scene load are black
            self.render(gates["pos"][0], (0.0, 0.0, 0.0, 1.0))

    def _pose(self, ned_pos, ned_quat):
        from projectairsim.types import Pose, Quaternion, Vector3
        return Pose({
            "translation": Vector3({"x": ned_pos[0], "y": ned_pos[1], "z": ned_pos[2]}),
            "rotation": Quaternion({"w": ned_quat[0], "x": ned_quat[1], "y": ned_quat[2], "z": ned_quat[3]}),
            "frame_id": "DEFAULT_ID",
        })

    def _gate_pose(self, pos, yaw):
        return self._pose(*to_ned(pos, _yaw_quat_xyzw(yaw + GATE_MESH_YAW)))

    def _on_image(self, _topic, msg):
        with self.cv:
            if msg["time_stamp"] == self.want:
                self.msg = msg
                self.cv.notify()

    def set_pose(self, pos, quat_xyzw):
        ned_pos, ned_quat = to_ned(pos, quat_xyzw)
        self.robot.set_pose(self._pose(ned_pos, ned_quat), True)
        return ned_pos, ned_quat

    def set_drone(self, pos, quat_xyzw):
        if self.drone_name is None:
            raise RuntimeError("this renderer has no drone model (drone_scale=0)")
        self.world.set_object_pose(self.drone_name, self._pose(*to_ned(pos, quat_xyzw)), True)

    def set_target(self, k, force=False):
        """Show gate k in the highlight colour (k = -1: none): the magenta gate k goes under the
        floor and the highlight gate takes its pose; the previous target gets its magenta gate back.
        Object moves only, applied before the next clock step like every other pose."""
        k = int(k)
        if self.hi_name is None:
            raise RuntimeError("this renderer has no target highlight (highlight_target=False)")
        if k == self.target and not force:
            return
        if self.target >= 0 and self.target != k:
            self.world.set_object_pose(self.gate_names[self.target], self._gate_pose(*self.gates[self.target]), True)
        if k >= 0:
            self.world.set_object_pose(self.gate_names[k], self._gate_pose(DRONE_HIDDEN_POS, 0.0), True)
            self.world.set_object_pose(self.hi_name, self._gate_pose(*self.gates[k]), True)
        else:
            self.world.set_object_pose(self.hi_name, self._gate_pose(DRONE_HIDDEN_POS, 0.0), True)
        self.target = k

    def step(self):
        """Advance the clock by exactly one step; return the image message stamped with that step."""
        with self.cv:
            self.want, self.msg = self.t + STEP_NS, None
        self.t = self.world.continue_for_single_step()
        if self.t != self.want:
            raise RuntimeError(f"clock stepped to {self.t}, expected {self.want}")
        with self.cv:
            if not self.cv.wait_for(lambda: self.msg is not None, timeout=self.a.frame_timeout):
                raise TimeoutError(f"no frame for sim time {self.want}")
            return self.msg

    def _check(self, msg, ned_pos, ned_quat):
        m_pos = (msg["pos_x"], msg["pos_y"], msg["pos_z"])
        m_q = (msg["rot_w"], msg["rot_x"], msg["rot_y"], msg["rot_z"])
        if (max(abs(p - q) for p, q in zip(m_pos, ned_pos)) > 1e-3
                or _quat_angle_deg(m_q, _quat_mul(ned_quat, self.tilt)) > 0.1):
            raise RuntimeError(f"camera pose in the frame message {m_pos} {m_q} != commanded pose")

    @staticmethod
    def _rgb(msg):
        return np.frombuffer(msg["data"], np.uint8).reshape(msg["height"], msg["width"], 3)[..., ::-1]

    def render(self, pos, quat, dpos=None, dquat=None, target=None):
        if target is not None:
            self.set_target(target)
        if dpos is not None:
            self.set_drone(dpos, dquat)
        ned = self.set_pose(pos, quat)
        self.step()  # this step renders the pose; its message still holds the previous render
        msg = self.step()
        self._check(msg, *ned)
        self.frames[0] = self._rgb(msg)

    def render_stream(self, poses, quats, dposes=None, dquats=None, targets=None):
        ned = None
        for k, (pos, quat) in enumerate(zip(poses, quats)):
            if targets is not None:
                self.set_target(targets[k])
            if dposes is not None:
                self.set_drone(dposes[k], dquats[k])
            ned = self.set_pose(pos, quat)
            msg = self.step()
            self._check(msg, *ned)
            if k:
                self.frames[k - 1] = self._rgb(msg)
        msg = self.step()  # flush: same pose, fetch the last render
        self._check(msg, *ned)
        self.frames[len(poses) - 1] = self._rgb(msg)

    def probe(self, centre, size, res):
        """Level geometry around ``centre`` (contract frame): world-axis bounding boxes of all scene
        objects (NED, metres) and AirSim's occupancy voxel grid of the cube of side ``size``."""
        import base64

        boxes = {}
        for name in self.world.list_objects(".*"):
            try:
                boxes[name] = self.world.get_3d_bounding_box(name, 0)  # 0 = world-axis aligned
            except Exception as e:  # noqa: BLE001
                boxes[name] = {"error": f"{type(e).__name__}: {e}"}
        grid = self.world.create_voxel_grid(self._pose(*to_ned(centre, (0, 0, 0, 1))), size, size, size, res)
        return {"objects": boxes, "voxels_b64": base64.b64encode(np.packbits(np.asarray(grid, bool)).tobytes()).decode(),
                "voxels_n": len(grid)}

    def set_gates(self, pos, yaw):
        self.gates = [(list(p), float(y)) for p, y in zip(pos, yaw)]
        for i, (name, p, y) in enumerate(zip(self.gate_names, pos, yaw)):
            if i != self.target:
                self.world.set_object_pose(name, self._gate_pose(p, y), True)
        if self.target >= 0:  # the highlight follows its gate to the new layout
            self.set_target(self.target, force=True)

    # ---- command loop
    def run(self):
        try:
            init = json.loads(self.inp.readline())
            self.launch_sim()
            self.connect(init["gates"])
            self.out.write(json.dumps({"ok": 1, "sim_pid": self.sim.pid, "gpu": self.sim_gpu()}) + "\n")
            for line in self.inp:
                c = json.loads(line)
                t0 = time.perf_counter()
                extra = {}
                if c["op"] == "render":
                    self.render(c["pos"], c["quat"], c.get("dpos"), c.get("dquat"), c.get("target"))
                elif c["op"] == "traj":
                    self.render_stream(c["pos"], c["quat"], c.get("dpos"), c.get("dquat"), c.get("target"))
                elif c["op"] == "gates":
                    self.set_gates(c["pos"], c["yaw"])
                elif c["op"] == "probe":
                    extra = self.probe(c["centre"], c["size"], c["res"])
                elif c["op"] == "quit":
                    break
                self.out.write(json.dumps({"ok": 1, "ms": round(1e3 * (time.perf_counter() - t0), 3), **extra}) + "\n")
        except Exception as e:  # noqa: BLE001 - reported to the parent, which restarts this slot
            import traceback
            traceback.print_exc()
            try:
                self.out.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
            except Exception:  # noqa: BLE001
                pass
        finally:
            if self.sim is not None:
                _kill_group(self.sim.pid)
            os._exit(0)  # the client library keeps non-daemon threads alive


def _worker_main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    for name, typ in (("slot", int), ("n-slots", int), ("gpu", int), ("width", int), ("height", int),
                      ("fd-in", int), ("fd-out", int), ("cvars", str), ("root", str), ("home", str), ("shm", str),
                      ("startup-timeout", float), ("frame-timeout", float)):
        ap.add_argument(f"--{name}", type=typ, required=True)
    ap.add_argument("--hfov", type=float, default=track.CAMERA["hfov_deg"])
    ap.add_argument("--mount-pitch", type=float, default=track.CAMERA["pitch_up_deg"])
    ap.add_argument("--drone-scale", type=float, default=0.0)
    ap.add_argument("--highlight", default="", help="r,g,b of the target-gate highlight; empty = off")
    a = ap.parse_args(argv)
    os.chdir(a.home)  # the client library writes its log into the working directory
    _Worker(a).run()


# ============================================================================= caller side
class AirsimRenderer:
    """See the module docstring. Use as a context manager or call `close()`.

    Args:
        gpus: nvidia-smi indices of the GPUs to render on.
        instances_per_gpu: sim instances per GPU; ``n = len(gpus) * instances_per_gpu`` slots.
        width, height: frame size in pixels (16:9 for the contract camera; pixels are square and the
            horizontal field of view is `fpv.track.CAMERA["hfov_deg"]`).
        look: "stock" (dynamic shadows on) or "fast" (dynamic shadows off).
        gates_pos, gates_yaw: initial layout for all slots, [6,3] and [6]; default `fpv.track.CIRCUIT`.
        root: directory prepared by `scripts/setup_airsim.sh`.
        step_timeout: seconds a slot may take for one `render` (or one trajectory chunk) before it is
            declared stalled, killed and restarted with its gates re-applied.
        startup_timeout: seconds a sim may take to open its service port.
        startup_retries: a slot whose sim or client fails to start (no service port, scene load
            timeout, crash) is killed and launched again, alone, up to this many times; the
            renderer only fails if one slot fails ``startup_retries + 1`` times in a row.
            ``launches`` (slot -> launches so far) and ``failed_launches`` record what happened.
        hfov_deg, mount_pitch_deg: camera field of view and mount pitch; default the FPV camera of
            `fpv.track.CAMERA`. Third-person views use ``mount_pitch_deg=0``.
        drone_scale: > 0 adds the quadrotor model (`drone_parts`, this many times true size) to every
            instance; ``render*`` then accept ``drone_pos`` / ``drone_quat``.
        highlight_target: adds one gate in ``highlight_rgb`` per instance; ``render*`` then accept
            ``target`` (index of the gate to show highlighted, -1 = none) per slot / per frame: that
            gate's magenta copy is moved away and the highlight gate takes its place in the same
            clock step. Off: the scene and frames are exactly those of the plain FPV renderer.
    """

    def __init__(self, gpus, instances_per_gpu=8, width=320, height=180, look="stock", *, gates_pos=None,
                 gates_yaw=None, root=AIRSIM_ROOT, startup_timeout=150.0, step_timeout=30.0, max_restarts=2,
                 startup_retries=3, hfov_deg=None, mount_pitch_deg=None, drone_scale=0.0, highlight_target=False,
                 highlight_rgb=HIGHLIGHT_RGB):
        if look not in LOOKS:
            raise ValueError(f"look must be one of {sorted(LOOKS)}, got {look!r}")
        self.gpus, self.n = [int(g) for g in gpus], len(gpus) * int(instances_per_gpu)
        self.width, self.height, self.look = int(width), int(height), look
        self.root = str(root)
        self.startup_timeout, self.step_timeout, self.max_restarts = startup_timeout, step_timeout, max_restarts
        self.startup_retries = int(startup_retries)
        self.hfov_deg = float(track.CAMERA["hfov_deg"] if hfov_deg is None else hfov_deg)
        self.mount_pitch_deg = float(track.CAMERA["pitch_up_deg"] if mount_pitch_deg is None else mount_pitch_deg)
        self.drone_scale = float(drone_scale)
        self.highlight_target = bool(highlight_target)
        self.highlight_rgb = tuple(float(v) for v in highlight_rgb)
        self.restarts = 0  # number of watchdog restarts so far
        self.launches = {}  # slot -> sim launches so far (start-up retries and watchdog restarts included)
        self.failed_launches = []  # {"slot", "error", "log"} for every launch that failed
        self.last_ms = {}  # slot -> time the worker spent on its latest command [ms]
        self._closed = False
        self._procs, self._fds, self._bufs, self._sim_pids = {}, {}, {}, {}
        self._check_install()
        if gates_pos is None:
            gates_pos, gates_yaw = track.circuit_arrays()
        self._gates_pos = np.repeat(np.asarray(gates_pos, np.float64)[None], self.n, 0)
        self._gates_yaw = np.repeat(np.asarray(gates_yaw, np.float64)[None], self.n, 0)
        # scratch names carry the owning pid; remove what a killed process left behind (close() and
        # atexit cannot run on SIGKILL/SIGTERM)
        for path in Path("/tmp").glob("fpv_airsim_run_*"), Path("/dev/shm").glob("fpv_airsim_*.u8"):
            for p in path:
                pid = p.name.replace("fpv_airsim_run_", "").replace("fpv_airsim_", "").split("_")[0]
                try:
                    if pid.isdigit() and not os.path.exists(f"/proc/{pid}") and p.stat().st_uid == os.getuid():
                        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
                except OSError:
                    pass
        tag = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._run_dir = f"/tmp/fpv_airsim_run_{tag}"
        self._shm_path = f"/dev/shm/fpv_airsim_{tag}.u8"
        os.makedirs(self._run_dir)
        self._frames = np.memmap(self._shm_path, np.uint8, "w+", shape=(self.n, CHUNK, self.height, self.width, 3))
        atexit.register(self.close)
        try:
            for s in range(self.n):
                self._spawn(s)
            self._await_ready(range(self.n))
        except BaseException:
            self.close()
            raise

    # ---- lifecycle
    def _check_install(self):
        for p in ("blocks/Blocks/Binaries/Linux/Blocks-Linux-Shipping", "venv/bin/python"):
            if not os.path.exists(os.path.join(self.root, p)):
                raise RuntimeError(f"{os.path.join(self.root, p)} is missing: run scripts/setup_airsim.sh on this machine")
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"nvidia-smi failed ({e}); no usable GPU") from e
        have = {int(v) for v in out.split()}
        missing = [g for g in self.gpus if g not in have]
        if missing:
            raise RuntimeError(f"GPU(s) {missing} not available on this machine (nvidia-smi lists {sorted(have)})")

    def _spawn(self, s):
        home = os.path.join(self._run_dir, f"slot{s}")
        shutil.rmtree(home, ignore_errors=True)
        os.makedirs(home)
        c_r, c_w = os.pipe()  # commands: caller -> worker
        r_r, r_w = os.pipe()  # replies: worker -> caller
        cmd = [os.path.join(self.root, "venv/bin/python"), os.path.abspath(__file__), "--worker",
               "--slot", str(s), "--n-slots", str(self.n), "--gpu", str(self.gpus[s % len(self.gpus)]),
               "--width", str(self.width), "--height", str(self.height),
               "--fd-in", str(c_r), "--fd-out", str(r_w), "--cvars", LOOKS[self.look], "--root", self.root,
               "--home", home, "--shm", self._shm_path, "--startup-timeout", str(self.startup_timeout),
               "--frame-timeout", str(max(5.0, self.step_timeout / 2))]
        if (self.hfov_deg, self.mount_pitch_deg) != (track.CAMERA["hfov_deg"], track.CAMERA["pitch_up_deg"]):
            cmd += ["--hfov", str(self.hfov_deg), "--mount-pitch", str(self.mount_pitch_deg)]
        if self.drone_scale > 0:
            cmd += ["--drone-scale", str(self.drone_scale)]
        if self.highlight_target:
            cmd += ["--highlight", ",".join(repr(v) for v in self.highlight_rgb)]
        log = open(os.path.join(home, "worker.log"), "wb")
        self._procs[s] = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                          pass_fds=(c_r, r_w), start_new_session=True)
        os.close(c_r)
        os.close(r_w)
        self._fds[s], self._bufs[s] = (c_w, r_r), b""
        self._sim_pids.pop(s, None)
        self._send(s, {"gates": {"pos": self._gates_pos[s].tolist(), "yaw": self._gates_yaw[s].tolist()}})

    def _await_ready(self, slots):
        """Wait for freshly spawned slots; relaunch each failed one alone, up to `startup_retries` times."""
        pending = list(slots)
        for attempt in range(self.startup_retries + 1):
            failed = []
            for s, rep in self._collect(pending, self.startup_timeout + 60).items():
                self.launches[s] = self.launches.get(s, 0) + 1
                if rep is None or "error" in rep:
                    failed.append(s)
                    self.failed_launches.append({"slot": s, "error": rep["error"] if rep else "no reply in time / worker died",
                                                 "log": self._log_tail(s)[-600:]})
                    continue
                self._sim_pids[s] = rep["sim_pid"]
                want = self.gpus[s % len(self.gpus)]
                if rep["gpu"] is not None and rep["gpu"] != want:
                    raise RuntimeError(f"slot {s} renders on GPU {rep['gpu']} instead of GPU {want}")
            if not failed:
                return
            if attempt == self.startup_retries:
                last = self.failed_launches[-1]
                raise RuntimeError(f"slot(s) {failed} failed to start {attempt + 1} times: {last['error']}\n{last['log']}")
            for s in failed:
                self._kill(s)
                self._spawn(s)
            pending = failed

    def _kill(self, s):
        if s in self._sim_pids:
            _kill_group(self._sim_pids.pop(s))
        p = self._procs.pop(s, None)
        if p is not None:
            _kill_group(p.pid)
            p.wait()
        for fd in self._fds.pop(s, ()):
            try:
                os.close(fd)
            except OSError:
                pass

    def close(self):
        """Stop every sim and client process and remove the scratch files. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        for s in list(self._procs):
            try:
                self._send(s, {"op": "quit"})
            except OSError:
                pass
        t_end = time.time() + 5
        for p in self._procs.values():
            try:
                p.wait(max(0.0, t_end - time.time()))
            except subprocess.TimeoutExpired:
                pass
        for s in list(self._procs):
            self._kill(s)
        for pid in list(self._sim_pids.values()):
            _kill_group(pid)
        self._frames = None
        for path in (self._shm_path,):
            try:
                os.unlink(path)
            except OSError:
                pass
        shutil.rmtree(self._run_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ---- messaging
    def _send(self, s, obj):
        os.write(self._fds[s][0], (json.dumps(obj) + "\n").encode())

    def _collect(self, slots, timeout):
        """One reply per slot: {slot: dict | None}; None = no reply in time or the worker died."""
        out, t_end = {}, time.time() + timeout
        pending = {self._fds[s][1]: s for s in slots}
        while pending:
            ready, _, _ = select.select(list(pending), [], [], max(0.0, t_end - time.time()))
            if not ready:
                break
            for fd in ready:
                s = pending[fd]
                data = os.read(fd, 65536)
                self._bufs[s] += data
                if b"\n" in self._bufs[s]:
                    line, self._bufs[s] = self._bufs[s].split(b"\n", 1)
                    out[s] = json.loads(line)
                    del pending[fd]
                elif not data:  # EOF: worker died
                    out[s] = None
                    del pending[fd]
        for s in pending.values():
            out[s] = None
        return out

    def _log_tail(self, s, n=15):
        try:
            lines = open(os.path.join(self._run_dir, f"slot{s}", "worker.log"), errors="replace").read().splitlines()
            return "\n".join(ln for ln in lines if "[INFO]" not in ln)[-2000:] if lines else ""
        except OSError:
            return ""

    def _call(self, cmds, timeout):
        """Run {slot: command} on all slots at once; restart stalled/failed slots and retry them.

        Any exception (including KeyboardInterrupt) closes the renderer: replies may be in flight,
        so the instances cannot be reused safely, and no sim must be left behind.
        """
        if self._closed:
            raise RuntimeError("renderer is closed")
        try:
            return self._call_all(cmds, timeout)
        except BaseException:
            self.close()
            raise

    def _call_all(self, cmds, timeout):
        todo, replies = dict(cmds), {}
        for attempt in range(self.max_restarts + 1):
            for s, c in todo.items():
                self._send(s, c)
            got = self._collect(list(todo), timeout)
            bad = [s for s, rep in got.items() if rep is None or "error" in rep]
            replies.update({s: rep for s, rep in got.items() if s not in bad})
            self.last_ms.update({s: rep.get("ms") for s, rep in got.items() if s not in bad})
            if not bad:
                return replies
            if attempt == self.max_restarts:
                raise RuntimeError(f"slots {bad} failed {attempt + 1} times: "
                                   f"{[got[s] and got[s]['error'] for s in bad]}\n{self._log_tail(bad[0])}")
            for s in bad:  # watchdog: fresh sim + client, gates re-applied by the worker at start-up
                self._kill(s)
                self._spawn(s)
            self._await_ready(bad)
            self.restarts += len(bad)
            todo = {s: cmds[s] for s in bad}
        return replies

    # ---- API
    def set_gates(self, slots, gates_pos, gates_yaw):
        """Move the six gates of each listed slot. slots int [k], gates_pos [k,6,3], gates_yaw [k,6]."""
        slots = np.atleast_1d(np.asarray(slots, np.int64))
        gp = np.asarray(gates_pos, np.float64).reshape(len(slots), 6, 3)
        gy = np.asarray(gates_yaw, np.float64).reshape(len(slots), 6)
        self._gates_pos[slots], self._gates_yaw[slots] = gp, gy
        self._call({int(s): {"op": "gates", "pos": gp[i].tolist(), "yaw": gy[i].tolist()}
                    for i, s in enumerate(slots)}, self.step_timeout)

    def render(self, pos, quat_xyzw, active=None, drone_pos=None, drone_quat=None, target=None):
        """One frame per slot from the body poses of THIS call.

        pos [n,3], quat_xyzw [n,4] (body->world, contract frame), active bool [n] or None (= all).
        drone_pos [n,3], drone_quat [n,4]: pose of the quadrotor model in this frame (only with
        ``drone_scale > 0``; the model stays where it was if omitted).
        target int [n]: gate shown in the highlight colour in this frame, -1 = none (only with
        ``highlight_target``; the highlight stays as it was if omitted).
        Returns uint8 [n,H,W,3] RGB; inactive slots are zeros and are not rendered.
        """
        pos = np.asarray(pos, np.float64).reshape(self.n, 3)
        quat = np.asarray(quat_xyzw, np.float64).reshape(self.n, 4)
        act = np.ones(self.n, bool) if active is None else np.asarray(active, bool).reshape(self.n)
        dpos, dquat = self._drone_args(drone_pos, drone_quat, (self.n,))
        tgt = self._target_args(target, (self.n,))
        out = np.zeros((self.n, self.height, self.width, 3), np.uint8)
        idx = np.flatnonzero(act)
        if len(idx):
            cmds = {}
            for s in idx:
                cmds[int(s)] = {"op": "render", "pos": pos[s].tolist(), "quat": quat[s].tolist()}
                if dpos is not None:
                    cmds[int(s)].update(dpos=dpos[s].tolist(), dquat=dquat[s].tolist())
                if tgt is not None:
                    cmds[int(s)]["target"] = int(tgt[s])
            self._call(cmds, self.step_timeout)
            out[idx] = self._frames[idx, 0]
        return out

    def _target_args(self, target, shape):
        if target is None:
            return None
        if not self.highlight_target:
            raise ValueError("target needs a renderer built with highlight_target=True")
        t = np.asarray(target, np.int64).reshape(shape)
        if ((t < -1) | (t >= 6)).any():
            raise ValueError("target must be a gate index 0..5 or -1")
        return t

    def _drone_args(self, drone_pos, drone_quat, lead):
        if drone_pos is None:
            return None, None
        if self.drone_scale <= 0:
            raise ValueError("drone poses need a renderer built with drone_scale > 0")
        return (np.asarray(drone_pos, np.float64).reshape(*lead, 3), np.asarray(drone_quat, np.float64).reshape(*lead, 4))

    def probe_scene(self, slot, centre, size, res):
        """Level geometry of one slot's scene (diagnostics, see `_Worker.probe`): dict with
        ``objects`` (name -> world-axis bounding box, AirSim NED metres) and ``voxels`` (bool
        occupancy grid of the cube of side ``size`` m centred on ``centre`` (contract frame),
        flat in AirSim's order)."""
        import base64

        rep = self._call({int(slot): {"op": "probe", "centre": [float(v) for v in centre], "size": float(size),
                                      "res": float(res)}}, max(self.step_timeout, 600.0))[int(slot)]
        bits = np.unpackbits(np.frombuffer(base64.b64decode(rep["voxels_b64"]), np.uint8))[: rep["voxels_n"]]
        return {"objects": rep["objects"], "voxels": bits.astype(bool)}

    def render_trajectories(self, slots, pos, quat_xyzw, drone_pos=None, drone_quat=None, target=None):
        """Open-loop rendering of whole trajectories, one capture per frame.

        slots int [k] (distinct), pos [k,T,3], quat_xyzw [k,T,4] -> uint8 [k,T,H,W,3] RGB.
        drone_pos [k,T,3], drone_quat [k,T,4]: quadrotor model pose per frame (``drone_scale > 0``).
        target int [k,T]: highlighted gate per frame, -1 = none (``highlight_target``).
        The result is one array (k * T * H * W * 3 bytes); split T on the caller's side if that is
        too large. Frame t of slot j is rendered from pose t of slot j with that slot's current gates.
        """
        slots = np.atleast_1d(np.asarray(slots, np.int64))
        if len(set(slots.tolist())) != len(slots):
            raise ValueError("slots must be distinct")
        pos = np.asarray(pos, np.float64).reshape(len(slots), -1, 3)
        quat = np.asarray(quat_xyzw, np.float64).reshape(len(slots), -1, 4)
        T = pos.shape[1]
        dpos, dquat = self._drone_args(drone_pos, drone_quat, (len(slots), T))
        tgt = self._target_args(target, (len(slots), T))
        out = np.empty((len(slots), T, self.height, self.width, 3), np.uint8)
        for t0 in range(0, T, CHUNK):
            t1 = min(T, t0 + CHUNK)
            cmds = {}
            for i, s in enumerate(slots):
                cmds[int(s)] = {"op": "traj", "pos": pos[i, t0:t1].tolist(), "quat": quat[i, t0:t1].tolist()}
                if dpos is not None:
                    cmds[int(s)].update(dpos=dpos[i, t0:t1].tolist(), dquat=dquat[i, t0:t1].tolist())
                if tgt is not None:
                    cmds[int(s)]["target"] = tgt[i, t0:t1].tolist()
            self._call(cmds, self.step_timeout * 4)
            out[:, t0:t1] = self._frames[slots, : t1 - t0]
        return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2:])
