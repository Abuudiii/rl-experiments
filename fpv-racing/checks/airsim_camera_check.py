#!/usr/bin/env python
"""Feasibility check: Project AirSim (v1.0.1 Blocks, UE 5.2) as a pose-driven FPV camera.

AirSim is used as a renderer only: a "non-physics" robot carrying one RGB camera is
teleported with SetPose and the frame rendered from that pose is read back (see FpvCam for
the protocols and the one-capture pixel lag they compensate).

  python airsim_camera_check.py all --gpu 4      # pose + sync + gates + single-instance throughput
  python airsim_camera_check.py pose|sync|gates|throughput|survey --gpu 4
  python airsim_camera_check.py multi --gpu 4 --instances 1 2 4 8 --res 160x120
  options: --mode free|pipelined|double, --cvars name=value,... (Unreal console variables)

Results are merged into out/airsim_results.json; frames and contact sheets go to out/.

Run from this directory with the venv that has the `projectairsim` client installed.
"""
import argparse
import collections
import json
import math
import os
import signal
import socket
import struct
import subprocess
import shlex
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import Pose, Quaternion, Vector3
from projectairsim.utils import load_scene_config_as_dict, unpack_image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BLOCKS_ROOT = os.environ.get("AIRSIM_BLOCKS", "/tmp/fpv_airsim/blocks")
CAPTURE_OVERRIDE = "{}"  # JSON merged into the camera capture-settings (set by --capture)
# Unreal console variables for the sim, "name=value,name=value" (set by --cvars). The Shipping build
# ignores -dpcvars/-ini:/-ExecCmds, but it does read a user Engine.ini, so launch_sim writes one
# into a private HOME (XDG_CONFIG_HOME is not honoured).
# Default: dynamic shadows off (they cost ~40% of the frame rate) and no one-frame lag between the
# game and render threads (lets render_free() use one Unreal frame per pose instead of two).
# Pass --cvars= (empty) for the stock look.
RENDER_CVARS = "r.ShadowQuality=0,r.OneFrameThreadLag=0"
CAM_TILT_DEG = 20.0  # must match the sensor origin in sim_config/robot_fpv_camera.jsonc
CAM_HFOV_DEG = 110.0  # must match fov-degrees in sim_config/robot_fpv_camera.jsonc
REF_TRACK = [  # the race circuit from the project contract (x fwd, y left, z up; yaw = travel direction)
    {"pos": [2, 0, 1.0], "yaw": 0},
    {"pos": [5, 0.5, 1.3], "yaw": 0.5},
    {"pos": [6.5, 3, 1.0], "yaw": 1.571},
    {"pos": [5, 5.5, 1.4], "yaw": 2.6},
    {"pos": [2, 5.5, 1.0], "yaw": 3.142},
    {"pos": [0, 3, 1.2], "yaw": -1.571},
]
GATE_OPENING, GATE_BAR, GATE_DEPTH = 0.4, 0.16, 0.05  # outer size = 0.4 + 2 * 0.16 = 0.72 m


# ----------------------------------------------------------------------------- frames
def quat_from_rpy(roll, pitch, yaw):
    """Right-handed ZYX euler -> quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


# Where the project-frame origin (ground level) sits in the Blocks level, in project axes relative to
# AirSim's NED origin. Measured with `survey`: the Blocks floor is 1.0 m above the NED origin, and a
# 2.5 m high slab covers x in [-2, 10), y in [-10, 12) there, so the track is moved to open floor.
SCENE_ORIGIN = (-20.0, -3.0, 1.0)


def to_airsim(pos, rpy=(0.0, 0.0, 0.0)):
    """Project frame -> AirSim frame.

    Project: x forward, y left, z up; rpy are right-handed rotations about those axes
    (yaw > 0 turns left, pitch > 0 is nose DOWN, roll > 0 is right side down).
    AirSim: NED (x north, y east, z down). The two differ by a 180 deg turn about x (plus the
    SCENE_ORIGIN shift), so position p -> (px + ox, -(py + oy), -(pz + oz)) and
    quaternion (w, x, y, z) -> (w, x, -y, -z).
    Returns (ned_xyz, ned_quat_wxyz).
    """
    w, x, y, z = quat_from_rpy(*rpy)
    ox, oy, oz = SCENE_ORIGIN
    return (pos[0] + ox, -(pos[1] + oy), -(pos[2] + oz)), (w, x, -y, -z)


def look_at(pos, target):
    """Project-frame body rpy that points the (tilted) camera axis from pos at target."""
    d = np.subtract(target, pos)
    yaw = math.atan2(d[1], d[0])
    elev = math.atan2(d[2], math.hypot(d[0], d[1]))
    return (0.0, -(elev - math.radians(CAM_TILT_DEG)), yaw)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_angle_deg(a, b):
    dot = abs(sum(p * q for p, q in zip(a, b)))
    return math.degrees(2 * math.acos(min(1.0, dot)))


# ----------------------------------------------------------------------------- pinhole model
def cam_rotation(rpy):
    """World<-camera rotation (project frame) for body rpy; camera axes are x right, y down, z forward."""
    r, p, y = rpy
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    t = math.radians(CAM_TILT_DEG)
    body_from_cam = np.array([[0, math.sin(t), math.cos(t)], [-1, 0, 0], [0, -math.cos(t), math.sin(t)]])
    return rz @ ry @ rx @ body_from_cam


def intrinsics(width, height):
    f = (width / 2) / math.tan(math.radians(CAM_HFOV_DEG) / 2)
    return np.array([[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1]])


def project(points, pos, rpy, width, height):
    """Project-frame world points -> (pixel coords [N,2] with integer = pixel centre, depth [N])."""
    pc = (np.asarray(points, float) - np.asarray(pos, float)) @ cam_rotation(rpy)
    uv = (pc / pc[:, 2:3]) @ intrinsics(width, height).T
    return uv[:, :2], pc[:, 2]


def contact_sheet(tiles, path, cols=4, size=(320, 240)):
    """tiles: [(label, BGR image)] -> one labelled grid image."""
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * size[1], cols * size[0], 3), np.uint8)
    for k, (label, img) in enumerate(tiles):
        t = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(t, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        r, c = divmod(k, cols)
        sheet[r * size[1]:(r + 1) * size[1], c * size[0]:(c + 1) * size[0]] = t
    cv2.imwrite(path, sheet)


# ----------------------------------------------------------------------------- sim process
def ports(idx):
    return 8989 + 10 * idx, 8990 + 10 * idx


def launch_sim(gpu, idx=0, sim_args=()):
    """Start one headless Blocks instance on Vulkan adapter `gpu`; returns the Popen."""
    tp, sp = ports(idx)
    exe = os.path.join(BLOCKS_ROOT, "Blocks/Binaries/Linux/Blocks-Linux-Shipping")
    cmd = [exe, "Blocks", "-RenderOffScreen", "-nosound", "-ResX=64", "-ResY=64",
           f"-graphicsadapter={gpu}", f"-topicsport={tp}", f"-servicesport={sp}", *sim_args]
    log = open(f"/tmp/fpv_airsim_sim{idx}.log", "wb")
    cfg_home = f"/tmp/fpv_airsim_home{idx}"
    ini_dir = os.path.join(cfg_home, ".config/Epic/Blocks/Saved/Config/Linux")
    os.makedirs(ini_dir, exist_ok=True)
    with open(os.path.join(ini_dir, "Engine.ini"), "w") as f:
        f.write("[SystemSettings]\n" + "\n".join(c for c in RENDER_CVARS.split(",") if c) + "\n")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env=dict(os.environ, HOME=cfg_home))
    return proc


def wait_port(port, proc, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"sim exited with code {proc.returncode}")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                time.sleep(3)  # let the default scene finish loading
                return
        time.sleep(0.5)
    raise TimeoutError(f"port {port} not listening")


def kill_sims(procs):
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for p in procs:
        p.wait()


def cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def gpu_of_pids(pids):
    """Which nvidia-smi GPU index each pid actually renders on (from `nvidia-smi pmon`)."""
    out = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "m"], capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) > 3 and f[1].isdigit() and int(f[1]) in pids:
            found[int(f[1])] = {"gpu": int(f[0]), "type": f[2], "fb_mib": f[3]}
    return found


class GpuSampler(threading.Thread):
    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu, self.samples, self.stop = gpu, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            out = subprocess.run(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            try:
                self.samples.append([float(v) for v in out.strip().split(",")])
            except ValueError:
                pass
            self.stop.wait(0.5)

    def summary(self):
        self.stop.set()
        a = np.array(self.samples) if self.samples else np.zeros((1, 2))
        return {"gpu_util_mean": round(float(a[:, 0].mean()), 1), "gpu_util_max": float(a[:, 0].max()),
                "gpu_mem_mib_max": float(a[:, 1].max())}


# ----------------------------------------------------------------------------- client
class FpvCam:
    """Pose in -> frame out.

    Measured quirk of AirSim 1.0.1: every image message carries the camera pose of the capture it
    was published for, but the PIXELS rendered by the previous capture (UnrealCamera::TickComponent
    queues the render with CaptureSceneDeferred, which runs at the end of the frame, and enqueues
    the read-back before it; the first message after a scene load is black). Protocols:
      render()        = lock-step: set pose, step the clock twice, keep the second message
      render_stream() = lock-step: set pose k, step once, pair message k+1 with pose k
      render_free()   = clock running freely: set pose k, wait for the first message whose pose
                        metadata equals pose k, take the pixels of the message after it
    """

    def __init__(self, idx=0, width=160, height=120, depth=False, capture=None):
        tp, sp = ports(idx)
        self.client = ProjectAirSimClient(port_topics=tp, port_services=sp)
        self.client.connect()
        # Patch the resolution into a temp copy of the config pair, then load it.
        cfg, _ = load_scene_config_as_dict("scene_fpv_camera.jsonc", os.path.join(HERE, "sim_config"))
        robot = cfg["actors"][0]["robot-config"]
        capture = dict(capture or {})
        cfg["clock"].update(capture.pop("clock", {}))  # "clock"/"capture-interval" are not capture keys
        if "capture-interval" in capture:
            robot["sensors"][0]["capture-interval"] = capture.pop("capture-interval")
        robot["sensors"][0]["capture-settings"][0].update(width=width, height=height, **capture)
        self.depth = depth
        if depth:  # planar depth in metres instead of RGB (used only to survey the scene)
            robot["sensors"][0]["capture-settings"][0].update({"image-type": 1, "pixels-as-float": True})
        self.step_ns = cfg["clock"]["step-ns"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg["actors"][0]["robot-config"] = "robot.jsonc"
            json.dump(cfg, open(os.path.join(tmp, "scene.jsonc"), "w"))
            json.dump(robot, open(os.path.join(tmp, "robot.jsonc"), "w"))
            self.world = World(self.client, "scene.jsonc", delay_after_load_sec=2, sim_config_path=tmp)
        self.robot = Drone(self.client, self.world, "Cam")
        self.cv = threading.Condition()
        self.want, self.msg, self.n_msgs, self.stray = -1, None, 0, 0
        self.free, self.queue = False, collections.deque()
        topic = "depth_planar_camera" if depth else "scene_camera"
        self.client.subscribe(self.robot.sensors["Fpv"][topic], self._on_image)
        self.world.pause()
        self.t = self.world.get_sim_time()
        self.timing = np.zeros(3)  # seconds spent in: set_pose, clock step, waiting for the frame

    def _on_image(self, _topic, msg):
        with self.cv:
            self.n_msgs += 1
            if self.free:
                self.queue.append(msg)
                self.cv.notify()
            elif msg["time_stamp"] == self.want:
                self.msg = msg
                self.cv.notify()
            else:
                self.stray += 1  # a frame we did not ask for (would be a duplicate/late frame)

    def set_pose(self, ned_pos, ned_quat):
        t0 = time.perf_counter()
        self.robot.set_pose(Pose({
            "translation": Vector3({"x": float(ned_pos[0]), "y": float(ned_pos[1]), "z": float(ned_pos[2])}),
            "rotation": Quaternion({"w": float(ned_quat[0]), "x": float(ned_quat[1]),
                                    "y": float(ned_quat[2]), "z": float(ned_quat[3])}),
        }), True)
        self.timing[0] += time.perf_counter() - t0

    def step(self):
        """Advance exactly one clock step and return (BGR image, message) published for it."""
        t1 = time.perf_counter()
        with self.cv:
            self.want, self.msg = self.t + self.step_ns, None
        self.t = self.world.continue_for_single_step()
        t2 = time.perf_counter()
        with self.cv:
            if not self.cv.wait_for(lambda: self.msg is not None, timeout=20):
                raise TimeoutError(f"no frame for sim time {self.want} (sim time now {self.t})")
            msg = self.msg
        self.timing[1:] += (t2 - t1, time.perf_counter() - t2)
        img = unpack_image(msg) if self.depth else msg_to_img(msg)
        return img, msg

    def render_naive(self, ned_pos, ned_quat):
        """What the API suggests (set pose, one step, take that step's image). One frame late."""
        self.set_pose(ned_pos, ned_quat)
        return self.step()

    def render(self, ned_pos, ned_quat):
        """One correct frame on demand: the second step's message holds the pixels of this pose."""
        self.set_pose(ned_pos, ned_quat)
        self.step()
        return self.step()

    def render_stream(self, poses):
        """Yield one image per (ned_pos, ned_quat) in order, using one clock step per pose."""
        pending = False
        for ned_pos, ned_quat in poses:
            self.set_pose(ned_pos, ned_quat)
            img, _ = self.step()
            if pending:
                yield img
            pending = True
        if pending:
            yield self.step()[0]  # flush: no pose change, just fetch the last render

    def render_free(self, poses, lag=1):
        """Yield one image per (ned_pos, ned_quat) in order while the sim clock runs freely.

        Frames are matched to poses by the pose metadata in each message, not by time: the pixels
        of pose k are in the message `lag` (=1, see class docstring) after the first message whose
        metadata equals pose k. Pose k+1 is sent as soon as that first message is seen.
        """
        tilt = quat_from_rpy(0, math.radians(CAM_TILT_DEG), 0)
        with self.cv:
            self.queue.clear()
            self.free = True
        self.world.resume()
        seq, due = 0, set()

        def next_msg():
            t0 = time.perf_counter()
            with self.cv:
                if not self.cv.wait_for(lambda: self.queue, timeout=20):
                    raise TimeoutError("no frame while free-running")
                msg = self.queue.popleft()
            self.timing[2] += time.perf_counter() - t0
            return msg

        try:
            for ned_pos, ned_quat in poses:
                self.set_pose(ned_pos, ned_quat)
                want_q = quat_mul(ned_quat, tilt)
                while True:
                    msg, seq = next_msg(), seq + 1
                    if seq in due:
                        due.discard(seq)
                        yield msg_to_img(msg)
                    m_pos, m_q = msg_pose(msg)
                    if np.abs(np.subtract(m_pos, ned_pos)).max() < 1e-4 and quat_angle_deg(m_q, want_q) < 0.05:
                        if lag == 0:
                            yield msg_to_img(msg)
                        else:
                            due.add(seq + lag)
                        break
            while due:
                msg, seq = next_msg(), seq + 1
                if seq in due:
                    due.discard(seq)
                    yield msg_to_img(msg)
        finally:
            self.world.pause()
            time.sleep(0.3)  # let in-flight frames drain before going back to lock-step
            with self.cv:
                self.free = False
            self.t = self.world.get_sim_time()

    def render_flu(self, pos, rpy):
        return self.render(*to_airsim(pos, rpy))

    def close(self):
        self.client.disconnect()


def msg_pose(msg):
    return (msg["pos_x"], msg["pos_y"], msg["pos_z"]), (msg["rot_w"], msg["rot_x"], msg["rot_y"], msg["rot_z"])


def msg_to_img(msg):
    return np.frombuffer(msg["data"], np.uint8).reshape(msg["height"], msg["width"], 3)


# ----------------------------------------------------------------------------- 1. pose test
BASE = (0.0, 0.0, 2.0)
POSES = [  # name, project-frame position, project-frame rpy in degrees
    ("base", BASE, (0, 0, 0)),
    ("yaw_right_30", BASE, (0, 0, -30)),
    ("yaw_left_30", BASE, (0, 0, 30)),
    ("nose_up_25", BASE, (0, -25, 0)),
    ("nose_down_25", BASE, (0, 25, 0)),
    ("roll_right_30", BASE, (30, 0, 0)),
    ("up_4m", (0.0, 0.0, 6.0), (0, 0, 0)),
    ("left_3m", (0.0, 3.0, 2.0), (0, 0, 0)),
    ("forward_5m", (5.0, 0.0, 2.0), (0, 0, 0)),
    ("combined", (-3.0, 2.0, 4.0), (20, 10, -45)),
]


def match_points(a, b):
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    return (np.float64([ka[m.queryIdx].pt for m in good]), np.float64([kb[m.trainIdx].pt for m in good]))


def rot_angle_deg(r):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def image_rotation_check(a, b, rpy_a, rpy_b):
    """Camera rotation recovered from feature matches (pure rotation => H = K R K^-1) vs commanded."""
    k = intrinsics(a.shape[1], a.shape[0])
    pa, pb = match_points(a, b)
    h, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    u, _, vt = np.linalg.svd(np.linalg.inv(k) @ h @ k)
    r_img = u @ vt * np.sign(np.linalg.det(u @ vt))
    r_cmd = cam_rotation(rpy_b).T @ cam_rotation(rpy_a)  # camera-a coords -> camera-b coords
    rv_img, rv_cmd = cv2.Rodrigues(r_img)[0].ravel(), cv2.Rodrigues(r_cmd)[0].ravel()
    return {"matches": int(inl.sum()), "commanded_rotvec_deg": np.degrees(rv_cmd).round(2).tolist(),
            "image_rotvec_deg": np.degrees(rv_img).round(2).tolist(),
            "rotation_err_deg": round(rot_angle_deg(r_img @ r_cmd.T), 3),
            "mean_feature_shift_px": (pb - pa)[inl.ravel() > 0].mean(0).round(1).tolist()}


def image_translation_check(a, b, pos_a, pos_b, rpy):
    """Direction of camera translation recovered from the essential matrix vs commanded."""
    k = intrinsics(a.shape[1], a.shape[0])
    pa, pb = match_points(a, b)
    e, inl = cv2.findEssentialMat(pa, pb, k, cv2.RANSAC, 0.999, 1.0)
    _, r, t, _ = cv2.recoverPose(e, pa, pb, k, mask=inl)
    t_cmd = cam_rotation(rpy).T @ np.subtract(pos_a, pos_b)  # x_b = R x_a + t
    t_cmd /= np.linalg.norm(t_cmd)
    return {"matches": int((inl > 0).sum()), "commanded_dir_cam": t_cmd.round(3).tolist(),
            "image_dir_cam": t.ravel().round(3).tolist(),
            "direction_err_deg": round(math.degrees(math.acos(min(1.0, float(t.ravel() @ t_cmd)))), 2),
            "spurious_rotation_deg": round(rot_angle_deg(r), 3)}


def test_pose(cam):
    tilt = quat_from_rpy(0, math.radians(CAM_TILT_DEG), 0)  # NED: +pitch = nose up
    rows, frames = [], {}
    for name, pos, rpy_deg in POSES:
        ned_pos, ned_q = to_airsim(pos, [math.radians(a) for a in rpy_deg])
        img, msg = cam.render(ned_pos, ned_q)
        gt = cam.robot.get_ground_truth_pose()
        gt_pos = [gt["translation"][k] for k in "xyz"]
        gt_q = [gt["rotation"][k] for k in "wxyz"]
        m_pos, m_q = msg_pose(msg)
        rows.append({
            "pose": name,
            "robot_pos_err_m": float(np.abs(np.subtract(gt_pos, ned_pos)).max()),
            "robot_rot_err_deg": quat_angle_deg(gt_q, ned_q),
            "frame_pos_err_m": float(np.abs(np.subtract(m_pos, ned_pos)).max()),
            "frame_rot_err_deg": quat_angle_deg(m_q, quat_mul(ned_q, tilt)),
        })
        frames[name] = img.copy()
        cv2.imwrite(os.path.join(OUT, f"pose_{name}.png"), img)
        print(json.dumps(rows[-1]))
    # the pose must hold with no further commands: step 50 times and read it back
    for _ in range(50):
        _, msg = cam.step()
    gt = cam.robot.get_ground_truth_pose()
    drift = float(np.abs(np.subtract([gt["translation"][k] for k in "xyz"], ned_pos)).max())
    drift_img = float(np.abs(msg_to_img(msg).astype(np.float32) - frames[POSES[-1][0]]).mean())

    spec = {n: (p, [math.radians(a) for a in r]) for n, p, r in POSES}
    res = {
        "actors": cam.world.list_actors(),
        "sensors": {k: sorted(v) for k, v in cam.robot.sensors.items()},
        "max_robot_pos_err_m": max(r["robot_pos_err_m"] for r in rows),
        "max_robot_rot_err_deg": max(r["robot_rot_err_deg"] for r in rows),
        "max_frame_pos_err_m": max(r["frame_pos_err_m"] for r in rows),
        "max_frame_rot_err_deg": max(r["frame_rot_err_deg"] for r in rows),
        "drift_after_50_idle_steps_m": drift,
        "image_mean_abs_change_after_50_idle_steps": round(drift_img, 3),
        "image_rotation": {n: image_rotation_check(frames["base"], frames[n], spec["base"][1], spec[n][1])
                           for n in ("yaw_right_30", "yaw_left_30", "nose_up_25", "nose_down_25", "roll_right_30")},
        "image_translation": {n: image_translation_check(frames["base"], frames[n], spec["base"][0], spec[n][0],
                                                         spec["base"][1])
                              for n in ("up_4m", "left_3m", "forward_5m")},
    }
    contact_sheet([(f"{n} p={p} rpy={r}", frames[n]) for n, p, r in POSES],
                  os.path.join(OUT, "pose_contact_sheet.png"), cols=4)
    print("POSE", json.dumps(res))
    return res


# ----------------------------------------------------------------------------- 2. synchrony test
def test_sync(cam, steps=200, only=None):
    """Jump between two very different views and check by pixels that frame k shows pose k."""
    views = [to_airsim(BASE, (0, 0, 0)), to_airsim((6.0, -4.0, 12.0), (0, math.radians(40), math.radians(170)))]
    refs = []
    for i, v in enumerate(views):  # reference frames: hold each view for several steps
        for _ in range(8):
            img, _ = cam.render(*v)
        refs.append(img.astype(np.float32))
        cv2.imwrite(os.path.join(OUT, f"sync_view{i}.png"), img)
    rng = np.random.default_rng(0)
    seqs = {"alternate": [k % 2 for k in range(steps)], "random": [int(v) for v in rng.integers(0, 2, steps)]}
    protocols = {
        "naive_single_step": lambda seq: [cam.render_naive(*views[v])[0] for v in seq],
        "double_step": lambda seq: [cam.render(*views[v])[0] for v in seq],
        "pipelined": lambda seq: list(cam.render_stream(views[v] for v in seq)),
        "free_running": lambda seq: list(cam.render_free(views[v] for v in seq)),
        "free_running_assuming_no_lag": lambda seq: list(cam.render_free((views[v] for v in seq), lag=0)),
    }
    out = {"ref_gap_mean_abs": round(float(np.abs(refs[0] - refs[1]).mean()), 2)}
    for pname, proto in protocols.items():
        if only and pname not in only:
            continue
        for sname, seq in seqs.items():
            cam.render(*views[1 - seq[0]])  # start from the opposite view
            n0, stray0 = cam.n_msgs, cam.stray
            frames = proto(seq)
            time.sleep(0.5)  # give late/duplicate frames a chance to show up
            dist = [[float(np.abs(f.astype(np.float32) - r).mean()) for r in refs] for f in frames]
            seen = [int(d[1] < d[0]) for d in dist]
            wrong = sum(s != v for s, v in zip(seen, seq))
            own = [d[v] for d, v in zip(dist, seq) if (d[1] < d[0]) == v]
            out[f"{pname}/{sname}"] = {
                "frames": len(frames),
                "wrong_view_frames": wrong,
                "failure_rate": wrong / steps,
                "messages_received": cam.n_msgs - n0,
                "stray_messages": cam.stray - stray0,
                # how well "frame k shows pose k-lag" explains what was seen
                **{f"match_rate_if_lag{lag}": round(sum(seen[k] == seq[k - lag] for k in range(lag, steps))
                                                  / (steps - lag), 3) for lag in (1, 2)},
                # pixel difference between a frame right after a jump and the settled reference
                "mean_abs_diff_to_own_ref": round(float(np.mean(own)), 3) if own else None,
                "max_abs_diff_to_own_ref": round(float(np.max(own)), 3) if own else None,
            }
            print("SYNC", pname, sname, json.dumps(out[f"{pname}/{sname}"]), flush=True)
            if (pname, sname) == ("pipelined", "random"):
                for k in (10, 11, 12):
                    cv2.imwrite(os.path.join(OUT, f"sync_pipelined_step{k}_view{seq[k]}.png"), frames[k])
    return out


# ----------------------------------------------------------------------------- 3. throughput
def run_steps(cam, seconds, start_at=None, seed=0, mode="pipelined"):
    """Render a smooth random flight for `seconds`; returns frames/s and time breakdown."""
    rng = np.random.default_rng(seed)
    state = {"pos": np.array([0.0, 0.0, 2.0]), "yaw": 0.0, "t_end": None}

    def flight():
        while time.perf_counter() < state["t_end"]:
            state["pos"] += rng.normal(0, 0.05, 3)
            state["pos"][2] = np.clip(state["pos"][2], 0.5, 5.0)
            state["yaw"] += rng.normal(0, 0.05)
            yield to_airsim(state["pos"], (rng.normal(0, 0.2), rng.normal(0, 0.2), state["yaw"]))

    for _ in range(20):  # warm-up
        cam.render_flu(state["pos"], (0, 0, 0))
    if start_at:
        time.sleep(max(0.0, start_at - time.time()))
    cam.timing[:] = 0
    n0, stray0, cpu0, t0 = cam.n_msgs, cam.stray, time.process_time(), time.perf_counter()
    state["t_end"] = t0 + seconds
    if mode == "pipelined":
        n = sum(1 for _ in cam.render_stream(flight()))
    elif mode == "free":
        n = sum(1 for _ in cam.render_free(flight()))
    else:
        n = sum(1 for p in flight() if cam.render(*p) is not None)
    wall = time.perf_counter() - t0
    return {
        "fps": round(n / wall, 1),
        "frames": n,
        "messages_received": cam.n_msgs - n0,
        "stray_messages": cam.stray - stray0,
        "client_cpu_cores": round((time.process_time() - cpu0) / wall, 2),
        "ms_set_pose": round(1e3 * cam.timing[0] / n, 2),
        "ms_clock_step": round(1e3 * cam.timing[1] / n, 2),
        "ms_wait_frame": round(1e3 * cam.timing[2] / n, 2),
    }


def measure(gpu, n_inst, width, height, seconds, sim_args, mode="pipelined"):
    """n_inst sims on one GPU, one client process each, all stepping at the same time."""
    procs = [launch_sim(gpu, i, sim_args) for i in range(n_inst)]
    try:
        for i, p in enumerate(procs):
            wait_port(ports(i)[1], p)
        start_at = time.time() + 12 + 2 * n_inst
        workers = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", "--idx", str(i),
                                     "--res", f"{width}x{height}", "--seconds", str(seconds),
                                     "--start-at", str(start_at), "--mode", mode, "--capture", CAPTURE_OVERRIDE],
                                    stdout=subprocess.PIPE, text=True)
                   for i in range(n_inst)]
        time.sleep(max(0.0, start_at - time.time()) + 1)
        sampler = GpuSampler(gpu)
        sampler.start()
        cpu0, t0 = [cpu_seconds(p.pid) for p in procs], time.time()
        time.sleep(seconds - 2)
        sim_cpu = [(cpu_seconds(p.pid) - c) / (time.time() - t0) for p, c in zip(procs, cpu0)]
        gpu_stats = sampler.summary()
        placement = gpu_of_pids({p.pid for p in procs})
        results = []
        for wk in workers:
            line = [ln for ln in wk.communicate()[0].splitlines() if ln.startswith("WORKER ")]
            results.append(json.loads(line[-1][7:]) if line else {"fps": 0.0, "error": "worker failed"})
        fps = [r["fps"] for r in results]
        row = {
            "instances": n_inst, "res": f"{width}x{height}", "mode": mode, "sim_args": " ".join(sim_args),
            "fps_total": round(sum(fps), 1), "fps_per_instance": round(float(np.mean(fps)), 1),
            "fps_min": min(fps), "fps_max": max(fps),
            **gpu_stats,
            "sim_cpu_cores_per_instance": round(float(np.mean(sim_cpu)), 2),
            "client_cpu_cores_per_instance": round(float(np.mean([r.get("client_cpu_cores", 0) for r in results])), 2),
            "ms_set_pose": results[0].get("ms_set_pose"), "ms_clock_step": results[0].get("ms_clock_step"),
            "ms_wait_frame": results[0].get("ms_wait_frame"),
            "sim_gpus": sorted({v["gpu"] for v in placement.values()}), "cvars": RENDER_CVARS,
            "frames_ok": all(r.get("stray_messages") == 0 and "error" not in r for r in results),
        }
        print("THROUGHPUT", json.dumps(row), flush=True)
        return row
    finally:
        kill_sims(procs)


# ----------------------------------------------------------------------------- 4. gates
def box(cx, cy, cz, sx, sy, sz):
    """24 vertices + normals + 36 indices of an axis-aligned box."""
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


def gate_glb(opening=GATE_OPENING, bar=GATE_BAR, depth=GATE_DEPTH, rgb=(1.0, 0.0, 0.8)):
    """Binary glTF of a square gate frame. glTF axes: frame lies in the X-Y plane, opening normal = Z."""
    c = opening / 2 + bar / 2
    outer = opening + 2 * bar
    V, N, I = [], [], []
    for cx, cy, sx, sy in ((0, c, outer, bar), (0, -c, outer, bar), (c, 0, bar, opening), (-c, 0, bar, opening)):
        v, n, i = box(cx, cy, 0, sx, sy, depth)
        I += [len(V) + k for k in i]
        V += v
        N += n
    V, N, I = np.array(V, np.float32), np.array(N, np.float32), np.array(I, np.uint16)
    blob = V.tobytes() + N.tobytes() + I.tobytes()
    blob += b"\0" * (-len(blob) % 4)
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [*rgb, 1.0], "metallicFactor": 0.0,
                                                "roughnessFactor": 0.8},
                       "emissiveFactor": [0.4 * c for c in rgb], "doubleSided": True}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": V.nbytes, "target": 34962},
                        {"buffer": 0, "byteOffset": V.nbytes, "byteLength": N.nbytes, "target": 34962},
                        {"buffer": 0, "byteOffset": V.nbytes + N.nbytes, "byteLength": I.nbytes, "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(V), "type": "VEC3",
                       "min": V.min(0).tolist(), "max": V.max(0).tolist()},
                      {"bufferView": 1, "componentType": 5126, "count": len(N), "type": "VEC3"},
                      {"bufferView": 2, "componentType": 5123, "count": len(I), "type": "SCALAR"}],
    }
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(blob)) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(blob), b"BIN\0") + blob)


def ned_pose(pos, yaw=0.0):
    ned_pos, q = to_airsim(pos, (0.0, 0.0, yaw))
    return Pose({
        "translation": Vector3({"x": float(ned_pos[0]), "y": float(ned_pos[1]), "z": float(ned_pos[2])}),
        "rotation": Quaternion({"w": q[0], "x": q[1], "y": q[2], "z": q[3]}),
        "frame_id": "DEFAULT_ID",
    })


def spawn_gates(cam, track=REF_TRACK):
    """Spawn one gate mesh per track entry at runtime; returns the spawned object names."""
    glb = gate_glb()
    return [cam.world.spawn_object_from_file(f"Gate{i}", "gltf", glb, True,
                                             ned_pose(g["pos"], g["yaw"] + GATE_MESH_YAW), [1.0, 1.0, 1.0], False)
            for i, g in enumerate(track)]


# Extra yaw that turns the imported mesh so that its opening normal points along the gate's travel direction.
GATE_MESH_YAW = math.pi / 2  # measured: the glTF +Z normal comes out along the project +y axis at yaw 0


def gate_axes(g):
    n = np.array([math.cos(g["yaw"]), math.sin(g["yaw"]), 0.0])  # travel direction
    return n, np.array([-n[1], n[0], 0.0]), np.array([0.0, 0.0, 1.0])  # normal, left, up


def gate_square(g, half, along=0.0):
    """Corners (up-left, up-right, down-right, down-left as seen when flying through) of a square in the gate."""
    n, l, u = gate_axes(g)
    c = np.asarray(g["pos"], float) + along * n
    return np.array([c + half * (a * l + b * u) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def gate_bars(g):
    """The four bar boxes of a gate as [8,3] vertex arrays."""
    n, l, u = gate_axes(g)
    c, outer = GATE_OPENING / 2 + GATE_BAR / 2, GATE_OPENING + 2 * GATE_BAR
    bars = []
    for cl, cu, sl, su in ((0, c, outer, GATE_BAR), (0, -c, outer, GATE_BAR),
                           (c, 0, GATE_BAR, GATE_OPENING), (-c, 0, GATE_BAR, GATE_OPENING)):
        bars.append(np.array([np.asarray(g["pos"], float) + (cl + a * sl / 2) * l + (cu + b * su / 2) * u
                              + d * GATE_DEPTH / 2 * n for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]))
    return bars


def expected_mask(gates, pos, rpy, width, height):
    """Silhouette of the gate frames predicted by the pinhole model."""
    m = np.zeros((height, width), np.uint8)
    for g in gates:
        for bar in gate_bars(g):
            uv, z = project(bar, pos, rpy, width, height)
            if z.min() > 0.05:
                cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return m.astype(bool)


def gate_mask(img):
    """Pixels with the gate colour (magenta): red and blue high, green low."""
    b, g, r = (img[..., i].astype(np.int32) for i in range(3))
    return (r > 120) & (b > 90) & (g < 0.75 * np.minimum(r, b))


def rect_corners(m):
    """Corner coordinates (ul, ur, dr, dl) of the bounding rectangle of a boolean mask, at pixel edges."""
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min() - 0.5, xs.max() + 0.5, ys.min() - 0.5, ys.max() + 0.5
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def survey(cam, centre=(0.0, 0.0), alt=40.0, half=16, cell=2.0):
    """Top-down planar-depth shot -> height (project z, m) of the highest surface per `cell` m square."""
    n = 512
    pos, rpy = (centre[0], centre[1], alt), (0.0, math.radians(90 + CAM_TILT_DEG), 0.0)
    depth = np.asarray(cam.render_flu(pos, rpy)[0], np.float64).reshape(n, n)
    u, v = np.meshgrid(np.arange(n), np.arange(n))
    rays = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3) @ np.linalg.inv(intrinsics(n, n)).T
    pw = np.asarray(pos) + (rays * depth.reshape(-1, 1)) @ cam_rotation(rpy).T
    pw = pw[np.isfinite(pw).all(1)]
    ix = np.floor((pw[:, 0] - centre[0]) / cell).astype(int) + half
    iy = np.floor((pw[:, 1] - centre[1]) / cell).astype(int) + half
    ok = (ix >= 0) & (ix < 2 * half) & (iy >= 0) & (iy < 2 * half)
    top = np.full((2 * half, 2 * half), -np.inf)
    np.maximum.at(top, (ix[ok], iy[ok]), pw[ok, 2])
    print(f"SURVEY depth range {np.nanmin(depth):.2f}..{np.nanmax(depth[np.isfinite(depth)]):.2f} m; "
          f"top-surface height per {cell} m cell, rows = x from {centre[0] - half * cell} (step {cell}), "
          f"columns = y from {centre[1] - half * cell}")
    for r in top:
        print(" ".join(f"{h:5.1f}" for h in r))
    return top


def test_gates(cam, width, height):
    names = spawn_gates(cam)
    res = {"spawned": names, "gates": [], "available_assets": len(cam.world.list_assets(".*"))}
    for n, g in zip(names, REF_TRACK):
        p = cam.world.get_object_pose(n)
        want = ned_pose(g["pos"], g["yaw"] + GATE_MESH_YAW)
        res["gates"].append({
            "name": n,
            "sim_pos_err_m": float(max(abs(p["translation"][k] - want["translation"][k]) for k in "xyz")),
            "sim_rot_err_deg": quat_angle_deg([p["rotation"][k] for k in "wxyz"],
                                              [want["rotation"][k] for k in "wxyz"]),
        })
    tiles = []
    half_in, half_out = GATE_OPENING / 2, GATE_OPENING / 2 + GATE_BAR
    for i, g in enumerate(REF_TRACK):
        # camera 1.5 m before the gate on its axis; body pitched nose-down by the camera tilt so that
        # the optical axis coincides with the gate axis
        n, _, _ = gate_axes(g)
        pos, rpy = np.asarray(g["pos"], float) - 1.5 * n, (0.0, math.radians(CAM_TILT_DEG), g["yaw"])
        img, _ = cam.render_flu(pos, rpy)
        cv2.imwrite(os.path.join(OUT, f"gates_through_gate{i + 1}.png"), img)
        mask, exp = gate_mask(img), expected_mask(REF_TRACK, pos, rpy, width, height)
        own = expected_mask([g], pos, rpy, width, height)
        roi = cv2.dilate(own.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
        roi &= ~cv2.dilate((exp & ~own).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        m = mask & roi
        row = res["gates"][i]
        row["mask_px"], row["expected_px"] = int(m.sum()), int(own.sum())
        row["iou_all_gates"] = round(float((mask & exp).sum() / max(1, (mask | exp).sum())), 4)
        row["mean_bgr_in_gate"] = img[own].mean(0).round(0).tolist()
        # the camera is on the gate axis: the outline seen is the near face's outer square and the
        # opening seen is the far face's inner square
        want_out, _ = project(gate_square(g, half_out, -GATE_DEPTH / 2), pos, rpy, width, height)
        want_in, _ = project(gate_square(g, half_in, GATE_DEPTH / 2), pos, rpy, width, height)
        mid, _ = project(gate_square(g, half_in), pos, rpy, width, height)
        row["opening_corners_px"] = mid.round(2).tolist()
        if m.sum() > 50:
            got_out = rect_corners(m)
            x0, y0, x1, y1 = (int(round(v)) for v in (*(got_out[0] + 0.5), *(got_out[2] + 0.5)))
            hole = np.zeros_like(m)
            hole[y0:y1, x0:x1] = ~mask[y0:y1, x0:x1]  # full mask: a far gate seen through the opening is not hole
            _, lab = cv2.connectedComponents(hole.astype(np.uint8))
            centre = np.round(mid.mean(0)).astype(int)
            got_in = rect_corners(lab == lab[centre[1], centre[0]]) if hole[centre[1], centre[0]] else None
            row["outer_corner_err_px"] = np.linalg.norm(got_out - want_out, axis=1).round(2).tolist()
            if got_in is not None:
                row["opening_corner_err_px"] = np.linalg.norm(got_in - want_in, axis=1).round(2).tolist()
                row["opening_centre_err_px"] = (got_in.mean(0) - want_in.mean(0)).round(2).tolist()
        print("GATE", json.dumps(row), flush=True)
        vis = img.copy()
        for u, v in mid:
            cv2.drawMarker(vis, (int(round(u)), int(round(v))), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
        tiles.append((f"G{i + 1} from 1.5 m (+ = projected opening corners)", vis))

    centre = np.mean([g["pos"] for g in REF_TRACK], axis=0)
    views = {"overview": (-3.5, -3.5, 5.0), "overview2": (10.0, 8.5, 4.0), "topdown": (centre[0], centre[1], 7.0)}
    for name, pos in views.items():
        rpy = (0.0, math.radians(90 + CAM_TILT_DEG), math.pi / 2) if name == "topdown" else look_at(pos, centre)
        img, _ = cam.render_flu(pos, rpy)
        cv2.imwrite(os.path.join(OUT, f"gates_{name}.png"), img)
        mask, exp = gate_mask(img), expected_mask(REF_TRACK, pos, rpy, width, height)
        errs = []
        for g in REF_TRACK:  # centroid of the gate-coloured pixels near each predicted gate vs prediction
            own = expected_mask([g], pos, rpy, width, height)
            near = mask & cv2.dilate(own.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            if near.any() and own.any():
                errs.append(float(np.linalg.norm(np.argwhere(near).mean(0) - np.argwhere(own).mean(0))))
        res[name] = {"iou": round(float((mask & exp).sum() / max(1, (mask | exp).sum())), 4),
                     "mask_px": int(mask.sum()), "expected_px": int(exp.sum()),
                     "gates_found": len(errs), "gate_centroid_err_px": np.round(errs, 2).tolist()}
        vis = img.copy()
        for g in REF_TRACK:
            uv, _ = project(gate_square(g, half_in), pos, rpy, width, height)
            cv2.polylines(vis, [np.round(uv).astype(np.int32)], True, (0, 255, 0), 1)
        tiles.append((f"{name} (green = projected openings)", vis))
    contact_sheet(tiles, os.path.join(OUT, "gates_contact_sheet.png"), cols=3, size=(480, 360))
    ok = [r for r in res["gates"] if "opening_corner_err_px" in r]
    res["max_opening_corner_err_px"] = max((max(r["opening_corner_err_px"]) for r in ok), default=None)
    res["max_outer_corner_err_px"] = max((max(r["outer_corner_err_px"]) for r in ok), default=None)
    res["gates_verified"] = len(ok)
    print("GATES", json.dumps({k: v for k, v in res.items() if k != "gates"}))
    return res


# ----------------------------------------------------------------------------- main
def parse_res(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main():
    global CAPTURE_OVERRIDE, RENDER_CVARS
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "pose", "sync", "throughput", "multi", "gates", "worker", "survey"])
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--res", default="160x120")
    ap.add_argument("--instances", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--start-at", type=float, default=0)
    ap.add_argument("--sim-args", default="", help="extra Unreal command-line switches, space separated")
    ap.add_argument("--no-launch", action="store_true", help="use an already running sim on --idx ports")
    ap.add_argument("--mode", choices=["pipelined", "double", "free"], default="free")
    ap.add_argument("--capture", default="{}", help="JSON merged into the camera capture-settings")
    ap.add_argument("--cvars", default=RENDER_CVARS, help="Unreal console variables, name=value,name=value")
    ap.add_argument("--tag", default="", help="label stored with throughput rows in out/airsim_results.json")
    ap.add_argument("--steps", type=int, default=200, help="sync test: frames per sequence")
    ap.add_argument("--protocols", nargs="*", help="sync test: only these protocols (default all)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sim_args = shlex.split(a.sim_args)
    CAPTURE_OVERRIDE, RENDER_CVARS = a.capture, a.cvars
    summary = {}

    if a.cmd == "worker":
        cam = FpvCam(a.idx, *parse_res(a.res), capture=json.loads(a.capture))
        print("WORKER " + json.dumps(run_steps(cam, a.seconds, a.start_at, a.idx, a.mode)), flush=True)
        cam.close()
        return

    if a.cmd in ("all", "pose", "sync", "gates", "survey"):
        procs = [] if a.no_launch else [launch_sim(a.gpu, a.idx, sim_args)]
        try:
            if procs:
                wait_port(ports(a.idx)[1], procs[0])
                print("sim placement:", gpu_of_pids({procs[0].pid}))
            if a.cmd in ("all", "pose"):
                cam = FpvCam(a.idx, 640, 360)  # 16:9, the aspect of the project camera (fpv/track.py)
                summary["pose"] = test_pose(cam)
                cam.close()
            if a.cmd in ("all", "sync"):
                cam = FpvCam(a.idx, 160, 120)
                key = "sync" if a.steps == 200 and not a.protocols else f"sync_{a.steps}_steps"
                summary[key] = test_sync(cam, a.steps, a.protocols)
                cam.close()
            if a.cmd in ("all", "gates"):
                cam = FpvCam(a.idx, 640, 360)
                summary["gates"] = test_gates(cam, 640, 360)
                cam.close()
            if a.cmd == "survey":
                cam = FpvCam(a.idx, 512, 512, depth=True)
                survey(cam)
                cam.close()
        finally:
            kill_sims(procs)

    if a.cmd in ("all", "throughput"):
        summary["throughput"] = [measure(a.gpu, 1, w, h, a.seconds, sim_args, a.mode)
                                 for w, h in ((64, 64), (160, 120), (320, 240))]
    if a.cmd == "multi":
        summary["multi"] = [measure(a.gpu, n, *parse_res(a.res), a.seconds, sim_args, a.mode) for n in a.instances]

    if summary:
        path, merged = os.path.join(OUT, "airsim_results.json"), {}
        if os.path.exists(path):
            merged = json.load(open(path))
        key = "+".join(f for f in (a.mode, a.sim_args, a.cvars, a.tag) if f)
        for k, v in summary.items():
            merged[k if k in ("pose", "sync", "gates") else f"{k} [{key}]"] = v
        json.dump(merged, open(path, "w"), indent=1)
    print("SUMMARY " + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
