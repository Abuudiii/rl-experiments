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
GATE_MESH_YAW = math.pi / 2  # the imported glTF frame (normal +Z) comes out with its normal along +y
STEP_NS = 10_000_000
CHUNK = 32  # frames per worker command in render_trajectories
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


def gate_glb() -> bytes:
    """Binary glTF of one gate: four bars around a square opening, in the X-Y plane, normal +Z."""
    opening, bar = track.GATE_OPENING, (track.GATE_OUTER - track.GATE_OPENING) / 2
    c, outer = opening / 2 + bar / 2, track.GATE_OUTER
    V, N, I = [], [], []
    for cx, cy, sx, sy in ((0, c, outer, bar), (0, -c, outer, bar), (c, 0, bar, opening), (-c, 0, bar, opening)):
        v, n, i = _box(cx, cy, 0, sx, sy, GATE_DEPTH)
        I += [len(V) + k for k in i]
        V += v
        N += n
    V, N, I = np.array(V, np.float32), np.array(N, np.float32), np.array(I, np.uint16)
    blob = V.tobytes() + N.tobytes() + I.tobytes()
    blob += b"\0" * (-len(blob) % 4)
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [*GATE_RGB, 1.0], "metallicFactor": 0.0,
                                                "roughnessFactor": 0.8},
                       "emissiveFactor": [0.4 * c for c in GATE_RGB], "doubleSided": True}],
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


def _sim_configs(width, height):
    robot = {
        "physics-type": "non-physics",  # AirSim never integrates dynamics; the robot only moves on SetPose
        "links": [{"name": "Frame", "collision": {"enabled": False}}],
        "sensors": [{
            "id": "Fpv", "type": "camera", "enabled": True, "parent-link": "Frame",
            "capture-interval": 0.001,  # below the clock step, so every step captures
            "capture-settings": [{
                "image-type": 0, "width": width, "height": height, "fov-degrees": track.CAMERA["hfov_deg"],
                "capture-enabled": True, "streaming-enabled": False, "pixels-as-float": False,
                "compress": False, "target-gamma": 2.5, "motion-blur-amount": 0.0,
            }],
            # NED body frame: +pitch = nose up
            "origin": {"xyz": "0.0 0.0 0.0", "rpy-deg": f"0 {track.CAMERA['pitch_up_deg']} 0"},
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
        self.tilt = (math.cos(math.radians(track.CAMERA["pitch_up_deg"]) / 2), 0.0,
                     math.sin(math.radians(track.CAMERA["pitch_up_deg"]) / 2), 0.0)  # NED: +pitch = nose up

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
        scene, robot = _sim_configs(a.width, a.height)
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
        self.gate_names = [
            self.world.spawn_object_from_file(f"Gate{i}", "gltf", glb, True, self._gate_pose(p, y), [1.0, 1.0, 1.0], False)
            for i, (p, y) in enumerate(zip(gates["pos"], gates["yaw"]))]
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

    def render(self, pos, quat):
        ned = self.set_pose(pos, quat)
        self.step()  # this step renders the pose; its message still holds the previous render
        msg = self.step()
        self._check(msg, *ned)
        self.frames[0] = self._rgb(msg)

    def render_stream(self, poses, quats):
        ned = None
        for k, (pos, quat) in enumerate(zip(poses, quats)):
            ned = self.set_pose(pos, quat)
            msg = self.step()
            self._check(msg, *ned)
            if k:
                self.frames[k - 1] = self._rgb(msg)
        msg = self.step()  # flush: same pose, fetch the last render
        self._check(msg, *ned)
        self.frames[len(poses) - 1] = self._rgb(msg)

    def set_gates(self, pos, yaw):
        for name, p, y in zip(self.gate_names, pos, yaw):
            self.world.set_object_pose(name, self._gate_pose(p, y), True)

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
                if c["op"] == "render":
                    self.render(c["pos"], c["quat"])
                elif c["op"] == "traj":
                    self.render_stream(c["pos"], c["quat"])
                elif c["op"] == "gates":
                    self.set_gates(c["pos"], c["yaw"])
                elif c["op"] == "quit":
                    break
                self.out.write(json.dumps({"ok": 1, "ms": round(1e3 * (time.perf_counter() - t0), 3)}) + "\n")
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
    """

    def __init__(self, gpus, instances_per_gpu=8, width=320, height=180, look="stock", *, gates_pos=None,
                 gates_yaw=None, root=AIRSIM_ROOT, startup_timeout=300.0, step_timeout=30.0, max_restarts=2):
        if look not in LOOKS:
            raise ValueError(f"look must be one of {sorted(LOOKS)}, got {look!r}")
        self.gpus, self.n = [int(g) for g in gpus], len(gpus) * int(instances_per_gpu)
        self.width, self.height, self.look = int(width), int(height), look
        self.root = str(root)
        self.startup_timeout, self.step_timeout, self.max_restarts = startup_timeout, step_timeout, max_restarts
        self.restarts = 0  # number of watchdog restarts so far
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
        log = open(os.path.join(home, "worker.log"), "wb")
        self._procs[s] = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                          pass_fds=(c_r, r_w), start_new_session=True)
        os.close(c_r)
        os.close(r_w)
        self._fds[s], self._bufs[s] = (c_w, r_r), b""
        self._sim_pids.pop(s, None)
        self._send(s, {"gates": {"pos": self._gates_pos[s].tolist(), "yaw": self._gates_yaw[s].tolist()}})

    def _await_ready(self, slots):
        for s, rep in self._collect(list(slots), self.startup_timeout + 60).items():
            if rep is None or "error" in rep:
                raise RuntimeError(f"slot {s} failed to start: {rep and rep['error']}\n{self._log_tail(s)}")
            self._sim_pids[s] = rep["sim_pid"]
            want = self.gpus[s % len(self.gpus)]
            if rep["gpu"] is not None and rep["gpu"] != want:
                raise RuntimeError(f"slot {s} renders on GPU {rep['gpu']} instead of GPU {want}")

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

    def render(self, pos, quat_xyzw, active=None):
        """One frame per slot from the body poses of THIS call.

        pos [n,3], quat_xyzw [n,4] (body->world, contract frame), active bool [n] or None (= all).
        Returns uint8 [n,H,W,3] RGB; inactive slots are zeros and are not rendered.
        """
        pos = np.asarray(pos, np.float64).reshape(self.n, 3)
        quat = np.asarray(quat_xyzw, np.float64).reshape(self.n, 4)
        act = np.ones(self.n, bool) if active is None else np.asarray(active, bool).reshape(self.n)
        out = np.zeros((self.n, self.height, self.width, 3), np.uint8)
        idx = np.flatnonzero(act)
        if len(idx):
            self._call({int(s): {"op": "render", "pos": pos[s].tolist(), "quat": quat[s].tolist()} for s in idx},
                       self.step_timeout)
            out[idx] = self._frames[idx, 0]
        return out

    def render_trajectories(self, slots, pos, quat_xyzw):
        """Open-loop rendering of whole trajectories, one capture per frame.

        slots int [k] (distinct), pos [k,T,3], quat_xyzw [k,T,4] -> uint8 [k,T,H,W,3] RGB.
        The result is one array (k * T * H * W * 3 bytes); split T on the caller's side if that is
        too large. Frame t of slot j is rendered from pose t of slot j with that slot's current gates.
        """
        slots = np.atleast_1d(np.asarray(slots, np.int64))
        if len(set(slots.tolist())) != len(slots):
            raise ValueError("slots must be distinct")
        pos = np.asarray(pos, np.float64).reshape(len(slots), -1, 3)
        quat = np.asarray(quat_xyzw, np.float64).reshape(len(slots), -1, 4)
        T = pos.shape[1]
        out = np.empty((len(slots), T, self.height, self.width, 3), np.uint8)
        for t0 in range(0, T, CHUNK):
            t1 = min(T, t0 + CHUNK)
            self._call({int(s): {"op": "traj", "pos": pos[i, t0:t1].tolist(), "quat": quat[i, t0:t1].tolist()}
                        for i, s in enumerate(slots)}, self.step_timeout * 4)
            out[:, t0:t1] = self._frames[slots, : t1 - t0]
        return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2:])
