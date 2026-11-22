"""Re-film a recorded Donkey Car drive in Unreal (Project AirSim "Blocks", offscreen). Visuals only.

  # on the GPU node, with the AirSim client venv made by fpv-racing/scripts/setup_airsim.sh:
  /tmp/fpv_airsim/venv/bin/python scripts/unreal_render.py --record runs/video_p3 --out /tmp/donkey_unreal
  /tmp/fpv_airsim/venv/bin/python scripts/unreal_render.py --record runs/video_p3 --out /tmp/donkey_unreal --probe

Nothing is simulated here: the car is put at each recorded pose (steps.jsonl) on a rebuilt copy of the
Donkey "generated_track" road (track.json centre-line nodes), and three cameras film it, one JPEG per view
and step: <out>/<view>/<frame>.jpg with view in pilot, chase, hawk (plus overhead_<frame>.jpg for the
--verify steps: a camera fixed to the car looking straight down, like the Donkey overhead_*.jpg frames).

Frames. Donkey/Unity world: left-handed, x right, y up, z forward; telemetry positions are world / 8;
yaw = eulerAngles.y in degrees (turns +z towards +x). Contract frame of fpv.render_airsim (right-handed,
x, y horizontal, z up, metres): (x, y, z) = SCALE * (ux, uz, uy) (minus the track centre, plus ORIGIN).
Seen from above this is the same picture (Unity x right / z up the page = contract x right / y up the
page), so nothing is mirrored; a Unity heading yaw becomes the contract heading psi = 90 deg - yaw.

Road (sdsandbox sdsim/Assets/Scripts/RoadBuilder.cs with the generated_track overrides of
Prefabs/Scripts/_controllers.prefab: roadWidth 5, roadOffsetW 2): the road surface spans path node
+/- 5 units, shifted 2 units to the LEFT of the path (vWidth = Cross(segment, up) points left in Unity).
The texture's yellow line is in the middle of that surface (2 units left of the path the car follows
with cte = 0) and its white edge lines are 4.05 units from the middle, with gravel out to 5 units
(measured on the Donkey overhead frames, see the report).

Reused from fpv-racing (imported, not copied): the glTF builders (_box, _cylinder, _glb, _material,
_merge, GLTF_FROM_BODY), view geometry (look_at_quat, mat_to_quat, _smooth), the sim configs
(_sim_configs), and the lock-step sim worker (_Worker: launch_sim with a private HOME/Engine.ini and
-graphicsadapter, step(), _on_image, _pose, sim_gpu, to_ned).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

FPV_ROOT = os.environ.get('FPV_ROOT', '/home/AMD/absheikh/fpv-racing')
sys.path.insert(0, FPV_ROOT)
from fpv import render_airsim as ra  # noqa: E402

SCALE = 1.0                  # Unreal metres per Donkey world unit
TELEMETRY = 8.0              # Donkey world units per telemetry unit
# Contract-frame position of the track centre. --probe: the Blocks level geometry (blocks, balls,
# cylinders) lies within AirSim NED x in [-40, 120] m, y in [-122, 117] m on a 40 km ground plane; the
# track with its grass (87 x 130 m) is centred at NED (-150, 3), 66 m clear of the nearest block.
ORIGIN = (-130.0, 0.0, 0.0)
ROAD_SHIFT = 2.0             # road middle is this far LEFT of the path nodes [world units]
ROAD_HALF = 5.0              # paved half-width incl. gravel shoulder
LINE_OFF = 4.05              # white edge lines, from the road middle
LINE_W = 0.25                # paint width (white and yellow)
CAR_LEN, CAR_WID = 2.2, 1.75  # measured on the Donkey overhead frames (tyre to tyre)
# Pilot camera, car frame: forward, left, up [units], pitch down [deg]. Measured on Donkey pilot frames
# of the start straight (cte 0): the lane lines meet 33.4 px above the centre of the 320x240 image and
# their spacing (yellow to right white = 4.04 units) gives a camera height of 1.75 units; with the
# prefab's 13.3 deg pitch that puts the focal length at 141 px (vertical fov 81 deg, horizontal 97 deg
# at 4:3), which also agrees with the overhead frames' 9.4 px/unit seen from 12 units above the camera.
# Forward offset: the start strip sits 0.3 (overhead frames) to 0.7 (pilot frames) units closer to the
# Donkey camera than to the car's telemetry position; 0.5 is used.
PILOT_CAM = (0.5, 0.0, 1.75, 13.3)
HFOV = {'pilot': 97.0, 'chase': 75.2, 'hawk': 50.0}
VIEWS = ('pilot', 'chase', 'hawk')
WARMUP = 20                  # renders of the first pose of a chunk (exposure / temporal AA settle)


# ----------------------------------------------------------------------------- recorded drive
def load(record):
  rows = [json.loads(line) for line in (record / 'steps.jsonl').read_text().splitlines()]
  end = next(i for i, row in enumerate(rows) if row['last'])
  rows = rows[:end + 1]
  nodes = np.array(json.loads((record / 'track.json').read_text()), np.float64)  # world units, [x, z]
  return rows, nodes


class Frame:
  """Donkey world (units) <-> contract frame (metres)."""

  def __init__(self, nodes):
    self.centre = (nodes.min(0) + nodes.max(0)) / 2

  def xy(self, ux, uz):
    """World x, z (units) -> contract x, y (metres) [..., 2]."""
    return np.stack([SCALE * (np.asarray(ux) - self.centre[0]) + ORIGIN[0],
                     SCALE * (np.asarray(uz) - self.centre[1]) + ORIGIN[1]], -1)


def yaw_quat(psi):
  psi = np.asarray(psi, np.float64)
  return np.stack([np.zeros_like(psi), np.zeros_like(psi), np.sin(psi / 2), np.cos(psi / 2)], -1)


def car_poses(rows, frame):
  """Car pose per step from the recorded x, y, z, yaw. Height: the recorded y relative to its median
  over the drive (the car resting on the road; track.json has no node heights to give the road's own
  level), so the model sits on the Unreal road and keeps every recorded vertical movement."""
  pos = frame.xy([r['x'] * TELEMETRY for r in rows], [r['z'] * TELEMETRY for r in rows])
  y = np.array([r['y'] for r in rows]) * TELEMETRY
  height = SCALE * (y - np.median(y))
  psi = np.radians(90.0 - np.array([r['yaw'] for r in rows]))
  psi = np.unwrap(psi)
  return np.concatenate([pos, height[:, None]], 1), psi


# ----------------------------------------------------------------------------- road geometry
def centre_line(nodes, step=0.2, sigma=1.0):
  """Closed centripetal Catmull-Rom through the nodes, resampled every `step` units, then smoothed
  along the loop (Gaussian, `sigma` units). -> points [n, 2] (world units), tangents, left normals."""
  P = np.asarray(nodes, np.float64)
  n = len(P)
  dense = []
  for i in range(n):
    p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % n], P[(i + 2) % n]
    t0 = 0.0
    t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
    t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
    t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
    k = max(2, int(np.ceil(np.linalg.norm(p2 - p1) / 0.05)))
    for t in np.linspace(t1, t2, k, endpoint=False):
      a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
      a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
      a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
      b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
      b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
      dense.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
  dense = np.array(dense)
  seg = np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1)
  s = np.concatenate([[0], np.cumsum(seg)])
  m = int(round(s[-1] / step))
  q = np.linspace(0, s[-1], m, endpoint=False)
  closed = np.vstack([dense, dense[:1]])
  pts = np.stack([np.interp(q, s, closed[:, 0]), np.interp(q, s, closed[:, 1])], -1)
  r = int(4 * sigma / step)
  kern = np.exp(-0.5 * (np.arange(-r, r + 1) * step / sigma) ** 2)
  kern /= kern.sum()
  wrap = np.vstack([pts[-r:], pts, pts[:r]])
  pts = np.stack([np.convolve(wrap[:, c], kern, mode='valid') for c in range(2)], -1)
  tan = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
  tan /= np.linalg.norm(tan, axis=1, keepdims=True)
  left = np.stack([-tan[:, 1], tan[:, 0]], -1)  # contract/top view: left of the heading
  return pts, tan, left


def _strip(pts, left, a, b, z):
  """Ribbon between lateral offsets a < b (world units, + = left) at height z (m), closed loop."""
  n = len(pts)
  inner, outer = pts + a * left, pts + b * left
  V, I = [], []
  for i in range(n):
    V += [[*inner[i], z], [*outer[i], z]]
  for i in range(n):
    j = (i + 1) % n
    I += [2 * i, 2 * j, 2 * i + 1, 2 * i + 1, 2 * j, 2 * j + 1]
  N = [[0.0, 0.0, 1.0]] * len(V)
  return V, N, I


def _cone(cx, cy, z0, r, h, seg=10):
  v, n, idx = [], [], []
  ang = 2 * np.pi * np.arange(seg) / seg
  slope = r / h
  for k in range(seg):
    a0, a1 = ang[k], ang[(k + 1) % seg]
    am = (a0 + a1) / 2
    nn = np.array([math.cos(am), math.sin(am), slope])
    nn = (nn / np.linalg.norm(nn)).tolist()
    base = len(v)
    v += [[cx + r * math.cos(a0), cy + r * math.sin(a0), z0], [cx + r * math.cos(a1), cy + r * math.sin(a1), z0], [cx, cy, z0 + h]]
    n += [nn, nn, nn]
    idx += [base, base + 1, base + 2]
  return v, n, idx


def track_parts(nodes, frame, seed=0):
  """Road, paint, start strip, grass and trees in the contract frame, relative to ORIGIN.
  -> [(V [n,3] metres, N, I, rgb, roughness)]"""
  pts_w, tan, left = centre_line(nodes)
  road = pts_w + ROAD_SHIFT * left  # road middle
  pts = frame.xy(road[:, 0], road[:, 1]) - np.array(ORIGIN[:2])
  S = SCALE
  parts = []

  def add(geo, rgb, rough=0.9):
    V, N, I = geo if len(geo) == 3 else ra._merge(geo)
    parts.append((np.asarray(V, np.float64), np.asarray(N, np.float64), np.asarray(I, np.int64), rgb, rough))

  z_road, z_paint = 0.03, 0.045
  add(_strip(pts, left, -S * LINE_OFF, S * LINE_OFF, z_road), (0.30, 0.28, 0.24))
  add(ra._merge([_strip(pts, left, -S * ROAD_HALF, -S * LINE_OFF, z_road),
                 _strip(pts, left, S * LINE_OFF, S * ROAD_HALF, z_road)]), (0.42, 0.34, 0.22))
  add(ra._merge([_strip(pts, left, -S * (LINE_OFF + LINE_W / 2), -S * (LINE_OFF - LINE_W / 2), z_paint),
                 _strip(pts, left, S * (LINE_OFF - LINE_W / 2), S * (LINE_OFF + LINE_W / 2), z_paint)]), (0.95, 0.95, 0.93), 0.6)
  add(_strip(pts, left, -S * LINE_W / 2, S * LINE_W / 2, z_paint), (0.95, 0.75, 0.05), 0.6)

  # start/finish checker strip across the paved width at node 0, 0.9 units deep, 2 x 12 squares
  p0 = frame.xy(nodes[0, 0], nodes[0, 1]) - np.array(ORIGIN[:2])
  d = nodes[1] - nodes[0]
  d = d / np.linalg.norm(d)
  lf = np.array([-d[1], d[0]])
  c0 = p0 + S * ROAD_SHIFT * lf
  cols, depth = 12, 0.9
  sq_w = 2 * ROAD_HALF / cols
  black, white = [], []
  for r_ in range(2):
    for c in range(cols):
      fwd = (r_ - 0.5) * depth / 2
      lat = -ROAD_HALF + (c + 0.5) * sq_w
      ctr = c0 + S * (fwd * d + lat * lf)
      ang = math.atan2(d[1], d[0])
      v, n, i = ra._box(0, 0, 0, S * depth / 2, S * sq_w, 0.004)
      R = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
      geo = ((np.array(v) @ R.T + [ctr[0], ctr[1], 0.05]).tolist(), (np.array(n) @ R.T).tolist(), i)
      (black if (r_ + c) % 2 else white).append(geo)
  add(ra._merge(black), (0.03, 0.03, 0.03))
  add(ra._merge(white), (0.95, 0.95, 0.95))

  # grass under and around the track
  lo, hi = pts.min(0) - 22 * S, pts.max(0) + 22 * S
  add(ra._box((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.01, hi[0] - lo[0], hi[1] - lo[1], 0.02), (0.09, 0.15, 0.035), 1.0)

  # low-poly trees, all at least 10 units from the road middle (5 beyond the paved edge)
  rng = np.random.default_rng(seed)
  trunks, crowns = [], []
  placed = []
  tries = 0
  while len(placed) < 90 and tries < 20000:
    tries += 1
    p = rng.uniform(lo + 3 * S, hi - 3 * S)
    if np.min(np.linalg.norm(pts - p, axis=1)) < 10 * S:
      continue
    if placed and np.min(np.linalg.norm(np.array(placed) - p, axis=1)) < 4.5 * S:
      continue
    placed.append(p)
    h = rng.uniform(4.0, 7.5) * S
    r = rng.uniform(1.4, 2.4) * S
    trunks.append(ra._cylinder(p[0], p[1], 0.35 * h / 2, 0.22 * S, 0.35 * h, seg=8))
    crowns.append(_cone(p[0], p[1], 0.3 * h, r, 0.75 * h, seg=9))
  add(ra._merge(trunks), (0.18, 0.11, 0.06))
  add(ra._merge(crowns), (0.05, 0.14, 0.035), 1.0)
  return parts


def car_parts():
  """Donkey-like RC car in its body frame (x forward, y left, z up, metres; origin on the ground at
  the car centre): dark chassis plate, white shell, white camera mast, black tyres."""
  S = SCALE
  L, W = CAR_LEN * S, CAR_WID * S
  r, tw = 0.32 * S, 0.28 * S
  wx, wy = 0.78 * S, (W - tw) / 2
  dark = [ra._box(0, 0, 0.30 * S, 1.75 * S, W - 2 * tw - 0.05 * S, 0.10 * S),
          ra._box(0.95 * S, 0, 0.28 * S, 0.12 * S, W - 0.1 * S, 0.08 * S),   # front bumper
          ra._box(-0.95 * S, 0, 0.28 * S, 0.12 * S, W - 0.1 * S, 0.08 * S)]
  shell = [ra._box(-0.05 * S, 0, 0.45 * S, 1.55 * S, W - 2 * tw - 0.10 * S, 0.20 * S),
           ra._box(0.10 * S, 0.22 * S, 0.80 * S, 0.12 * S, 0.08 * S, 0.55 * S),  # camera mast posts
           ra._box(0.10 * S, -0.22 * S, 0.80 * S, 0.12 * S, 0.08 * S, 0.55 * S),
           ra._box(0.10 * S, 0, 1.08 * S, 0.30 * S, 0.56 * S, 0.06 * S)]
  accent = [ra._box(-0.55 * S, 0, 0.57 * S, 0.45 * S, 0.55 * S, 0.05 * S)]
  tyres = []
  Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)  # cylinder axis z -> y
  for sx in (1, -1):
    for sy in (1, -1):
      v, n, i = ra._cylinder(0, 0, 0, r, tw, seg=18)
      tyres.append(((np.array(v) @ Rx.T + [sx * wx, sy * wy, r]).tolist(), (np.array(n) @ Rx.T).tolist(), i))
  out = []
  for parts, rgb, rough in ((dark, (0.08, 0.08, 0.10), 0.7), (shell, (0.92, 0.92, 0.95), 0.5),
                            (accent, (0.10, 0.25, 0.80), 0.5), (tyres, (0.03, 0.03, 0.03), 0.9)):
    V, N, I = ra._merge(parts)
    out.append((V, N, I, rgb, rough))
  return out


def to_glb(parts):
  M = np.asarray(ra.GLTF_FROM_BODY, np.float64)
  prims = []
  for V, N, I, rgb, rough in parts:
    if len(V) >= 65535:
      raise ValueError(f'primitive with {len(V)} vertices does not fit 16-bit indices')
    if np.linalg.det(M) < 0:
      I = I.reshape(-1, 3)[:, ::-1].reshape(-1)
    prims.append((V @ M.T, N @ M.T, I, ra._material(rgb, 0.0, rough)))
  return ra._glb(prims)


# ----------------------------------------------------------------------------- cameras
def _cam_quat(forward, up_img):
  """xyzw quaternion of a carrier whose +x is `forward` and whose image up is `up_img`."""
  f = np.asarray(forward, np.float64)
  f /= np.linalg.norm(f)
  u = np.asarray(up_img, np.float64)
  u = u - f * (u @ f)
  u /= np.linalg.norm(u)
  return ra.mat_to_quat(np.stack([f, np.cross(u, f), u], -1))


def _down_quat(up_img, tilt_deg=0.5):
  """Camera looking (almost) straight down with image up = `up_img` (horizontal). Tilted by
  `tilt_deg` towards image up: AirSim converts poses through Euler angles and drops the yaw of an
  exactly vertical camera (pitch -90 deg, gimbal lock)."""
  u = np.asarray(up_img, np.float64)
  t = math.radians(tilt_deg)
  return _cam_quat(np.array([0.0, 0.0, -math.cos(t)]) + math.sin(t) * u / np.linalg.norm(u), u)


def pilot_cams(pos, psi):
  fwd_, left_, up_, pitch = PILOT_CAM
  c, s = np.cos(psi), np.sin(psi)
  cam = pos + SCALE * np.stack([fwd_ * c - left_ * s, fwd_ * s + left_ * c, np.full_like(psi, up_)], -1)
  p = math.radians(pitch)
  fwd = np.stack([c * math.cos(p), s * math.cos(p), np.full_like(psi, -math.sin(p))], -1)
  return cam, ra.look_at_quat(cam, cam + fwd)


def chase_cams(pos, psi, back=5.0, up=2.4, ahead=3.0, look_up=0.6, sigma=4.0):
  """Behind and above the car along its heading smoothed over `sigma` steps; follows the position exactly."""
  ps = ra._smooth(psi[:, None], sigma)[:, 0]
  c, s = np.cos(ps), np.sin(ps)
  cam = pos + SCALE * np.stack([-back * c, -back * s, np.full_like(ps, up)], -1)
  tgt = pos + SCALE * np.stack([ahead * c, ahead * s, np.full_like(ps, look_up)], -1)
  return cam, ra.look_at_quat(cam, tgt)


def hawk_cam(nodes, frame, width, height, margin=1.08):
  """Fixed camera straight above the track centre looking down; image right = Unity +z (the track's
  long axis), image up = Unity -x. Height chosen so the paved road and the margin fit."""
  pts_w, _, left = centre_line(nodes)
  edge = np.vstack([pts_w + (ROAD_SHIFT + ROAD_HALF) * left, pts_w + (ROAD_SHIFT - ROAD_HALF) * left])
  xy = frame.xy(edge[:, 0], edge[:, 1])
  centre = (xy.min(0) + xy.max(0)) / 2
  half_right = (xy[:, 1].max() - xy[:, 1].min()) / 2 * margin   # contract y = image right
  half_up = (xy[:, 0].max() - xy[:, 0].min()) / 2 * margin      # contract x = image down
  th = math.tan(math.radians(HFOV['hawk']) / 2)
  h = max(half_right / th, half_up / (th * height / width))
  cam = np.array([centre[0], centre[1], h])
  return cam, _down_quat([-1, 0, 0])


def overhead_cam(pos, psi, width=640, hfov=HFOV['chase'], cover=34.0):
  """Camera fixed to the car looking straight down, image up = car forward (like Donkey's overhead
  camera); its height makes the image `cover` units wide (Donkey overhead: about 34 units)."""
  h = SCALE * cover / 2 / math.tan(math.radians(hfov) / 2)
  cam = pos + np.array([0, 0, h])
  return cam, _down_quat([math.cos(psi), math.sin(psi), 0])


# ----------------------------------------------------------------------------- one sim instance
class Filmer(ra._Worker):
  """One Blocks instance + client: the track and (optionally) the car model as spawned objects, the
  camera as the AirSim robot. Lock-step like ra._Worker.render_stream."""

  def __init__(self, a):  # noqa: super().__init__ needs the renderer's pipes and shared memory
    self.a = a
    self.sim = None
    self.cv = threading.Condition()
    self.want, self.msg = -1, None
    self.tilt = (1.0, 0.0, 0.0, 0.0)
    self.car = None

  def connect_scene(self, track_glb, car_glb):
    from projectairsim import Drone, ProjectAirSimClient, World
    a = self.a
    scene, robot = ra._sim_configs(a.width, a.height, a.hfov, 0.0)
    json.dump(scene, open(os.path.join(a.home, 'scene.jsonc'), 'w'))
    json.dump(robot, open(os.path.join(a.home, 'robot.jsonc'), 'w'))
    self.client = ProjectAirSimClient(port_topics=self.port_topics, port_services=self.port_services)
    self.client.connect()
    self.world = World(self.client, 'scene.jsonc', delay_after_load_sec=2, sim_config_path=a.home)
    self.robot = Drone(self.client, self.world, 'Cam')
    self.client.subscribe(self.robot.sensors['Fpv']['scene_camera'], self._on_image)
    self.world.pause()
    self.t = self.world.get_sim_time()
    self.world.spawn_object_from_file('DonkeyTrack', 'gltf', track_glb, True,
                                      self._pose(*ra.to_ned(ORIGIN, (0, 0, 0, 1))), [1.0, 1.0, 1.0], False)
    if car_glb is not None:
      self.car = self.world.spawn_object_from_file('DonkeyCar', 'gltf', car_glb, True,
                                                   self._pose(*ra.to_ned(ra.DRONE_HIDDEN_POS, (0, 0, 0, 1))), [1.0, 1.0, 1.0], False)

  def shoot(self, cam_pos, cam_quat, car_pos=None, car_quat=None):
    """Set car + camera for this frame, step once; returns the message (pixels of the PREVIOUS step)."""
    if self.car is not None and car_pos is not None:
      self.world.set_object_pose(self.car, self._pose(*ra.to_ned(car_pos, car_quat)), True)
    ned = ra.to_ned(cam_pos, cam_quat)
    self.robot.set_pose(self._pose(*ned), True)
    msg = self.step()
    self._check(msg, *ned)
    return msg


def run_job(job):
  """Render one list of shots on one sim; writes <out>/<path> JPEGs."""
  import cv2
  a = argparse.Namespace(**job['sim'])
  os.makedirs(a.home, exist_ok=True)
  os.chdir(a.home)
  f = Filmer(a)
  shots = job['shots']
  t0 = time.time()
  try:
    f.launch_sim()
    record = Path(job['record'])
    rows, nodes = load(record)
    frame = Frame(nodes)
    f.connect_scene(to_glb(track_parts(nodes, frame)), to_glb(car_parts()) if job['car'] else None)
    first = shots[0]
    for _ in range(WARMUP):
      f.shoot(first['cam'], first['camq'], first.get('car'), first.get('carq'))
    t_start = time.time()
    prev = None
    for shot in shots + [None]:  # the last None flushes: same pose, fetch the last render
      s = shot or prev
      msg = f.shoot(s['cam'], s['camq'], s.get('car'), s.get('carq'))
      if prev is not None:
        img = np.frombuffer(msg['data'], np.uint8).reshape(msg['height'], msg['width'], 3)  # BGR
        out = Path(job['out']) / prev['path']
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
      prev = shot
    dt = time.time() - t_start
    print(json.dumps({'job': job['name'], 'shots': len(shots), 'render_s': round(dt, 2),
                      'fps': round(len(shots) / dt, 2), 'startup_s': round(t_start - t0, 1),
                      'gpu': f.sim_gpu()}), flush=True)
  except BaseException:
    import traceback
    traceback.print_exc()
    code = 1
  else:
    code = 0
  if f.sim is not None:
    ra._kill_group(f.sim.pid)
  sys.stdout.flush()
  os._exit(code)  # the client library keeps non-daemon threads alive


# ----------------------------------------------------------------------------- probe
def probe(args):
  """List the Blocks level objects (bounding boxes) and film the track area from high above."""
  import cv2
  a = sim_args(args, 'probe', 0, 960, 720, 70.0)
  os.makedirs(a.home, exist_ok=True)
  os.chdir(a.home)
  f = Filmer(a)
  try:
    f.launch_sim()
    rows, nodes = load(args.record)
    frame = Frame(nodes)
    f.connect_scene(to_glb(track_parts(nodes, frame)), to_glb(car_parts()))
    boxes = {}
    for name in f.world.list_objects('.*'):
      try:
        boxes[name] = f.world.get_3d_bounding_box(name, 0)
      except Exception as e:  # noqa: BLE001
        boxes[name] = str(e)
    (args.out / 'probe_objects.json').write_text(json.dumps(boxes, indent=1, default=str))
    pos, psi = car_poses(rows, frame)
    shots = {
        'probe_top': (np.array([ORIGIN[0], ORIGIN[1], 160.0]), _down_quat([1, 0, 0])),
        'probe_side': (np.array([ORIGIN[0] - 90, ORIGIN[1] - 90, 60.0]), ra.look_at_quat(np.array([ORIGIN[0] - 90, ORIGIN[1] - 90, 60.0]), np.array(ORIGIN))),
        'probe_wide': (np.array([0.0, 0.0, 400.0]), _down_quat([1, 0, 0])),
    }
    car_q = yaw_quat(psi[0])
    for name, (cam, q) in shots.items():
      for _ in range(WARMUP):
        f.shoot(cam, q, pos[0], car_q)
      msg = f.shoot(cam, q, pos[0], car_q)
      img = np.frombuffer(msg['data'], np.uint8).reshape(msg['height'], msg['width'], 3)
      cv2.imwrite(str(args.out / f'{name}.jpg'), img)
    print('probe written to', args.out, flush=True)
  except BaseException:
    import traceback
    traceback.print_exc()
    code = 1
  else:
    code = 0
  if f.sim is not None:
    ra._kill_group(f.sim.pid)
  sys.stdout.flush()
  os._exit(code)


def sim_args(args, name, gpu, width, height, hfov):
  return argparse.Namespace(
      home=str(args.out / 'homes' / name), root=ra.AIRSIM_ROOT, gpu=gpu, cvars=ra.LOOKS['stock'],
      startup_timeout=150.0, frame_timeout=30.0, width=width, height=height, hfov=hfov, mount_pitch=0.0)


# ----------------------------------------------------------------------------- main
def plan(args):
  rows, nodes = load(args.record)
  frame = Frame(nodes)
  n = len(rows) if args.frames <= 0 else min(args.frames, len(rows))
  pos, psi = car_poses(rows, frame)
  carq = yaw_quat(psi)
  W, H = args.width, args.height
  cams = {'pilot': pilot_cams(pos, psi), 'chase': chase_cams(pos, psi)}
  hc, hq = hawk_cam(nodes, frame, W, H)
  cams['hawk'] = (np.repeat(hc[None], len(rows), 0), np.repeat(hq[None], len(rows), 0))
  shots = {v: [{'path': f"{v}/{rows[k]['frame']}.jpg", 'cam': cams[v][0][k].tolist(), 'camq': cams[v][1][k].tolist(),
                'car': pos[k].tolist(), 'carq': carq[k].tolist()} for k in range(n)] for v in args.views}
  verify = []
  for k in args.verify:
    c, q = overhead_cam(pos[k], psi[k], W)
    verify.append({'path': f"verify/overhead_{rows[k]['frame']}.jpg", 'cam': c.tolist(), 'camq': q.tolist(),
                   'car': pos[k].tolist(), 'carq': carq[k].tolist()})
  gpus = [int(g) for g in args.gpus.split(',')]
  jobs = []
  per_view = max(1, args.instances // len(args.views))
  for v in args.views:
    chunks = np.array_split(np.arange(n), per_view)
    for ci, idx in enumerate(chunks):
      jobs.append({'name': f'{v}{ci}', 'view': v, 'shots': [shots[v][i] for i in idx], 'car': v != 'pilot'})
  if verify:
    jobs.append({'name': 'verify', 'view': 'chase', 'shots': verify, 'car': True})
  for j, job in enumerate(jobs):
    job.update(record=str(args.record.resolve()), out=str(args.out.resolve()),
               sim=vars(sim_args(args, job['name'], gpus[j % len(gpus)], W, H, HFOV[job['view']])))
    job['sim']['home'] = str(Path(job['sim']['home']).resolve())
  meta = {'scale_m_per_unit': SCALE, 'origin': ORIGIN, 'hawk_cam': hc.tolist(), 'frames': n,
          'hfov': HFOV, 'width': W, 'height': H}
  return jobs, meta


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--record', type=Path, default=Path('runs/video_p3'))
  ap.add_argument('--out', type=Path, default=Path('/tmp/donkey_unreal'))
  ap.add_argument('--gpus', default='0,1')
  ap.add_argument('--instances', type=int, default=6, help='sims in total, split evenly over views and GPUs')
  ap.add_argument('--views', default=','.join(VIEWS))
  ap.add_argument('--frames', type=int, default=0, help='first N steps only (0 = all)')
  ap.add_argument('--verify', default='0,100,300', help='steps to also film with a car-fixed overhead camera')
  ap.add_argument('--width', type=int, default=640)
  ap.add_argument('--height', type=int, default=480)
  ap.add_argument('--probe', action='store_true')
  ap.add_argument('--job', type=Path, help=argparse.SUPPRESS)
  args = ap.parse_args()
  if args.job:
    run_job(json.loads(args.job.read_text()))
    return
  args.record, args.out = args.record.resolve(), args.out.resolve()
  args.out.mkdir(parents=True, exist_ok=True)
  args.views = [v for v in args.views.split(',') if v]
  args.verify = [int(v) for v in args.verify.split(',') if v]
  if args.probe:
    probe(args)
    return
  jobs, meta = plan(args)
  (args.out / 'meta.json').write_text(json.dumps(meta, indent=1))
  procs = []
  t0 = time.time()
  for job in jobs:
    path = args.out / f"job_{job['name']}.json"
    path.write_text(json.dumps(job))
    log = open(args.out / f"job_{job['name']}.log", 'w')
    procs.append((job['name'], subprocess.Popen([sys.executable, os.path.abspath(__file__), '--job', str(path)],
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)))
  failed = []
  try:
    for name, p in procs:
      if p.wait() != 0:
        failed.append(name)
  finally:
    for _, p in procs:
      ra._kill_group(p.pid)
  dt = time.time() - t0
  total = sum(len(j['shots']) for j in jobs)
  for name, _ in procs:
    lines = (args.out / f'job_{name}.log').read_text().splitlines()
    print(name, next((ln for ln in reversed(lines) if ln.startswith('{"job"')), 'NO RESULT: ' + ' | '.join(lines[-3:])))
  print(f'{total} frames from {len(jobs)} sims in {dt:.0f} s wall ({total / dt:.1f} frames/s overall)')
  missing = [j['name'] for j in jobs if not all((args.out / s['path']).exists() for s in j['shots'])]
  if missing or failed:
    raise SystemExit(f'jobs with missing frames: {missing}, failed: {failed}')


if __name__ == '__main__':
  main()
