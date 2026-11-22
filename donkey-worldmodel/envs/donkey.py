import time

import elements
import embodied
import numpy as np
from PIL import Image


class Donkey(embodied.Env):
  """Donkey Car simulator (already running, one sim per env at port_base + index)."""

  def __init__(
      self, task, index=0, size=(64, 64), length=2000, max_laps=0,
      port_base=9200, host='127.0.0.1', max_cte=4.0,
      throttle_min=0.2, throttle_max=0.6, stall_steps=200, reload_seconds=0,
      record='', steer_rate=0.0, smooth_cost=0.0, steer_smooth=0.0):
    self._id = f'donkey-{task}-v0'
    self._conf = dict(
        host=host, port=int(port_base) + int(index), max_cte=float(max_cte),
        steer_limit=1.0, throttle_min=float(throttle_min),
        throttle_max=float(throttle_max), start_delay=0.0, log_level=30)
    self._size = tuple(size)
    self._length = int(length)
    self._max_laps = int(max_laps)
    self._stall_steps = int(stall_steps)
    self._max_cte = float(max_cte)
    self._reload_seconds = float(reload_seconds)
    self._loaded_at = None
    self._steer_rate = float(steer_rate)     # largest steering change per step; 0 = unlimited
    self._smooth_cost = float(smooth_cost)   # reward lost per unit the wheel actually moves in a step
    # Low-pass on steering: each step the wheel covers this fraction of the gap to the request; 0 = off.
    self._steer_smooth = float(steer_smooth)
    self._last_step = 0.0                    # previous step's steering movement, to count reversals
    self._steer = 0.0                        # steering currently applied to the car
    self._asked = 0.0                        # steering the policy asked for on the previous step
    self._record = record     # folder for demo-video material, or '' for none
    self._recorder = None
    self._action = np.zeros(2, np.float32)
    self._low = np.array([-1.0, throttle_min], np.float32)
    self._high = np.array([1.0, throttle_max], np.float32)
    self._env = None
    self._handler = None
    self._done = True
    self._progress = 0.0
    self.frame = None  # last raw camera frame, for inspection scripts

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self._size + (3,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/laps': elements.Space(np.float32),
        'log/sim_laps': elements.Space(np.float32),
        'log/sim_lap_time': elements.Space(np.float32),
        'log/cte_abs': elements.Space(np.float32),
        'log/hit': elements.Space(np.float32),
        'log/speed': elements.Space(np.float32),
        'log/dt': elements.Space(np.float32),
        'log/steer_change': elements.Space(np.float32),
        'log/steer_flip': elements.Space(np.float32),
        'log/start_speed': elements.Space(np.float32),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.float32, (2,), self._low, self._high),
        'reset': elements.Space(bool),
    }

  @property
  def handler(self):
    return self._handler

  def _connect(self):
    import gymnasium
    import gym_donkeycar  # noqa: registers the env ids
    self._env = gymnasium.make(
        self._id, conf=self._conf, disable_env_checker=True)
    handler = self._env.unwrapped.viewer.handler
    handler.active_node = None
    handler.total_nodes = None
    original = handler.fns['telemetry']
    def telemetry(message):
      if 'activeNode' in message:
        handler.active_node = int(message['activeNode'])
        handler.total_nodes = int(message['totalNodes'])
      original(message)
    handler.fns['telemetry'] = telemetry
    self._handler = handler
    self._loaded_at = time.time()
    if self._record:
      from . import donkey_record
      frame, _ = self._env.reset()   # telemetry must be flowing before the track size is known
      self._recorder = donkey_record.Recorder(handler, self._record)

  def _reload(self):
    """Load the scene again; the sim then regenerates trees, cones and lighting."""
    self._handler.send_exit_scene()
    time.sleep(1.0)
    self._env.close()
    # The sim needs several seconds to drop the old car; reconnecting sooner (3 s) starts the new car
    # in a second start slot beside the centre line.
    time.sleep(6.0)
    self._connect()   # a new connection from the sim's menu loads the scene

  def _reset(self):
    if self._reload_seconds and time.time() - self._loaded_at > self._reload_seconds:
      self._reload()
    for _ in range(4):
      frame, info = self._env.reset()
      # A start off the centre line means the old car was still there; reload again.
      if abs(info['cte']) < self._max_cte / 2:
        return frame, info
      self._reload()
    raise RuntimeError(f"car keeps starting off the centre line, cte {info['cte']}")

  def step(self, action):
    if self._env is None:
      self._connect()
    if action['reset'] or self._done:
      frame, info = self._reset()
      if self._handler.active_node is None:
        raise RuntimeError('sim telemetry has no activeNode/totalNodes')
      self._done = False
      self._steps = 0
      self._progress = 0.0      # nodes travelled forward this episode
      self._best = 0.0
      self._best_step = 0
      self._node = self._handler.active_node
      self._steer = 0.0
      self._asked = 0.0
      self._last_step = 0.0
      self._time = time.time()
      return self._obs(frame, 0.0, info, 0.0, is_first=True)
    asked = np.asarray(action['action'], np.float32)
    self._asked = float(asked[0])
    # The wheel cannot jump: move the applied steering towards the request by at most steer_rate.
    step = float(asked[0]) - self._steer
    if self._steer_smooth:
      step *= self._steer_smooth
    if self._steer_rate:
      step = float(np.clip(step, -self._steer_rate, self._steer_rate))
    self._steer_change = abs(step)
    # A reversal: the wheel was moving one way and now moves the other (ignoring tiny movements).
    self._steer_flip = float(step * self._last_step < 0 and abs(step) > 0.02 and abs(self._last_step) > 0.02)
    self._last_step = step
    self._steer = float(self._steer + step)
    self._action = np.array([self._steer, asked[1]], np.float32)
    frame, _, over, _, info = self._env.step(self._action)
    total = self._handler.total_nodes
    delta = (self._handler.active_node - self._node) % total
    if delta > total / 2:
      delta -= total            # moved backwards across the wrap
    self._node = self._handler.active_node
    self._progress += delta
    self._steps += 1
    if self._progress > self._best:
      self._best, self._best_step = self._progress, self._steps
    # One full lap = +100; moving the wheel costs a little. The cost is on the applied steering, after the
    # filter: training samples noisy requests, and a cost on those would mostly punish the noise.
    reward = 100.0 * delta / total - self._smooth_cost * self._steer_change
    laps = self._progress / total
    terminal = bool(over)                 # left the track or hit something
    if terminal:
      reward = -10.0
    timeout = (
        self._steps >= self._length or
        self._steps - self._best_step >= self._stall_steps or
        (self._max_laps > 0 and laps >= self._max_laps))
    self._done = terminal or timeout
    dt = time.time() - self._time
    self._time = time.time()
    return self._obs(
        frame, reward, info, dt, is_last=self._done, is_terminal=terminal)

  def _obs(self, frame, reward, info, dt, is_first=False, is_last=False,
           is_terminal=False):
    self.frame = frame
    image = np.asarray(Image.fromarray(frame).resize(
        self._size[::-1], Image.BILINEAR), np.uint8)
    laps = self._progress / max(1, self._handler.total_nodes)
    obs = {
        'image': image,
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
        'log/laps': np.float32(laps),
        # The simulator's own start-line counter and last lap time, as an independent check of log/laps.
        'log/sim_laps': np.float32(info['lap_count']),
        'log/sim_lap_time': np.float32(info['last_lap_time']),
        'log/cte_abs': np.float32(abs(info['cte'])),
        'log/hit': np.float32(info['hit'] != 'none'),   # 1 when the sim reports a collision
        'log/speed': np.float32(info['speed']),
        'log/dt': np.float32(dt),
        # How far the applied steering moved this step; its episode average measures smoothness.
        'log/steer_change': np.float32(0.0 if is_first else self._steer_change),
        # 1 on steps where the wheel reverses direction; episode average x 20 = reversals per second.
        'log/steer_flip': np.float32(0.0 if is_first else self._steer_flip),
        # Speed on the first step of the episode; should be 0 if the reset really stopped the car.
        'log/start_speed': np.float32(info['speed'] if is_first else 0.0),
    }
    if self._recorder:
      self._recorder.write(dict(
          x=info['pos'][0], y=info['pos'][1], z=info['pos'][2], yaw=info['car'][2],
          cte=info['cte'], speed=info['speed'],
          steer=float(self._action[0]), asked=self._asked, throttle=float(self._action[1]), laps=float(laps),
          sim_laps=info['lap_count'], first=is_first, last=is_last, terminal=is_terminal))
    return obs

  def close(self):
    if self._env is not None:
      self._env.close()
