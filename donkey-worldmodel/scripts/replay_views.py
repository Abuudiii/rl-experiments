"""Film a recorded drive at high resolution: pilot, chase and overhead views.

  python scripts/replay_views.py --port P --record DIR [--width 512 --height 384]

The drive itself is recorded with the agent's normal camera only (envs/donkey_record.py), so the simulator
runs at full speed. This puts a car back at each recorded position, in the same simulator and scene, and
saves pilot_<frame>.png, chase_<frame>.png and overhead_<frame>.png into DIR. Run it right after the
recording, before the simulator is restarted, so trees and lighting are the same.

A car has two cameras, so the drive is replayed twice: pilot + chase, then overhead.
"""
import argparse
import base64
import json
import math
import pathlib
import time

import gymnasium
import gym_donkeycar  # noqa: registers the env ids

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--record', type=pathlib.Path, required=True)
parser.add_argument('--task', default='generated-track')
parser.add_argument('--width', type=int, default=512)    # the simulator's largest camera is 512 wide
parser.add_argument('--height', type=int, default=384)
parser.add_argument('--overhead', type=float, default=12.0, help='height of the overhead camera')
args = parser.parse_args()

SIZE = dict(img_w=args.width, img_h=args.height, img_d=3, img_enc='PNG')
# Second camera, behind and above the car, tilted down (found by trial; units are the sim's).
CHASE = dict(offset_y=1.5, offset_z=-3.0, rot_x=18)

rows = [json.loads(line) for line in (args.record / 'steps.jsonl').read_text().splitlines()]
end = next((i for i, row in enumerate(rows) if row['last']), len(rows) - 1)
rows = rows[:end + 1]   # the first episode

env = gymnasium.make(
    f'donkey-{args.task}-v0', disable_env_checker=True,
    conf=dict(host='127.0.0.1', port=args.port, start_delay=0.0, log_level=30, max_cte=1e9))
handler = env.unwrapped.viewer.handler
latest = {'count': 0}
telemetry = handler.fns['telemetry']
def capture(message):
  latest['a'] = message['image']
  # gym-donkeycar looks for "image_b" but the sim sends "imageb", so read it here.
  latest['b'] = message.get('imageb')
  latest['count'] += 1
  telemetry(message)
handler.fns['telemetry'] = capture
env.reset()


def replay(views):
  """views: {name: 'a' or 'b'}; saves <name>_<frame>.png for every recorded step."""
  for number, row in enumerate(rows):
    half = math.radians(row['yaw']) / 2
    handler.blocking_send({
        'msg_type': 'set_position',
        # Telemetry positions are the sim's world positions divided by 8.
        'pos_x': str(row['x'] * 8), 'pos_y': str(row['y'] * 8), 'pos_z': str(row['z'] * 8),
        'Qx': '0', 'Qy': str(math.sin(half)), 'Qz': '0', 'Qw': str(math.cos(half))})
    handler.send_control(0, 0, 1.0)   # hand brake, so the car stays where it was put
    # Wait for two fresh frames: the first may have been rendered before the move.
    seen = latest['count']
    while latest['count'] < seen + 2:
      time.sleep(0.002)
    for name, camera in views.items():
      (args.record / f"{name}_{row['frame']}.png").write_bytes(base64.b64decode(latest[camera]))
    if number % 200 == 0:
      print(f'{list(views)}: {number}/{len(rows)}', flush=True)


handler.send_cam_config(**SIZE)                                  # camera A where the agent's camera is
handler.send_cam_config(msg_type='cam_config_b', **SIZE, **CHASE)
time.sleep(1.5)
replay({'pilot': 'a', 'chase': 'b'})
handler.send_cam_config(**SIZE, offset_y=args.overhead, rot_x=90)   # camera A above the car, looking down
time.sleep(1.5)
replay({'overhead': 'a'})
env.close()
print('done', len(rows))
