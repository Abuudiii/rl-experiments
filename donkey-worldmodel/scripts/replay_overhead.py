"""Film a recorded drive from above.

  python scripts/replay_overhead.py --port P --record DIR [--height 12]

A car has only two cameras, and both were used during the drive (pilot and chase). This puts a car back at
each recorded position, in the same simulator and scene, with its camera above it looking straight down, and
saves one overhead_<frame>.jpg per recorded step into DIR. Run it right after the recording, before the
simulator is restarted, so trees and cones are the same.
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
parser.add_argument('--height', type=float, default=12.0)
args = parser.parse_args()

rows = [json.loads(line) for line in (args.record / 'steps.jsonl').read_text().splitlines()]
env = gymnasium.make(
    f'donkey-{args.task}-v0', disable_env_checker=True,
    conf=dict(host='127.0.0.1', port=args.port, start_delay=0.0, log_level=30, max_cte=1e9))
handler = env.unwrapped.viewer.handler
latest = {'count': 0}
telemetry = handler.fns['telemetry']
def capture(message):
  latest['image'] = message['image']
  latest['count'] += 1
  telemetry(message)
handler.fns['telemetry'] = capture
env.reset()
handler.send_cam_config(img_w=320, img_h=240, img_d=3, img_enc='JPG', offset_y=args.height, rot_x=90)
time.sleep(1.0)

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
  (args.record / f"overhead_{row['frame']}.jpg").write_bytes(base64.b64decode(latest['image']))
  if number % 200 == 0:
    print(f'{number}/{len(rows)}', flush=True)
env.close()
print('done', len(rows))
