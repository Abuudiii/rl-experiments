"""Saves what is needed for a demo video while the Donkey env runs: both cameras, telemetry and the track."""
import base64
import json
import pathlib
import time

SIZE = dict(img_w=320, img_h=240, img_d=3, img_enc='JPG')
# Second camera, behind and above the car, tilted down (found by trial; units are the sim's).
CHASE = dict(offset_y=1.5, offset_z=-3.0, rot_x=18)


class Recorder:

  def __init__(self, handler, folder):
    self._handler = handler
    self._folder = pathlib.Path(folder)
    self._folder.mkdir(parents=True, exist_ok=True)
    self._pilot = None
    self._chase = None
    self._nodes = {}
    self._count = 0
    self._log = open(self._folder / 'steps.jsonl', 'a')
    telemetry = handler.fns['telemetry']
    def capture(message):
      self._pilot = message['image']
      # gym-donkeycar looks for "image_b" but the sim sends "imageb", so read it here.
      self._chase = message.get('imageb', self._chase)
      telemetry(message)
    handler.fns['telemetry'] = capture
    handler.fns['node_position'] = self._node
    handler.send_cam_config(**SIZE)
    handler.send_cam_config(msg_type='cam_config_b', **SIZE, **CHASE)
    for index in range(handler.total_nodes or 0):
      handler.blocking_send({'msg_type': 'node_position', 'index': str(index)})
    time.sleep(1.0)
    track = [self._nodes[i] for i in sorted(self._nodes)]
    (self._folder / 'track.json').write_text(json.dumps(track))

  def _node(self, message):
    self._nodes[int(message['index'])] = (message['pos_x'], message['pos_z'])

  def write(self, row):
    if self._pilot is None or self._chase is None:
      return
    name = f'{self._count:06d}'
    (self._folder / f'pilot_{name}.jpg').write_bytes(base64.b64decode(self._pilot))
    (self._folder / f'chase_{name}.jpg').write_bytes(base64.b64decode(self._chase))
    self._log.write(json.dumps({'frame': name, 'time': time.time(), **row}) + '\n')
    self._log.flush()
    self._count += 1
