"""Saves what is needed for a demo video while the Donkey env runs: the agent's camera frames, telemetry
with poses, and the track. It does not touch the cameras, so the simulator runs exactly as in evaluation.
The high-resolution views are filmed afterwards by scripts/replay_views.py from the recorded poses."""
import base64
import json
import pathlib
import time


class Recorder:

  def __init__(self, handler, folder):
    self._handler = handler
    self._folder = pathlib.Path(folder)
    self._folder.mkdir(parents=True, exist_ok=True)
    self._agent = None
    self._nodes = {}
    self._count = 0
    self._log = open(self._folder / 'steps.jsonl', 'a')
    telemetry = handler.fns['telemetry']
    def capture(message):
      self._agent = message['image']
      telemetry(message)
    handler.fns['telemetry'] = capture
    handler.fns['node_position'] = self._node
    for index in range(handler.total_nodes or 0):
      handler.blocking_send({'msg_type': 'node_position', 'index': str(index)})
    time.sleep(1.0)
    track = [self._nodes[i] for i in sorted(self._nodes)]
    (self._folder / 'track.json').write_text(json.dumps(track))

  def _node(self, message):
    self._nodes[int(message['index'])] = (message['pos_x'], message['pos_z'])

  def write(self, row):
    if self._agent is None:
      return
    name = f'{self._count:06d}'
    # The camera frame as the simulator sent it (160x120 JPEG); the agent gets it resized to 64x64.
    (self._folder / f'agent_{name}.jpg').write_bytes(base64.b64decode(self._agent))
    self._log.write(json.dumps({'frame': name, 'time': time.time(), **row}) + '\n')
    self._log.flush()
    self._count += 1
