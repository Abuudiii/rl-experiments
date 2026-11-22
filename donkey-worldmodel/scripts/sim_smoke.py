"""Drive one running simulator with random actions and report whether it is usable for training."""
import argparse
import pathlib
import sys
import time

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'dreamerv3'))
from embodied.envs.donkey import Donkey  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--steps', type=int, default=1200)
parser.add_argument('--out', type=pathlib.Path, required=True)
parser.add_argument('--task', default='generated-track')
parser.add_argument('--max-cte', type=float, default=4.0)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

env = Donkey(args.task, index=0, port_base=args.port, max_cte=args.max_cte)
space = env.act_space['action']
rng = np.random.default_rng(args.port)
tag = f'p{args.port}'


def save(name, obs):
  Image.fromarray(env.frame).save(args.out / f'{tag}_{name}_raw.png')
  Image.fromarray(obs['image']).save(args.out / f'{tag}_{name}_64.png')


obs = env.step({'reset': True, 'action': np.zeros(2, np.float32)})
save('first', obs)
print('telemetry fields: active_node', env.handler.active_node, 'total_nodes', env.handler.total_nodes)
dts, nodes, ctes, means, stds = [], [], [], [], []
ends = {'terminal': 0, 'stall_or_length': 0}
laps_best = 0.0
start = time.time()
for step in range(args.steps):
  if obs['is_last']:
    kind = 'terminal' if obs['is_terminal'] else 'stall_or_length'
    ends[kind] += 1
    if kind == 'terminal' and ends[kind] <= 3:
      save(f'terminal{ends[kind]}', obs)
    obs = env.step({'reset': True, 'action': np.zeros(2, np.float32)})
    continue
  action = rng.uniform(space.low, space.high).astype(np.float32)
  obs = env.step({'reset': False, 'action': action})
  dts.append(float(obs['log/dt']))
  nodes.append(env.handler.active_node)
  ctes.append(float(obs['log/cte_abs']))
  means.append(obs['image'].mean())
  stds.append(obs['image'].std())
  laps_best = max(laps_best, float(obs['log/laps']))
elapsed = time.time() - start
env.close()
dts = np.array(dts)
print(f'[{tag}] steps/s incl. resets {args.steps / elapsed:.1f}; steps/s while driving {1 / dts.mean():.1f}')
print(f'[{tag}] dt mean {dts.mean():.4f} p95 {np.percentile(dts, 95):.4f} max {dts.max():.4f}')
print(f'[{tag}] episode ends {ends}; best laps {laps_best:.3f}')
print(f'[{tag}] activeNode min {min(nodes)} max {max(nodes)} totalNodes {env.handler.total_nodes}')
print(f'[{tag}] cte_abs max {max(ctes):.2f} p50 {np.median(ctes):.2f}')
print(f'[{tag}] image mean {np.mean(means):.1f} std {np.mean(stds):.1f}')
