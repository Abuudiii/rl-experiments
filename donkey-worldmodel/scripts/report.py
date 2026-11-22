"""Summarise a Donkey DreamerV3 run: time to first lap, learning curve, eval laps, and the videos worth showing.

  python scripts/report.py RUN_DIR OUT_DIR [--eval EVAL_DIR ...] [--hz 20]
"""
import argparse
import json
import pathlib
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SCORE = 'episode/score'


def series(path, key):
  """(steps, values) of one scalar from a DreamerV3 metrics.jsonl."""
  steps, values = [], []
  for line in path.read_text().splitlines():
    row = json.loads(line)
    if key in row:
      steps.append(row['step'])
      values.append(row[key])
  return steps, values


def videos(run, key):
  """Video files logged under a metric key, oldest first."""
  folder = run / 'scope' / (key.replace('/', '-') + '.mp4')
  return sorted(folder.glob('*.mp4')) if folder.is_dir() else []


parser = argparse.ArgumentParser()
parser.add_argument('run', type=pathlib.Path)
parser.add_argument('out', type=pathlib.Path)
parser.add_argument('--eval', type=pathlib.Path, nargs='*', default=[])
parser.add_argument('--hz', type=float, default=20.0)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
minutes = lambda steps: [s / args.hz / 60 for s in steps]

# Return is 100 per lap, minus 10 if the episode ended off the track, so return >= 100 means a full lap.
metrics = args.run / 'metrics.jsonl'
score_steps, scores = series(metrics, SCORE)
laps = [max(0.0, s / 100) for s in scores]
print(f'run {args.run}: {len(scores)} episodes, {score_steps[-1] if score_steps else 0} steps')
for target in (1, 5, 10):
  first = next((s for s, v in zip(score_steps, laps) if v >= target), None)
  if first is None:
    print(f'no episode reached {target} lap(s)')
  else:
    print(f'first episode with {target} lap(s): step {first} = {first / args.hz / 60:.1f} minutes of driving')
print(f'best episode: {max(laps, default=0):.2f} laps')

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(minutes(score_steps), laps, '.', markersize=3, alpha=0.5)
ax.axhline(1.0, color='gray', linestyle=':')
ax.set_ylabel('laps per episode (episodes capped at 100 s)')
ax.set_xlabel('minutes of driving (all cars summed)')
fig.tight_layout()
fig.savefig(args.out / 'curve.png', dpi=150)
print('wrote', args.out / 'curve.png')

for folder in args.eval:
  _, eval_scores = series(folder / 'metrics.jsonl', SCORE)
  _, lengths = series(folder / 'metrics.jsonl', 'episode/length')
  print(f'eval {folder.name}: laps per episode', [round(max(0.0, s / 100), 2) for s in eval_scores],
        'steps', [int(n) for n in lengths])

policy = videos(args.run, 'epstats/policy_image')
dream = videos(args.run, 'report/openloop/image')
for name, files, index in [('drive_before.mp4', policy, 0), ('drive_after.mp4', policy, -1), ('dream.mp4', dream, -1)]:
  if files:
    shutil.copy(files[index], args.out / name)
    print('wrote', args.out / name, '<-', files[index].name)
  else:
    print('missing video for', name)
