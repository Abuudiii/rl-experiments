# QWOP — DQN

A DQN agent learning to run in [QWOP](https://www.foddy.net/Athletics.html), using
[qwop-python](https://github.com/bicrick/qwop-python) (headless Box2D recreation of the game)
and Stable-Baselines3.

**Model + video:** https://huggingface.co/Abuudiii/dqn-QWOP

## Result (1M steps, AMD Radeon AI PRO R9700)

Deterministic evaluation, 10 episodes (the env is deterministic, so episodes are identical):

| Checkpoint | Mean reward | Distance |
|---|---|---|
| 200k | 21.1 | 2.3 m |
| 400k | 34.1 | 2.2 m |
| 600k | 108.7 | 5.5 m |
| 800k | −140.2 | 1.3 m (stands still until the step limit) |
| **1M** | **178.7** | **8.8 m in 14.3 s, then falls** |

## Files

| File | Purpose |
|---|---|
| `qwop-python.patch` | Our changes to qwop-python at commit `0105698` |
| `record_video.py` | Record a trained model to mp4 without a display |

The patch changes:
- `config/train_dqn.yml`: `total_timesteps` 200k → 1M, `target_update_interval` 100k → 10k,
  `exploration_fraction` 0.01 → 0.3. With the original values, epsilon hit its floor before
  `learning_starts` (no exploration once learning began) and the rare target updates made reward swing.
- `train_sb3.py`: print the SB3 stats table to stdout (every 100 episodes), not only to TensorBoard.

## Setup

```bash
git clone https://github.com/bicrick/qwop-python && cd qwop-python
git checkout 0105698
git apply ../qwop-python.patch && cp ../record_video.py .
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install -e ".[sb3]" imageio-ffmpeg
```

qwop-python pins `gymnasium ~= 0.29` and `numpy ~= 1.24`, so keep it in its own venv
(separate from the rl_zoo3 one, which needs gymnasium 1.x).

## Run

```bash
qwop-python train_dqn                                         # saves to data/DQN-<run_id>/
python record_video.py data/DQN-<run_id>/model.zip qwop.mp4   # any checkpoint works
```

To continue training from a model, set `model_load_file` in `config/train_dqn.yml`.
