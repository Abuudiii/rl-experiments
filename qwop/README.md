# QWOP — DQN and QR-DQN

Teaching an agent to run in [QWOP](https://www.foddy.net/Athletics.html), using
[qwop-python](https://github.com/bicrick/qwop-python) (headless Box2D recreation of the game)
and Stable-Baselines3. All runs on an AMD Radeon AI PRO R9700 (ROCm).

| Run | Model + video | Result |
|---|---|---|
| QR-DQN, 10M steps | https://huggingface.co/Abuudiii/qrdqn-QWOP | **Finishes 100 m** (100.7 m in 107.1 s) |
| DQN, 1M steps | https://huggingface.co/Abuudiii/dqn-QWOP | 8.8 m in 14.3 s, then falls |

## QR-DQN, 10M steps (`config/train_qrdqn_10m.yml`)

Settings from qwop-python's world-record attempt config (`train_qrdqn_wr.yml`): learning rate 0.001,
target network refresh every 512 steps, 9 actions instead of 16, gamma 0.997, epsilon 0.2 → 0,
episode cap 1000. Changes: 10M steps instead of 32M, 8 parallel games (`n_envs: 8`) with
`gradient_steps: 8` to keep 1 update per 4 game steps. 80 min of training.

Every checkpoint, deterministic (the env is deterministic, so one episode is the result):

| Steps | Reward | Distance | Time | End |
|---|---|---|---|---|
| 0.5M | −30.5 | 0.0 m | 133.3 s | timeout |
| 1.0M | 380.8 | 40.4 m | 133.3 s | timeout |
| 1.5M | 282.1 | 25.5 m | 133.3 s | timeout |
| 2.0M | 9.3 | 1.2 m | 133.3 s | timeout |
| 2.5M | 76.4 | 5.2 m | 19.0 s | fell |
| 3.0M | 737.4 | 68.5 m | 133.3 s | timeout |
| 3.5M | 1128.5 | 100.3 m | 121.8 s | **finished** |
| 4.0M | 919.6 | 87.9 m | 133.3 s | timeout |
| 4.5M | 1093.8 | 100.2 m | 131.4 s | **finished** |
| 5.0M | 26.1 | 1.7 m | 3.3 s | fell |
| 5.5M | 972.8 | 91.6 m | 133.3 s | timeout |
| 6.0M | 1127.8 | 100.5 m | 116.8 s | **finished** |
| 6.5M | 1135.7 | 100.1 m | 114.3 s | **finished** |
| 7.0M | 1135.5 | 100.9 m | 116.1 s | **finished** |
| 7.5M | 48.7 | 2.9 m | 6.0 s | fell |
| 8.0M | 1131.5 | 100.3 m | 115.1 s | **finished** |
| 8.5M | 1148.7 | 100.3 m | 111.3 s | **finished** |
| **9.0M** | **1152.4** | **100.7 m** | **107.1 s** | **finished (uploaded)** |
| 9.5M | 1035.2 | 95.3 m | 133.3 s | timeout |
| 10.0M | 1111.6 | 100.6 m | 127.1 s | **finished** |

Some checkpoints still collapse (5.0M, 7.5M), so pick the best checkpoint rather than the last.

## DQN, 1M steps (`config/train_dqn.yml`)

| Checkpoint | Reward | Distance |
|---|---|---|
| 200k | 21.1 | 2.3 m |
| 400k | 34.1 | 2.2 m |
| 600k | 108.7 | 5.5 m |
| 800k | −140.2 | 1.3 m (stands still until the step limit) |
| **1M** | **178.7** | **8.8 m in 14.3 s, then falls** |

Changed from the repo defaults: `total_timesteps` 200k → 1M, `target_update_interval` 100k → 10k,
`exploration_fraction` 0.01 → 0.3 (with 0.01, epsilon hit its floor before `learning_starts`).

## Files

| File | Purpose |
|---|---|
| `qwop-python.patch` | Our changes to qwop-python at commit `0105698` (configs above, stats table printed to stdout) |
| `record_video.py` | Record a trained model (DQN or QR-DQN) to mp4 without a display |

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
TMPDIR=/tmp qwop-python -c config/train_qrdqn_10m.yml train_qrdqn   # or: qwop-python train_dqn
python record_video.py data/QRDQN-<id>/model_9000000_steps.zip qwop.mp4 150
```

- `TMPDIR=/tmp`: with `n_envs > 1`, Python multiprocessing fails to clean up its temp dir on NFS homes.
- `metadata.yml` is written only when training ends. To record a checkpoint of a run that is still
  going (or was killed), pass the config: `python record_video.py <ckpt.zip> out.mp4 150 config/train_qrdqn_10m.yml`.
- To continue training from a model, set `model_load_file` in the config.
