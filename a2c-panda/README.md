# A2C — Panda robot arm
Hugging Face [Deep RL course, unit 6](https://huggingface.co/learn/deep-rl-course/unit6/introduction):
Advantage Actor-Critic (A2C) with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) on
[panda-gym](https://github.com/qgallouedec/panda-gym) `PandaReachDense-v3` (and optionally `PandaPickAndPlace-v3`).
Settings are the notebook's: 4 parallel envs, `VecNormalize` on observations and rewards, default A2C, 1M steps.

**Model:** https://huggingface.co/Abuudiii/a2c-PandaReachDense-v3
## Result
| | |
|---|---|
| Training | 1M steps, ~24 min on an NVIDIA RTX PRO 4000 Blackwell; mean episode reward −10 → −0.26 by 50k steps, −0.20 at 1M |
| Evaluation (10 episodes) | −0.22 ± 0.12 (score −0.34, certification needs ≥ −3.5) |

## Setup
```bash
uv venv .venv && . .venv/bin/activate
uv pip install torch "stable-baselines3[extra]" gymnasium huggingface_sb3 huggingface_hub panda_gym moviepy imageio-ffmpeg
```
No `xvfb`, `python-opengl` or system `ffmpeg` is needed: pybullet renders headless.

## Run
```bash
python a2c.py PandaReachDense-v3          # train on the GPU, evaluate, save logs/PandaReachDense-v3/
hf auth login
python a2c.py PandaReachDense-v3 --push <hf-user>   # -> <hf-user>/a2c-PandaReachDense-v3
```
`--steps N` changes the training length. `--push` evaluates the saved model, records `replay.mp4` and uploads
model, normalisation stats, model card and video. `huggingface_sb3`'s own replay video doesn't work with current
SB3 (it reads a removed attribute and calls a system `ffmpeg`), so the script uploads its own.
