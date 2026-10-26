# Space Invaders — DQN

Hugging Face [Deep RL course, unit 3](https://huggingface.co/learn/deep-rl-course/unit3/introduction):
a DQN agent trained on `SpaceInvadersNoFrameskip-v4` with [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

**Model:** https://huggingface.co/Abuudiii/dqn-SpaceInvadersNoFrameskip-v4

## Result

| | |
|---|---|
| Training | 1M steps, ~48 min on an AMD Radeon AI PRO R9700 (ROCm) |
| Hub evaluation (10 episodes) | mean reward 544.5 ± 184.6 (score 359.9, certification needs ≥ 200) |

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install git+https://github.com/DLR-RM/rl-baselines3-zoo "gymnasium[atari]" opencv-python-headless moviepy imageio-ffmpeg huggingface_hub
```

`opencv-python-headless` is preinstalled on Colab but missing elsewhere; training crashes without it.
No `xvfb`/virtual display is needed: video recording works headless.

## Run

```bash
./run.sh       # train (dqn.yml, unchanged from the course), evaluate, record a video
hf auth login  # token with write access
python -m rl_zoo3.push_to_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 \
  --repo-name dqn-SpaceInvadersNoFrameskip-v4 -orga <hf-user> -f logs/
```

`push_to_hub` silently skips the replay video with current SB3 (it reads a removed
`env.video_recorder` attribute and calls a system `ffmpeg`). Upload the video from
`logs/dqn/SpaceInvadersNoFrameskip-v4_1/videos/` as `replay.mp4` instead:

```python
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="<video>.mp4", path_in_repo="replay.mp4",
                    repo_id="<hf-user>/dqn-SpaceInvadersNoFrameskip-v4")
```
