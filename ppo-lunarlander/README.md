# PPO — LunarLander, from scratch
Hugging Face [Deep RL course, unit 8 part 1](https://huggingface.co/learn/deep-rl-course/unit8/introduction):
the notebook's CleanRL `ppo.py` (PPO written in plain PyTorch) on LunarLander.
The algorithm, network and hyperparameters are the notebook's.

**Model:** https://huggingface.co/Abuudiii/ppo-LunarLander-v3
## Result
| | |
|---|---|
| Training | 1M steps with the notebook's hyperparameters, ~2,300 steps/s on an NVIDIA RTX PRO 4000 Blackwell (~7 min) |
| Evaluation (10 episodes) | −80.1 ± 72.6; not landing yet (200 = solved) |

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch "gymnasium[box2d]" tensorboard huggingface_hub imageio imageio-ffmpeg wasabi
```
The notebook pins `gym==0.22`, which doesn't install on Python 3.12 and needs `xvfb` to render, so `ppo.py`
is ported to `gymnasium`. Gymnasium 1.x removed `LunarLander-v2`; `LunarLander-v3` is the same task with bug fixes.
No `xvfb`, `swig` or system `ffmpeg` is needed.

## Run
```bash
hf auth login   # once, token with write access
python ppo.py --total-timesteps 1000000 --repo-id <hf-user>/ppo-LunarLander-v3
tensorboard --logdir runs
```
Training runs on the GPU and saves `runs/<run>/model.pt`. With `--repo-id` it then evaluates 10 episodes,
records `replay.mp4` and pushes the model, model card and TensorBoard logs. The notebook's default is 50k steps,
which is too short to land; the course has no minimum score, only a pushed model.
