# PPO — Super Mario Bros 1-1
Goal: clear world 1-1. Same PPO as [`../ppo-lunarlander`](../ppo-lunarlander/) (the course's CleanRL script), with
the changes CleanRL's `ppo_atari.py` makes for learning from pixels (see below).

**Model:** https://huggingface.co/Abuudiii/ppo-SuperMarioBros-1-1
## Result
5M steps, ~2h45 on an NVIDIA RTX PRO 4000 Blackwell (~510 steps/s). **Cleared 1-1 in 9/10 evaluation runs** (reward 2960 ± 343).

| Steps | Avg distance (`x_pos`, flag ≈ 3160) | Training episodes cleared |
|---|---|---|
| 0–1M | 867 | 0.2% |
| 1–2M | 1,241 | 1.6% |
| 2–3M | 2,200 | 25% |
| 3–4M | 2,793 | 64% |
| 4–5M | 2,977 | 84% |

## How it differs from LunarLander

Same PPO loop; the changes are for learning from pixels. Actor and critic share the CNN, so both losses update it
(`vf-coef` balances them).

| | LunarLander | Mario |
|---|---|---|
| Input | 8 numbers | game screen: grayscale, 84×84, last 4 frames stacked |
| Network | two small MLPs (actor, critic) | one CNN shared by actor and critic, plus a head for each |
| Actions | 4 | 7 (`SIMPLE_MOVEMENT`: right, right+jump, run, …) each held for 4 frames |
| Clip | 0.2 | 0.1 |
| Games in parallel | 4 | 16, one process each (the NES emulator runs on the CPU) |

The game's reward (progress to the right, death penalty, flag bonus) is clipped to ±15 by the environment;
it's divided by 15 for training. Episode = one life on 1-1.

## Setup
`gym-super-mario-bros` 9.x needs Python 3.13.
```bash
uv venv -p 3.13 .venv && . .venv/bin/activate
uv pip install --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
  torch gym-super-mario-bros "gymnasium[other]" opencv-python-headless tensorboard huggingface_hub \
  imageio imageio-ffmpeg wasabi "setuptools<81"
```

## Run
```bash
python ppo.py --repo-id <hf-user>/ppo-SuperMarioBros-1-1   # 5M steps on the GPU, then evaluate + push
grep cleared=True train.log | wc -l                          # how many episodes reached the flag
```
Each finished episode prints its return, how far right Mario got (`x_pos`, the flag is at ~3160) and `cleared`.
A checkpoint is saved to `runs/<run>/model.pt` every 50 updates (~100k steps). If a run is killed, push the
checkpoint with `python ppo.py --load-model runs/<run>/model.pt --repo-id <hf-user>/ppo-SuperMarioBros-1-1`.
The push evaluates 10 episodes (reports how many cleared the level) and uploads a `replay.mp4`: the first of up
to 5 recorded attempts that reaches the flag, or else the one that got furthest.
