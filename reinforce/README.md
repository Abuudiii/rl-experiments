# REINFORCE — CartPole and Pixelcopter
Hugging Face [Deep RL course, unit 4](https://huggingface.co/learn/deep-rl-course/unit4/introduction):
a policy-gradient (REINFORCE) agent written in PyTorch, trained on `CartPole-v1` and `Pixelcopter-PLE-v0`.
Hyperparameters and network shapes are the notebook's.

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install gymnasium huggingface_hub imageio imageio-ffmpeg pygame \
  git+https://github.com/ntasfi/PyGame-Learning-Environment.git
```
The notebook's `gym` + `gym_pygame` stack doesn't install on Python 3.12, so `reinforce.py` uses
`gymnasium` and wraps PLE's Pixelcopter directly (same observations, actions and rewards).
No `xvfb`/virtual display is needed: pygame runs with `SDL_VIDEODRIVER=dummy`.

## Run
```bash
python reinforce.py cartpole       # train, evaluate, save logs/CartPole-v1.{pt,mp4}
python reinforce.py pixelcopter    # 50k episodes; run it in tmux
hf auth login                      # token with write access
python reinforce.py cartpole --push <hf-user>/Reinforce-CartPole-v1
python reinforce.py pixelcopter --push <hf-user>/Reinforce-Pixelcopter-PLE-v0
```
`--push` loads the saved model, evaluates it, records `replay.mp4` and uploads everything with the model card.
Training runs on the GPU (`cuda:0`, i.e. the ROCm device). `--episodes N` shortens training.
