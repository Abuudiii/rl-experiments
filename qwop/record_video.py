"""Record a trained QWOP model to mp4 without a display.

Env settings come from metadata.yml in the model's run directory, which the trainer writes only
when training finishes. For checkpoints of a running (or killed) run, pass the training config.

usage: python record_video.py <model.zip> [out.mp4] [max_seconds] [train_config.yml]
"""
import os
import sys
import zipfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # off-screen pygame, no display needed
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import imageio_ffmpeg
import pygame
import yaml
from sb3_contrib import QRDQN
from stable_baselines3 import DQN

from qwop_python import QWOPEnv

FPS = 30  # QWOP advances 1/30 s of game time per frame

model_file = sys.argv[1]
out_file = sys.argv[2] if len(sys.argv) > 2 else "qwop.mp4"
max_frames = int(float(sys.argv[3]) * FPS) if len(sys.argv) > 3 else 120 * FPS

meta_file = os.path.join(os.path.dirname(model_file), "metadata.yml")
if os.path.exists(meta_file):
    env_kwargs = yaml.safe_load(open(meta_file))["env"]
elif len(sys.argv) > 4:
    env_kwargs = yaml.safe_load(open(sys.argv[4]))["env_kwargs"]
    include = env_kwargs.pop("__include__", None)  # same expansion as common.expand_env_kwargs
    if include:
        env_kwargs = yaml.safe_load(open(include)) | env_kwargs
else:
    sys.exit(f"{meta_file} not found (run still training?): pass the training config as 4th argument")

frames_per_action = env_kwargs["frames_per_step"]  # replay each action for as many frames as in training
env_kwargs.update(frames_per_step=1, render_mode="human")
env = QWOPEnv(**env_kwargs)
with zipfile.ZipFile(model_file) as z:  # the saved policy class tells DQN and QR-DQN apart
    algo = QRDQN if b"sb3_contrib.qrdqn" in z.read("data") else DQN
model = algo.load(model_file, device="cpu")

obs, _ = env.reset()
frames, done = [], False
while not done and len(frames) < max_frames:
    action, _ = model.predict(obs, deterministic=True)
    for _ in range(frames_per_action):
        obs, _, terminated, truncated, info = env.step(action)
        env.render()
        frames.append(pygame.surfarray.array3d(env._screen).swapaxes(0, 1))
        done = terminated or truncated
        if done:
            break
env.close()

h, w, _ = frames[0].shape
writer = imageio_ffmpeg.write_frames(out_file, (w, h), fps=FPS, macro_block_size=1)
writer.send(None)  # start the generator
for f in frames:
    writer.send(f.copy(order="C"))
writer.close()
print(f"{out_file}: {len(frames)} frames ({len(frames) / FPS:.1f}s), last info: "
      f"distance={info.get('distance')}, time={info.get('time')}, success={info.get('is_success')}")
