"""Record a trained QWOP model to mp4 without a display.

usage: python record_video.py <model.zip> [out.mp4] [max_seconds]
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # off-screen pygame, no display needed
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import imageio_ffmpeg
import pygame
import yaml
from stable_baselines3 import DQN

from qwop_python import QWOPEnv

FRAMES_PER_ACTION = 4  # must match frames_per_step used in training (config/train_dqn.yml)
FPS = 30  # QWOP advances 1/30 s of game time per frame

model_file = sys.argv[1]
out_file = sys.argv[2] if len(sys.argv) > 2 else "qwop.mp4"
max_frames = int(float(sys.argv[3]) * FPS) if len(sys.argv) > 3 else 120 * FPS

env_kwargs = yaml.safe_load(open("config/env.yml"))
env_kwargs.update(frames_per_step=1, render_mode="human")
env = QWOPEnv(**env_kwargs)
model = DQN.load(model_file, device="cpu")

obs, _ = env.reset()
frames, done = [], False
while not done and len(frames) < max_frames:
    action, _ = model.predict(obs, deterministic=True)
    for _ in range(FRAMES_PER_ACTION):
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
