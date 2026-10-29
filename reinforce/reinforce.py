"""REINFORCE on CartPole-v1 and Pixelcopter-PLE-v0 (HF Deep RL course, unit 4).

Train:  python reinforce.py cartpole            -> logs/CartPole-v1.pt, logs/CartPole-v1.mp4
Push:   python reinforce.py cartpole --push <hf-user>/Reinforce-CartPole-v1
"""
import argparse
import datetime
import json
import os
import tempfile
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame, must precede pygame init
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # no sound card on GPU nodes

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0")
LOGS = Path(__file__).parent / "logs"


class Pixelcopter(gym.Env):
    """Gymnasium-API port of gym_pygame's Pixelcopter-PLE-v0 (gym_pygame needs the dead `gym` package)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self):
        from ple import PLE
        from ple.games.pixelcopter import Pixelcopter as Game

        self.game = Game()
        self.ple = PLE(self.game, fps=30, display_screen=False)
        self.ple.init()
        self.actions = self.ple.getActionSet()
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(self.game.getGameState()),), np.float32)

    def _obs(self):
        return np.array(list(self.game.getGameState().values()), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.ple.reset_game()
        return self._obs(), {}

    def step(self, action):
        reward = self.ple.act(self.actions[action])
        return self._obs(), reward, self.ple.game_over(), False, {}

    def render(self):
        return np.fliplr(np.rot90(self.ple.getScreenRGB(), 3))


ENVS = {
    "cartpole": dict(make=lambda: gym.make("CartPole-v1", render_mode="rgb_array"), h={
        "env_id": "CartPole-v1", "h_size": 16, "n_training_episodes": 1000, "n_evaluation_episodes": 10,
        "max_t": 1000, "gamma": 1.0, "lr": 1e-2}),
    "pixelcopter": dict(make=Pixelcopter, h={
        "env_id": "Pixelcopter-PLE-v0", "h_size": 64, "n_training_episodes": 50000, "n_evaluation_episodes": 10,
        "max_t": 10000, "gamma": 0.99, "lr": 1e-4}),
}


class Policy(nn.Module):
    """The notebook's two policies: 1 hidden layer for CartPole, 2 for Pixelcopter."""

    def __init__(self, s_size, a_size, h_size, deep):
        super().__init__()
        layers = [nn.Linear(s_size, h_size), nn.ReLU()]
        if deep:
            layers += [nn.Linear(h_size, h_size * 2), nn.ReLU()]
        layers.append(nn.Linear(layers[-2].out_features, a_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        m = Categorical(self.forward(state).cpu())
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    eps = np.finfo(np.float32).eps.item()
    for i_episode in range(1, n_training_episodes + 1):
        log_probs, rewards = [], []
        state, _ = env.reset()
        for _ in range(max_t):
            action, log_prob = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if terminated or truncated:
                break
        scores_deque.append(sum(rewards))

        returns = deque(maxlen=max_t)  # G_t = r_t + gamma * G_{t+1}, built back to front
        for r in reversed(rewards):
            returns.appendleft(r + gamma * (returns[0] if returns else 0))
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        loss = (-torch.cat(log_probs) * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}", flush=True)


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    episode_rewards = []
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        total = 0
        for _ in range(max_steps):
            action, _ = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        episode_rewards.append(total)
    return np.mean(episode_rewards), np.std(episode_rewards)


def record_video(env, policy, out, max_steps, fps=30):
    state, _ = env.reset()
    images = [env.render()]
    for _ in range(max_steps):
        action, _ = policy.act(state)
        state, _, terminated, truncated, _ = env.step(action)
        images.append(env.render())
        if terminated or truncated:
            break
    imageio.mimsave(out, images, fps=fps)


def push_to_hub(repo_id, model, h, env, video_fps=30):
    from huggingface_hub import HfApi
    from huggingface_hub.repocard import metadata_eval_result, metadata_save

    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        torch.save(model, d / "model.pt")
        (d / "hyperparameters.json").write_text(json.dumps(h))
        mean_reward, std_reward = evaluate_agent(env, h["max_t"], h["n_evaluation_episodes"], model)
        (d / "results.json").write_text(json.dumps({
            "env_id": h["env_id"], "mean_reward": mean_reward,
            "n_evaluation_episodes": h["n_evaluation_episodes"],
            "eval_datetime": datetime.datetime.now().isoformat()}))
        env_id = h["env_id"]
        readme = d / "README.md"
        readme.write_text(
            f"# **Reinforce** Agent playing **{env_id}**\n"
            f"This is a trained model of a **Reinforce** agent playing **{env_id}** .\n"
            "To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: "
            "https://huggingface.co/deep-rl-course/unit4/introduction\n")
        metadata_save(readme, {
            "tags": [env_id, "reinforce", "reinforcement-learning", "custom-implementation", "deep-rl-class"],
            **metadata_eval_result(
                model_pretty_name=repo_id.split("/")[1], task_pretty_name="reinforcement-learning",
                task_id="reinforcement-learning", metrics_pretty_name="mean_reward", metrics_id="mean_reward",
                metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
                dataset_pretty_name=env_id, dataset_id=env_id)})
        record_video(env, model, d / "replay.mp4", h["max_t"], video_fps)
        api.upload_folder(repo_id=repo_id, folder_path=d, path_in_repo=".")
    print(f"mean_reward {mean_reward:.2f} +/- {std_reward:.2f}; pushed to {repo_url}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("env", choices=ENVS)
    p.add_argument("--push", metavar="HF_USER/REPO", help="push the saved model instead of training")
    p.add_argument("--episodes", type=int, help="override n_training_episodes")
    args = p.parse_args()

    cfg = ENVS[args.env]
    h, env = dict(cfg["h"]), cfg["make"]()
    h["state_space"], h["action_space"] = env.observation_space.shape[0], int(env.action_space.n)
    if args.episodes:
        h["n_training_episodes"] = args.episodes
    path = LOGS / f"{h['env_id']}.pt"
    print(f"device: {device}")

    if args.push:
        push_to_hub(args.push, torch.load(path, weights_only=False, map_location=device), h, env)
        return

    policy = Policy(h["state_space"], h["action_space"], h["h_size"], deep=args.env == "pixelcopter").to(device)
    optimizer = optim.Adam(policy.parameters(), lr=h["lr"])
    reinforce(env, policy, optimizer, h["n_training_episodes"], h["max_t"], h["gamma"],
              100 if args.env == "cartpole" else 1000)
    LOGS.mkdir(exist_ok=True)
    torch.save(policy, path)
    mean, std = evaluate_agent(env, h["max_t"], h["n_evaluation_episodes"], policy)
    print(f"eval over {h['n_evaluation_episodes']} episodes: {mean:.2f} +/- {std:.2f} (score {mean - std:.2f})")
    record_video(env, policy, LOGS / f"{h['env_id']}.mp4", h["max_t"])
    print(f"saved {path} and {path.with_suffix('.mp4')}")


if __name__ == "__main__":
    main()
