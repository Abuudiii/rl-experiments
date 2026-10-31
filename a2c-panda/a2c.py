"""A2C on panda-gym robot-arm tasks (HF Deep RL course, unit 6).

Train:  python a2c.py PandaReachDense-v3            -> logs/PandaReachDense-v3/{model.zip,vec_normalize.pkl,replay.mp4}
Push:   python a2c.py PandaReachDense-v3 --push <hf-user>
"""
import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import panda_gym  # noqa: F401  registers the Panda envs
from huggingface_hub import HfApi
from huggingface_sb3 import package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_eval_env(env_id, stats):
    """Single env with the training normalisation stats, frozen, and real (unnormalised) rewards."""
    env = VecNormalize.load(stats, DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")]))
    env.training = False
    env.norm_reward = False
    return env


def record_video(model, env, path, steps=200):
    obs = env.reset()
    frames = [env.render()]
    for _ in range(steps):
        obs, *_ = env.step(model.predict(obs, deterministic=True)[0])
        frames.append(env.render())
    imageio.mimsave(path, frames, fps=25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("env", nargs="?", default="PandaReachDense-v3")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--push", metavar="HF_USER", help="push the saved model instead of training")
    args = p.parse_args()

    out = Path(__file__).parent / "logs" / args.env
    model_path, stats = out / "model.zip", out / "vec_normalize.pkl"

    if not args.push:
        env = VecNormalize(make_vec_env(args.env, n_envs=4), norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = A2C("MultiInputPolicy", env, verbose=1, device="cuda")
        model.learn(args.steps)
        out.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        env.save(stats)

    model = A2C.load(model_path, device="cuda")
    eval_env = make_eval_env(args.env, stats)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"eval over 10 episodes: {mean:.2f} +/- {std:.2f} (score {mean - std:.2f})")
    record_video(model, eval_env, out / "replay.mp4")
    print(f"saved {model_path}, {stats}, {out / 'replay.mp4'}")

    if args.push:
        repo_id = f"{args.push}/a2c-{args.env}"
        package_to_hub(model=model, model_name=f"a2c-{args.env}", model_architecture="A2C", env_id=args.env,
                       eval_env=eval_env, repo_id=repo_id, commit_message="Initial commit")
        # package_to_hub's own replay needs a system ffmpeg and a removed SB3 attribute, so it skips the video
        HfApi().upload_file(path_or_fileobj=out / "replay.mp4", path_in_repo="replay.mp4", repo_id=repo_id)
        print(f"pushed https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
