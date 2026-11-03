# PPO on Super Mario Bros 1-1. Same PPO as ../ppo-lunarlander/ppo.py (the unit 8 CleanRL script), with the
# changes CleanRL's ppo_atari.py makes for pixel input: skip 4 frames per action, grayscale 84x84, stack 4
# frames, a CNN shared by actor and critic, clip 0.1. Actions are Mario's 7-button SIMPLE_MOVEMENT set.

import argparse
import datetime
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import gym_super_mario_bros  # noqa: F401  registers the Mario envs
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from nes_py.wrappers import JoypadSpace
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from wasabi import Printer

msg = Printer()


def strtobool(x):  # distutils.util.strtobool is gone in Python 3.12
    return x.lower() in ("y", "yes", "t", "true", "on", "1")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).removesuffix(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="SuperMarioBros-1-1-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    # Hugging Face: push after training if set
    parser.add_argument("--repo-id", type=str, default=None,
        help="id of the model repository from the Hugging Face Hub {username/repo_name}")
    parser.add_argument("--load-model", type=str, default=None,
        help="skip training: load this model.pt (e.g. a checkpoint) and only evaluate/push it")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def package_to_hub(repo_id, model, hyperparameters, eval_env, video_fps=30, commit_message="Push agent to the Hub", token=None, logs=None):
    """Evaluate, record a replay video, write the model card and push everything to the Hub."""
    repo_url = HfApi().create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        torch.save(model.state_dict(), tmpdirname / "model.pt")

        mean_reward, std_reward, clears = _evaluate_agent(eval_env, 10, model)
        evaluate_data = {
            "env_id": hyperparameters.env_id,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_evaluation_episodes": 10,
            "levels_cleared": clears,
            "eval_datetime": datetime.datetime.now().isoformat(),
        }
        with open(tmpdirname / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)

        record_video(eval_env, model, tmpdirname / "replay.mp4", video_fps)

        generated_model_card, metadata = _generate_model_card("PPO", hyperparameters.env_id, mean_reward, std_reward, hyperparameters)
        _save_model_card(tmpdirname, generated_model_card, metadata)

        if logs:
            _add_logdir(tmpdirname, Path(logs))

        msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")
        upload_folder(repo_id=repo_id, folder_path=tmpdirname, path_in_repo="", commit_message=commit_message, token=token)

    msg.info(f"mean_reward {mean_reward:.2f} +/- {std_reward:.2f}, cleared {clears}/10. Your model is pushed to the Hub: {repo_url}")
    return repo_url


def _evaluate_agent(env, n_eval_episodes, policy):
    episode_rewards, clears = [], 0
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        total_rewards_ep = 0
        while not done:
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(torch.Tensor(state).unsqueeze(0).to(device))
            state, reward, terminated, truncated, info = env.step(action.item())
            total_rewards_ep += reward
            done = terminated or truncated
        episode_rewards.append(total_rewards_ep)
        clears += bool(info["flag_get"])
    return np.mean(episode_rewards), np.std(episode_rewards), clears


def record_video(env, policy, out_directory, fps=30, attempts=5):
    """Record up to `attempts` episodes; keep the first that clears the level, else the one that got furthest."""
    best, best_x = None, -1
    for _ in range(attempts):
        state, _ = env.reset()
        images = [env.render()]
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(torch.Tensor(state).unsqueeze(0).to(device))
            state, _, terminated, truncated, info = env.step(action.item())
            images.append(env.render())
            done = terminated or truncated
        if info["flag_get"]:
            best = images
            break
        if info["x_pos"] > best_x:
            best, best_x = images, info["x_pos"]
    imageio.mimsave(out_directory, best, fps=fps)


def _generate_model_card(model_name, env_id, mean_reward, std_reward, hyperparameters):
    metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)
    converted_str = "\n".join(str(vars(hyperparameters)).split(", "))
    model_card = f"""
  # PPO Agent Playing {env_id}

  This is a trained model of a PPO agent playing {env_id}.

  # Hyperparameters
  ```python
  {converted_str}
  ```
  """
    return model_card, metadata


def generate_metadata(model_name, env_id, mean_reward, std_reward):
    metadata = {
        "tags": [env_id, "ppo", "deep-reinforcement-learning", "reinforcement-learning", "custom-implementation", "deep-rl-course"]
    }
    eval = metadata_eval_result(
        model_pretty_name=model_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_id,
        dataset_id=env_id,
    )
    return {**metadata, **eval}


def _save_model_card(local_path, generated_model_card, metadata):
    readme_path = local_path / "README.md"
    readme_path.write_text(generated_model_card, encoding="utf-8")
    metadata_save(readme_path, metadata)


def _add_logdir(local_path: Path, logdir: Path):
    if logdir.is_dir():
        shutil.copytree(logdir, local_path / "logs", dirs_exist_ok=True)


def make_env(env_id, render_mode=None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = gym.wrappers.MaxAndSkipObservation(env, skip=4)  # repeat each action for 4 frames
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        return gym.wrappers.FrameStackObservation(env, 4)  # last 4 frames, so motion is visible

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # CleanRL ppo_atari.py: one CNN body shared by actor and critic, then a small head for each
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        probs = Categorical(logits=self.actor(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda")  # GPU only: fails loudly if no GPU is visible

    # env setup. Each game runs in its own process (the NES emulator is the bottleneck). SAME_STEP autoreset:
    # a finished env returns the next episode's first observation in the same step, so storage/GAE are unchanged.
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id) for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs.single_action_space.n).to(device)
    if args.load_model:
        agent.load_state_dict(torch.load(args.load_model, map_location=device))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    envs.action_space.seed(args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = 0 if args.load_model else args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1) / 15.0  # game reward is clipped to ±15; scale to ±1
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "episode" in info:
                for i in np.flatnonzero(info["_episode"]):
                    x_pos, cleared = info["final_info"]["x_pos"][i], bool(info["final_info"]["flag_get"][i])
                    print(f"global_step={global_step}, episodic_return={info['episode']['r'][i]:.1f}, x_pos={x_pos}, cleared={cleared}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"][i], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"][i], global_step)
                    writer.add_scalar("charts/x_pos", x_pos, global_step)
                    writer.add_scalar("charts/cleared", int(cleared), global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if update % 50 == 0:  # checkpoint, so a killed run can still be pushed with --load-model
            torch.save(agent.state_dict(), f"runs/{run_name}/model.pt")

    envs.close()
    writer.close()
    torch.save(agent.state_dict(), f"runs/{run_name}/model.pt")
    print(f"saved runs/{run_name}/model.pt")

    if args.repo_id:
        package_to_hub(
            repo_id=args.repo_id,
            model=agent,
            hyperparameters=args,
            eval_env=make_env(args.env_id, render_mode="rgb_array")(),
            video_fps=15,  # one frame per action, 4 game frames each: 60 fps / 4
            logs=str(Path(args.load_model).parent) if args.load_model else f"runs/{run_name}",
        )
