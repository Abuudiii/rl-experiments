# rl-experiments

Reinforcement learning experiments, one per directory. Most were trained on AMD GPUs with ROCm; `ppo-mario/`,
`grpo-rlhf-qwen/` and `fpv-racing/` on NVIDIA GPUs.

| Directory | What | Model |
|---|---|---|
| [`space-invaders-dqn/`](space-invaders-dqn/) | DQN on Atari Space Invaders (HF Deep RL course, unit 3) | [hub](https://huggingface.co/Abuudiii/dqn-SpaceInvadersNoFrameskip-v4) |
| [`qwop/`](qwop/) | DQN and QR-DQN learning to run in QWOP; QR-DQN finishes the 100 m | [QR-DQN](https://huggingface.co/Abuudiii/qrdqn-QWOP), [DQN](https://huggingface.co/Abuudiii/dqn-QWOP) |
| [`reinforce/`](reinforce/) | REINFORCE (policy gradient) on CartPole and Pixelcopter (HF Deep RL course, unit 4) | [CartPole](https://huggingface.co/Abuudiii/Reinforce-CartPole-v1), [Pixelcopter](https://huggingface.co/Abuudiii/Reinforce-Pixelcopter-PLE-v0) |
| [`mlagents/`](mlagents/) | PPO with Unity ML-Agents on SnowballTarget and Pyramids (HF Deep RL course, unit 5) | [SnowballTarget](https://huggingface.co/Abuudiii/ppo-SnowballTarget), [Pyramids](https://huggingface.co/Abuudiii/ppo-Pyramids) |
| [`a2c-panda/`](a2c-panda/) | A2C on the panda-gym robot arm, PandaReachDense (HF Deep RL course, unit 6) | [hub](https://huggingface.co/Abuudiii/a2c-PandaReachDense-v3) |
| [`soccertwos/`](soccertwos/) | 2v2 soccer with MA-POCA self-play in ML-Agents (HF Deep RL course, unit 7) | [hub](https://huggingface.co/Abuudiii/poca-SoccerTwos) |
| [`ppo-lunarlander/`](ppo-lunarlander/) | PPO written from scratch (CleanRL) on LunarLander (HF Deep RL course, unit 8 part 1) | [hub](https://huggingface.co/Abuudiii/ppo-LunarLander-v3) |
| [`ppo-mario/`](ppo-mario/) | PPO from pixels (CNN) on Super Mario Bros 1-1: clears the level 9/10 | [hub](https://huggingface.co/Abuudiii/ppo-SuperMarioBros-1-1) |
| [`grpo-rlhf-qwen/`](grpo-rlhf-qwen/) | RLHF on Qwen2.5-1.5B: reward models from Anthropic's HH-RLHF human votes, then GRPO (TRL) rewarding helpful only, harmless only, or both | [helpful](https://huggingface.co/Abuudiii/grpo-Qwen2.5-1.5B-hh-rlhf-helpful), [harmless](https://huggingface.co/Abuudiii/grpo-Qwen2.5-1.5B-hh-rlhf-harmless), [combined](https://huggingface.co/Abuudiii/grpo-Qwen2.5-1.5B-hh-rlhf-combined) |
| [`fpv-racing/`](fpv-racing/) | FPV drone gate racing in simulation: a PPO teacher flying from state (Crazyflow), then a camera-only CNN student trained by imitation (DAgger) on Unreal frames; the student completes three laps in 95.8% of held-out episodes | [hub](https://huggingface.co/Abuudiii/fpv-gate-racing) |
