# rl-experiments

Reinforcement learning experiments, one per directory. Trained on AMD GPUs with ROCm.

| Directory | What | Model |
|---|---|---|
| [`space-invaders-dqn/`](space-invaders-dqn/) | DQN on Atari Space Invaders (HF Deep RL course, unit 3) | [hub](https://huggingface.co/Abuudiii/dqn-SpaceInvadersNoFrameskip-v4) |
| [`qwop/`](qwop/) | DQN and QR-DQN learning to run in QWOP; QR-DQN finishes the 100 m | [QR-DQN](https://huggingface.co/Abuudiii/qrdqn-QWOP), [DQN](https://huggingface.co/Abuudiii/dqn-QWOP) |
| [`reinforce/`](reinforce/) | REINFORCE (policy gradient) on CartPole and Pixelcopter (HF Deep RL course, unit 4) | [CartPole](https://huggingface.co/Abuudiii/Reinforce-CartPole-v1), [Pixelcopter](https://huggingface.co/Abuudiii/Reinforce-Pixelcopter-PLE-v0) |
| [`mlagents/`](mlagents/) | PPO with Unity ML-Agents on SnowballTarget and Pyramids (HF Deep RL course, unit 5) | [SnowballTarget](https://huggingface.co/Abuudiii/ppo-SnowballTarget), [Pyramids](https://huggingface.co/Abuudiii/ppo-Pyramids) |
| [`a2c-panda/`](a2c-panda/) | A2C on the panda-gym robot arm, PandaReachDense (HF Deep RL course, unit 6) | [hub](https://huggingface.co/Abuudiii/a2c-PandaReachDense-v3) |
