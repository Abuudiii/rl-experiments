# ML-Agents — SnowballTarget and Pyramids
Hugging Face [Deep RL course, unit 5](https://huggingface.co/learn/deep-rl-course/unit5/introduction):
PPO agents trained with Unity [ML-Agents](https://github.com/Unity-Technologies/ml-agents) on two Unity games.
`SnowballTarget.yaml` is the notebook's config; `PyramidsRND.yaml` is upstream's with `max_steps` set to 2M (the notebook says 1M, which wasn't enough here).

**Models:** [SnowballTarget](https://huggingface.co/Abuudiii/ppo-SnowballTarget), [Pyramids](https://huggingface.co/Abuudiii/ppo-Pyramids)
## Result
| | |
|---|---|
| SnowballTarget | 200k steps, ~9 min on an AMD Instinct MI100 (ROCm); mean reward 4.3 → 26.0 ± 2.3 |
| Pyramids | 2M steps (1M, then `--resume` to 2M), ~83 min on the MI100; reward stays near −1 until ~700k, 0.9 at 1M, ~1.67 at 2M (course target 1.75) |

## Setup
ML-Agents needs Python 3.10 (≤ 3.10.12) and torch ≤ 2.8.
```bash
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents
uv venv -p 3.10.12 .venv && . .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/rocm6.4 \
  "torch==2.8.0+rocm6.4" -e ./ml-agents/ml-agents-envs -e ./ml-agents/ml-agents
mkdir -p training-envs-executables/linux && cd training-envs-executables/linux
curl -LO https://github.com/huggingface/Snowball-Target/raw/main/SnowballTarget.zip
curl -LO https://huggingface.co/spaces/unity/ML-Agents-Pyramids/resolve/main/Pyramids.zip
unzip SnowballTarget.zip && unzip Pyramids.zip && chmod -R 755 SnowballTarget Pyramids && cd -
```

## Run
```bash
mlagents-learn SnowballTarget.yaml --env=training-envs-executables/linux/SnowballTarget/SnowballTarget \
  --run-id=SnowballTarget1 --no-graphics --torch-device cuda
mlagents-learn PyramidsRND.yaml --env=training-envs-executables/linux/Pyramids/Pyramids \
  --run-id=Pyramids1 --no-graphics --torch-device cuda
hf auth login
mlagents-push-to-hf --run-id=SnowballTarget1 --local-dir=results/SnowballTarget1 \
  --repo-id=<hf-user>/ppo-SnowballTarget --commit-message="First push"
```
Results and checkpoints go to `results/<run-id>/`. Add `--resume` to continue a stopped run, `--force` to overwrite one.
