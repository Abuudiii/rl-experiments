# SoccerTwos — MA-POCA self-play
Hugging Face [Deep RL course, unit 7](https://huggingface.co/learn/deep-rl-course/unit7/introduction):
2v2 soccer trained with ML-Agents' MA-POCA trainer and self-play. `SoccerTwos.yaml` is the course config (5M steps).

The course's [AI vs. AI leaderboard](https://huggingface.co/spaces/huggingface-projects/AIvsAI-SoccerTwos) is
unmaintained (the Space was in `RUNTIME_ERROR` at setup time), but pushing a model still passes the unit, and the
[SoccerTwos demo](https://huggingface.co/spaces/unity/ML-Agents-SoccerTwos) still plays it in the browser.

**Model:** https://huggingface.co/Abuudiii/poca-SoccerTwos
## Result
| | |
|---|---|
| Training | 5M steps, ~1h45 on an NVIDIA RTX PRO 4000 Blackwell (stopped at 500k, continued with `--resume`) |
| Self-play ELO | 1200 → 1167; stayed flat-to-slightly-down the whole run |

## Setup
Reuses the ML-Agents clone from [`../mlagents`](../mlagents/); this venv has CUDA torch (≤ 2.8 for ML-Agents).
```bash
uv venv -p 3.10.12 .venv && . .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" -e ../mlagents/ml-agents/ml-agents-envs -e ../mlagents/ml-agents/ml-agents gdown "setuptools<81"
# Linux build from the course (Google Drive); REQUESTS_CA_BUNDLE is only needed behind a TLS-inspecting proxy
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt gdown 1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL -O SoccerTwos.zip
mkdir -p training-envs-executables/SoccerTwos && unzip SoccerTwos.zip -d training-envs-executables/SoccerTwos
chmod -R 755 training-envs-executables/SoccerTwos
```

## Run
```bash
mlagents-learn SoccerTwos.yaml --env=training-envs-executables/SoccerTwos/SoccerTwos \
  --run-id=SoccerTwos1 --no-graphics --torch-device cuda
hf auth login
mlagents-push-to-hf --run-id=SoccerTwos1 --local-dir=results/SoccerTwos1 \
  --repo-id=<hf-user>/poca-SoccerTwos --commit-message="First push"
```
Press Ctrl-C **once** to stop early: ML-Agents writes the final `.onnx` before exiting. `--resume` continues a run.
The model card needs the `ML-Agents-SoccerTwos` tag and a `SoccerTwos.onnx` file for the challenge.
