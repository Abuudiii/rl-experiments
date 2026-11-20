# RL stack notes (TRL GRPOTrainer + Harbor, one H100)

Everything below was run on job 68480301 (rocm-ccs-aus-d03-08, 1x H100 80GB) on 2026-11-20.

## Versions (`/home/AMD/absheikh/pr-description-eval/.venv-rl`, Python 3.12.3)

| package | version |
|---|---|
| trl | 1.18.2 (`trl[harbor,vllm]`) |
| vllm | 0.33.0 |
| transformers | 5.20.2 |
| torch | 2.14.0+cu132 (driver reports CUDA 13.1; worked) |
| harbor | 0.25.0 (pulled in by `trl[harbor]`; PyPI latest is 0.26.0) |

Install command that worked:

```bash
export UV_CACHE_DIR=/var/tmp/absheikh-prdesc/uv-cache UV_LINK_MODE=copy
uv venv -p 3.12 .venv-rl
uv pip install --no-config --index-url https://pypi.org/simple -p .venv-rl/bin/python \
  "trl[harbor,vllm]==1.18.2" vllm==0.33.0 "transformers>=5.2" --torch-backend=auto
```

## Incompatibilities and workarounds

1. **uv on the H100 node hangs on resolve.** `/etc/uv/uv.toml` forces an internal mirror
   (`smci250-ccs-aus-c14-43...:5000/index`) with `no-cache=true`; a resolve of trl+vllm sat for >9 min with no
   download. Workaround: `--no-config --index-url https://pypi.org/simple` (PyPI is reachable), cache on `/var/tmp`.
   Install then took ~4.5 min.
2. **vllm / transformers pins.** TRL's Harbor path needs `transformers>=5.2` (environment_factory); TRL's `vllm`
   extra pins `vllm<0.34`; vllm 0.33.0 pins `transformers>=5.16.1,<5.21` and `torch==2.14.0`. Pinning
   `vllm==0.33.0` resolves all three (transformers 5.20.2). No source builds needed.
3. **`docker compose` missing on the node.** Harbor's docker backend shells out to `docker compose`. The plugin was
   installed user-level at `/home/AMD/absheikh/.docker/cli-plugins/docker-compose` (v2.32.4) by the TaskFamily
   agent; nothing else needed.
4. **No `cpus` / `memory_mb` in task.toml.** Harbor turns them into compose resource limits, and containers on this
   node start inside the Slurm step's cgroup, where limits fail ("cannot enter cgroupv2 ... threaded mode"). Our
   task.toml sets neither. Side effect: sandboxes are not CPU-pinned (Harbor has no cpuset option); only the trainer
   process is pinned with `taskset`.
5. **TRL's built-in `HarborBashEnv` prompt does not fit this task.** Its prompt suffix says data is in
   `/home/user/input` and that the answer must be written to `/workdir/answer.txt`. Custom harness
   `rl/prdesc_env.py:PrDescBashEnv` exposes the same single `bash` tool with a task-neutral suffix ("when finished,
   reply without a tool call").
6. **`HarborEnv._start` re-uploads every Dockerfile `COPY` source** (an E2B workaround). On docker this overwrote the
   fake `gh` (mode taken from the host file) and re-created `/opt/task-setup` (deleted at image build). The harness'
   `_setup` restores `chmod 755 /usr/local/bin/gh` and removes `/opt/task-setup`, `/workdir`, `/home/user/input`.
7. **Sandbox lifecycle is serial.** HarborEnv starts/stops one sandbox at a time, and TRL executes tool calls
   rollout by rollout. Measured per sandbox: start ~3-5 s (image cached), verifier ~2.5-2.8 s, `compose down` ~12.5 s.
8. **Qwen3.5 linear-attention layers train on slow fallbacks.** transformers warns that `causal_conv1d` and
   `flash-linear-attention` are not installed, so `causal_conv1d_fn` and `chunk_gated_delta_rule` use reference
   PyTorch code during the policy forward/backward. Not installed here (would need CUDA builds); installing them is
   the first speed-up to try.
9. **Qwen3.5 checkpoints are `Qwen3_5ForConditionalGeneration` (vision+text).** For serving we pass
   `--language-model-only`; GRPOTrainer loaded the checkpoint from the model id without changes.

## Harbor task (`rl/harbor_tasks/logslice/`)

- `environment/`: the original `task/environment` (Dockerfile defaults to `CONTRIBUTING=0`, i.e. the `base`
  variant). `task.toml`: `network_mode = "no-network"`, `workdir = "/repo"`, shared verifier, agent timeout 1800 s.
- `tests/test.sh` -> `tests/verify.py`, which imports the original `grade.py` (verbatim copy; fixtures for the PR
  branch's tests and the original PR body under `tests/fixtures/`) and writes `/logs/verifier/reward.json`:
  `reward = 0 if not tests_untouched else 0.5 * hidden_test_pass_fraction + 0.5 * mean(desc_no_stale_claims,
  desc_mentions_new_behavior)`, plus each grader as 0/1. Differences from `grade.py`: hidden tests run inside the
  agent's container against `/repo` (grade.py ran them in a fresh container on the extracted repo), and the test
  result is a pass fraction (junit XML) instead of all-or-nothing.
- `solution/solve.sh`: the original oracle.

Validation (docker backend, on the node):

| rollout | path | reward | components |
|---|---|---|---|
| oracle | `harbor run -a oracle` and `rl/probe_env.py` (TRL HarborEnv path) | 1.0 / 1.0 | 18/18 tests, desc both true |
| code only (`solve.sh --code-only`) | `rl/probe_env.py` | 0.5 | 18/18 tests, description untouched |
| no-op | `harbor run -a nop` and `rl/probe_env.py` | 0.1944 / 0.1944 | 7/18 hidden tests already pass on the PR branch |

## GRPO smoke test (`rl/grpo_smoke.py`)

Qwen/Qwen3.5-2B, bf16 weights, `num_generations=4`, `per_device_train_batch_size=1`,
`gradient_accumulation_steps=4`, `max_completion_length=12288`, `vllm_max_model_length=16384`,
`max_tool_calling_iterations=30`, thinking on, vLLM colocate at `gpu_memory_utilization=0.3`, gradient checkpointing.

- Step 1 completed: 4 rollouts ran in 4 docker sandboxes (13, 0, 0, 29 bash calls), the verifier rewards came back
  (all 0.1944), and the optimizer step ran. Step time 77.3 s, torch peak allocated 36.7 GB (reserved 37.8 GB);
  nvidia-smi showed ~37-40 GB used in total.
- Because all 4 rewards were equal, `reward_std = 0`, `loss = 0`, `grad_norm = 0`: no learning signal for 2B on this
  task (its 10 baseline trials also score 0.194 in 9/10 cases).
- Step 2 was still generating after 6.5 min and was stopped to free the GPU for the 4B run.

Logs: `rl/runs/smoke1.log`, `rl/runs/qwen35-2b-smoke1/{env_timings.jsonl,trainer/steps.jsonl}`, `rl/runs/smoke1_gpu.csv`.

## Qwen3.5-4B GRPO pilot (3 steps)

Same script, `--model Qwen/Qwen3.5-4B --thinking --max-steps 3 --max-tool-calling-iterations 25
--max-completion-length 8192 --vllm-max-model-length 12288 --learning-rate 1e-5 --save`, bf16 weights.

- Steps: 110 s, 164 s, 170 s (445 s total train time + ~2.5 min startup). torch peak allocated 61.5 GB (reserved
  63.6 GB); nvidia-smi peak ~66 GB of 80 GB. Nonzero reward std in all 3 steps (0.10, 0.16, 0.03), so real gradient
  updates (grad_norm 1.38, 0.26, 1.18).
- 50-100% of completions hit the 8192-token cap; tool calls per rollout 6-23.
- Rough budget at these settings: ~2.5-3 min/step (4 rollouts) -> ~20-24 steps/hour on one H100.
- Saved model: `rl/runs/qwen35-4b-grpo/final` (served by vLLM 0.33.0 from `.venv-rl` with `--language-model-only`).
- A watcher that waited for `SAVED` in the log by grepping for `Traceback` exited early: `log_completions` prints tool
  outputs (including Python tracebacks from the sandbox) into the training log. Grep for exact markers only.
