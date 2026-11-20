# Pilot: GRPO on the logslice PR-description task (single task, 30-minute budget)

Status: written at the 06:21Z deadline. Everything below is what actually ran; anything not finished is marked.

## Model choice

Baselines (existing harness `harness/run_trials.py`, mini-swe-agent, n=10, `base` variant, same sampling and limits as
the Qwen3.8-27B run; Qwen3.5 served by vLLM 0.33.0 on one RTX PRO 4000 Blackwell each, thinking on). Shaped reward
R = 0 if tests_untouched is false, else 0.5 * hidden-test pass fraction + 0.5 * mean(desc_no_stale_claims,
desc_mentions_new_behavior), computed from each trial's grades.json:

| model | tests_pass | desc_updated | overall success | mean R | R values (10 trials) | mean model calls |
|---|---|---|---|---|---|---|
| Qwen3.8-27B (earlier run) | 10/10 | 6/10 | 5/10 | n/a | n/a | 22.3 |
| Qwen3.5-4B | 3/10 | 0/10 | 0/10 | 0.367 | 0.472 0.5 0.5 0.194 0.472 0 0.5 0.306 0.278 0.444 | 56.4 |
| Qwen3.5-2B | 0/10 | 0/10 | 0/10 | 0.181 | 0.194 x9, 0.056 | 57.7 |
| Qwen3.5-9B | not finished (stopped at the deadline change, 0 trials graded) | | | | | |

- 2B has almost no reward spread (9/10 trials score exactly 0.194, the no-op score: 7/18 hidden tests already pass on
  the unmodified PR branch). GRPO gets zero advantage from it: in the 2B GRPO smoke step all 4 rollouts scored
  0.1944, so `reward_std = 0`, `loss = 0`, `grad_norm = 0`.
- 4B has spread (0 to 0.5), but all of it comes from the code half of the reward: no 4B trial edited the PR
  description (desc_updated 0/10), and 7/10 runs hit the 60-step limit. So GRPO on 4B can only learn "fix the code",
  not "update the description", unless some rollout happens to edit the body.
- **Chosen: Qwen/Qwen3.5-4B** (only candidate with nonzero reward variance).

## Method actually used

TRL 1.18.2 `GRPOTrainer` + `trl.experimental.harbor.HarborSpec`, vLLM 0.33.0 colocated, one H100 80GB, docker
sandboxes. Task converted to Harbor format (`rl/harbor_tasks/logslice/`, verifier = the grade.py graders with the
reward above; validated: oracle 1.0, code-only 0.5, no-op 0.194). Harness `rl/prdesc_env.py:PrDescBashEnv` (one
`bash` tool). Details and every workaround: `rl/STACK_NOTES.md`.

Settings (`rl/grpo_smoke.py`):

```
model Qwen/Qwen3.5-4B (bf16 weights), thinking on, temperature 1.0
num_generations 4, per_device_train_batch_size 1, gradient_accumulation_steps 4 (1 prompt x 4 rollouts per step)
max_steps 3, learning_rate 1e-5, max_tool_calling_iterations 25
max_completion_length 8192, vllm_max_model_length 12288, vllm_gpu_memory_utilization 0.3, gradient checkpointing
```

Command (node, project root):

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRL_EXPERIMENTAL_SILENCE=1 taskset -c 28-31,60-63,92-95,124-127 \
  .venv-rl/bin/python rl/grpo_smoke.py --model Qwen/Qwen3.5-4B --name qwen35-4b-grpo --thinking --max-steps 3 \
  --max-tool-calling-iterations 25 --max-completion-length 8192 --vllm-max-model-length 12288 \
  --learning-rate 1e-5 --save
```

Only 3 steps fit the budget (12 rollouts total). The stack worked end to end: every rollout ran in its own docker
sandbox, the verifier reward came back, and each step did a real update.

| step | rollout rewards | mean / std | loss | grad_norm | completions clipped at 8192 | step time | torch peak alloc |
|---|---|---|---|---|---|---|---|
| 1 | 0.167 0.389 0.194 0.194 | 0.236 / 0.103 | -0.270 | 1.375 | 3/4 | 110 s | 54.3 GB |
| 2 | 0.278 0.083 0.194 0.472 | 0.257 / 0.164 | -0.014 | 0.264 | 4/4 | 164 s | 61.4 GB |
| 3 | 0.194 0.139 0.194 0.194 | 0.181 / 0.028 | 0.008 | 1.18 | 2/4 | 170 s | 61.5 GB |

Total train time 445 s (+ ~2.5 min model/vLLM load). Sandbox start ~3 s, plus ~12.5 s `compose down` of the
previous sandbox from step 2 on; verifier ~2.5 s. No rollout edited the PR description during training, so the
description half of the reward was 0 throughout. Rewards did not trend up over 3 steps (0.236, 0.257, 0.181), and
most rollouts hit the 8192-token completion cap.

Rough throughput at these settings: ~2.5-3 min per step of 4 rollouts, i.e. ~20-24 steps/hour (80-100
rollouts/hour) on one H100.

## Before / after (existing harness + graders, `base` variant)

The trained checkpoint (`rl/runs/qwen35-4b-grpo/final`) was served by vLLM 0.33.0 on the H100 with the same serving
flags and the same mini-swe-agent harness/sampling as the baseline (baseline ran on an RTX PRO 4000). 5 trials were
started at 06:12Z; 4 finished before the deadline, trial 1 was still running at 06:17Z and was stopped (not counted).
No LLM judge was run (no trial in either column edited the PR body).

| | Qwen3.5-4B baseline (n=10) | baseline trials 1-5 | Qwen3.5-4B after 3 GRPO steps (n=4) |
|---|---|---|---|
| tests_pass | 3/10 | 2/5 | 1/4 |
| tests_untouched | 9/10 | 5/5 | 4/4 |
| desc_updated / no_stale / mentions_new | 0/10 | 0/5 | 0/4 |
| overall success | 0/10 | 0/5 | 0/4 |
| mean shaped reward R | 0.367 | 0.428 | 0.292 |
| per-trial R | 0.472 0.5 0.5 0.194 0.472 0 0.5 0.306 0.278 0.444 | | 0.194 0.278 0.5 0.194 |
| hit the 60-step limit | 6/10 | 4/5 | 4/4 |

Reading: no improvement. With 3 updates on 12 rollouts the trained model is not distinguishable from the base
model at this n (if anything lower: 0.29 vs 0.37 mean R), and neither ever touches the PR description. Two of the 4
eval trials (2 and 5) used all 60 steps in 39 s with ~3.6k completion tokens each; I did not get to inspect
those transcripts before the deadline.

## Caveats

- One task, trained and evaluated on the same task: any gain is memorisation of this task, not a general habit of
  updating PR descriptions.
- Tiny n (10 baseline trials, 5 eval trials at most) and only a handful of optimizer steps.
- The training harness differs from the eval harness: training uses TRL's tool loop with a short context
  (12k tokens) and a 25-call cap; evaluation uses mini-swe-agent with 60 steps and 64k context.
- The reward's description half was never triggered by 4B in the baseline, so it carried no signal.
- Not run: the LLM judge on the Qwen3.5 trials (no run edited the PR body, so desc_accurate would be false anyway),
  Qwen3.5-9B baseline, rejection-sampling SFT fallback (the Harbor stack worked, so it was not needed).
