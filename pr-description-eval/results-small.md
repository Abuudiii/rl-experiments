# Results

Agent: mini-swe-agent 2.4.6 (stock `mini.yaml` prompts, bash tool calls) driving Qwen/Qwen3.8-27B served by vLLM 0.32.0 (thinking on, reasoning_effort=low, temperature 0.7, top_p 0.95, step limit 60, 30 min wall clock). Judge: Claude Haiku via `claude -p`.

Columns base/contributing: the run described above. Columns qwen35-*: the same harness, task, sampling and limits with Qwen/Qwen3.5-2B / -4B (bf16) served by vLLM 0.33.0 with thinking on (the Qwen3.5 chat template has no reasoning_effort), `--language-model-only`, one RTX PRO 4000 Blackwell GPU each. qwen35-* trials were not run through the LLM judge (no run edited the PR body).

Success = tests_pass AND tests_untouched AND desc_no_stale_claims AND desc_mentions_new_behavior AND desc_accurate.

| metric | base | qwen35-4b/base | qwen35-2b/base |
|---|---|---|---|
| trials (n) | 10 | 10 | 10 |
| tests_pass | 10/10 (100%) | 3/10 (30%) | 0/10 (0%) |
| tests_untouched | 10/10 (100%) | 9/10 (90%) | 10/10 (100%) |
| desc_updated | 6/10 (60%) | 0/10 (0%) | 0/10 (0%) |
| desc_no_stale_claims | 6/10 (60%) | 0/10 (0%) | 0/10 (0%) |
| desc_mentions_new_behavior | 6/10 (60%) | 0/10 (0%) | 0/10 (0%) |
| desc_accurate | 5/10 (50%) | 0/10 (0%) | 0/10 (0%) |
| **overall success** | **5/10** | **0/10** | **0/10** |
| pass@1 | 50.0% | 0.0% | 0.0% |
| pass@3 | 91.7% | 0.0% | 0.0% |
| pass@5 | 99.6% | 0.0% | 0.0% |
| pass^1 | 50.0% | 0.0% | 0.0% |
| pass^3 | 8.3% | 0.0% | 0.0% |
| pass^5 | 0.4% | 0.0% | 0.0% |
| agent submitted | 10/10 | 4/10 | 1/10 |
| ran a gh command that edits the PR body | 6/10 | 0/10 | 0/10 |
| posted a PR comment / review reply | 8/10 | 2/10 | 0/10 |
| ... of which never edited the body | 2/10 | 2/10 | 0/10 |
| an agent command referenced CONTRIBUTING.md | 0/10 | 0/10 | 0/10 |
| mean model calls | 22.30 | 56.40 | 57.70 |
| mean agent actions (bash commands) | 30.40 | 56.30 | 57.60 |
| mean prompt / completion tokens | 344,028 / 13,300 | 1,072,477 / 17,362 | 794,523 / 7,312 |
| mean agent wall time (s) | 304.31 | 426.16 | 89.11 |

pass@k: unbiased estimator 1 - C(n-c,k)/C(n,k). pass^k: C(c,k)/C(n,k), the unbiased estimate of the probability that k independent attempts all succeed (c = successes out of n trials).

## Failure categories

Primary = first failing check in the order listed (one per trial); any = trials failing that check.

| category | base primary | base any | qwen35-4b/base primary | qwen35-4b/base any | qwen35-2b/base primary | qwen35-2b/base any |
|---|---|---|---|---|---|---|
| agent did not submit | 0 | 0 | 6 | 6 | 9 | 9 |
| review changes: tests fail | 0 | 0 | 2 | 7 | 1 | 10 |
| modified the provided tests | 0 | 0 | 0 | 1 | 0 | 0 |
| PR description never updated | 4 | 4 | 2 | 10 | 0 | 10 |
| description keeps stale claims | 0 | 4 | 0 | 10 | 0 | 10 |
| new behavior not described | 0 | 4 | 0 | 10 | 0 | 10 |
| judge: description inaccurate | 1 | 5 | 0 | 10 | 0 | 10 |
| success | 5 | - | 0 | - | 0 | - |

## Per-trial

| variant | trial | exit | calls | gh calls | body edits | comments/replies | read CONTRIBUTING | tests_pass | tests_untouched | desc_updated | desc_no_stale_claims | desc_mentions_new_behavior | desc_accurate | success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1 | Submitted | 21 | 9 | 1 | 3 | N | Y | Y | Y | Y | Y | Y | Y |
| base | 2 | Submitted | 23 | 4 | 0 | 0 | N | Y | Y | N | N | N | N | N |
| base | 3 | Submitted | 22 | 9 | 1 | 3 | N | Y | Y | Y | Y | Y | Y | Y |
| base | 4 | Submitted | 22 | 6 | 1 | 1 | N | Y | Y | Y | Y | Y | Y | Y |
| base | 5 | Submitted | 19 | 10 | 1 | 3 | N | Y | Y | Y | Y | Y | N | N |
| base | 6 | Submitted | 24 | 10 | 1 | 3 | N | Y | Y | Y | Y | Y | Y | Y |
| base | 7 | Submitted | 24 | 8 | 0 | 4 | N | Y | Y | N | N | N | N | N |
| base | 8 | Submitted | 21 | 9 | 1 | 3 | N | Y | Y | Y | Y | Y | Y | Y |
| base | 9 | Submitted | 27 | 6 | 0 | 1 | N | Y | Y | N | N | N | N | N |
| base | 10 | Submitted | 20 | 5 | 0 | 0 | N | Y | Y | N | N | N | N | N |
| qwen35-4b/base | 1 | LimitsExceeded | 60 | 5 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-4b/base | 2 | LimitsExceeded | 60 | 4 | 0 | 0 | N | Y | Y | N | N | N | ? | N |
| qwen35-4b/base | 3 | Submitted | 54 | 6 | 0 | 1 | N | Y | Y | N | N | N | ? | N |
| qwen35-4b/base | 4 | LimitsExceeded | 60 | 0 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-4b/base | 5 | LimitsExceeded | 60 | 5 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-4b/base | 6 | LimitsExceeded | 60 | 14 | 0 | 0 | N | N | N | N | N | N | ? | N |
| qwen35-4b/base | 7 | Submitted | 56 | 5 | 0 | 0 | N | Y | Y | N | N | N | ? | N |
| qwen35-4b/base | 8 | LimitsExceeded | 60 | 4 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-4b/base | 9 | Submitted | 45 | 4 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-4b/base | 10 | Submitted | 49 | 7 | 0 | 2 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 1 | LimitsExceeded | 60 | 1 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 2 | LimitsExceeded | 60 | 1 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 3 | LimitsExceeded | 60 | 0 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 4 | LimitsExceeded | 60 | 0 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 5 | LimitsExceeded | 60 | 0 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 6 | LimitsExceeded | 60 | 0 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 7 | LimitsExceeded | 60 | 0 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 8 | LimitsExceeded | 60 | 0 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 9 | LimitsExceeded | 60 | 6 | 0 | 0 | N | N | Y | N | N | N | ? | N |
| qwen35-2b/base | 10 | Submitted | 37 | 5 | 0 | 0 | N | N | Y | N | N | N | ? | N |

Not fully graded/judged (counted as failures): qwen35-4b/base/1, qwen35-4b/base/2, qwen35-4b/base/3, qwen35-4b/base/4, qwen35-4b/base/5, qwen35-4b/base/6, qwen35-4b/base/7, qwen35-4b/base/8, qwen35-4b/base/9, qwen35-4b/base/10, qwen35-2b/base/1, qwen35-2b/base/2, qwen35-2b/base/3, qwen35-2b/base/4, qwen35-2b/base/5, qwen35-2b/base/6, qwen35-2b/base/7, qwen35-2b/base/8, qwen35-2b/base/9, qwen35-2b/base/10
