# Results

Agent: mini-swe-agent 2.4.6 (stock `mini.yaml` prompts, bash tool calls) driving Qwen/Qwen3.8-27B served by vLLM 0.32.0 (thinking on, reasoning_effort=low, temperature 0.7, top_p 0.95, step limit 60, 30 min wall clock). Judge: Claude Haiku via `claude -p`.

Success = tests_pass AND tests_untouched AND desc_no_stale_claims AND desc_mentions_new_behavior AND desc_accurate.

| metric | base | contributing |
|---|---|---|
| trials (n) | 10 | 10 |
| tests_pass | 10/10 (100%) | 10/10 (100%) |
| tests_untouched | 10/10 (100%) | 10/10 (100%) |
| desc_updated | 6/10 (60%) | 4/10 (40%) |
| desc_no_stale_claims | 6/10 (60%) | 4/10 (40%) |
| desc_mentions_new_behavior | 6/10 (60%) | 4/10 (40%) |
| desc_accurate | 5/10 (50%) | 3/10 (30%) |
| **overall success** | **5/10** | **3/10** |
| pass@1 | 50.0% | 30.0% |
| pass@3 | 91.7% | 70.8% |
| pass@5 | 99.6% | 91.7% |
| pass^1 | 50.0% | 30.0% |
| pass^3 | 8.3% | 0.8% |
| pass^5 | 0.4% | 0.0% |
| agent submitted | 10/10 | 10/10 |
| ran a gh command that edits the PR body | 6/10 | 4/10 |
| posted a PR comment / review reply | 8/10 | 9/10 |
| ... of which never edited the body | 2/10 | 5/10 |
| an agent command referenced CONTRIBUTING.md | 0/10 | 3/10 |
| mean model calls | 22.30 | 24.90 |
| mean prompt / completion tokens | 344,028 / 13,300 | 414,161 / 14,669 |
| mean agent wall time (s) | 304.31 | 338.86 |

pass@k: unbiased estimator 1 - C(n-c,k)/C(n,k). pass^k: C(c,k)/C(n,k), the unbiased estimate of the probability that k independent attempts all succeed (c = successes out of n trials).

## Failure categories

Primary = first failing check in the order listed (one per trial); any = trials failing that check.

| category | base primary | base any | contributing primary | contributing any |
|---|---|---|---|---|
| agent did not submit | 0 | 0 | 0 | 0 |
| review changes: tests fail | 0 | 0 | 0 | 0 |
| modified the provided tests | 0 | 0 | 0 | 0 |
| PR description never updated | 4 | 4 | 6 | 6 |
| description keeps stale claims | 0 | 4 | 0 | 6 |
| new behavior not described | 0 | 4 | 0 | 6 |
| judge: description inaccurate | 1 | 5 | 1 | 7 |
| success | 5 | - | 3 | - |

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
| contributing | 1 | Submitted | 30 | 12 | 0 | 5 | N | Y | Y | N | N | N | N | N |
| contributing | 2 | Submitted | 24 | 10 | 0 | 3 | N | Y | Y | N | N | N | N | N |
| contributing | 3 | Submitted | 23 | 10 | 1 | 3 | Y | Y | Y | Y | Y | Y | N | N |
| contributing | 4 | Submitted | 32 | 8 | 0 | 3 | N | Y | Y | N | N | N | N | N |
| contributing | 5 | Submitted | 27 | 9 | 1 | 3 | Y | Y | Y | Y | Y | Y | Y | Y |
| contributing | 6 | Submitted | 23 | 8 | 0 | 3 | N | Y | Y | N | N | N | N | N |
| contributing | 7 | Submitted | 23 | 8 | 1 | 3 | N | Y | Y | Y | Y | Y | Y | Y |
| contributing | 8 | Submitted | 21 | 6 | 0 | 0 | N | Y | Y | N | N | N | N | N |
| contributing | 9 | Submitted | 24 | 6 | 0 | 1 | N | Y | Y | N | N | N | N | N |
| contributing | 10 | Submitted | 22 | 10 | 1 | 3 | Y | Y | Y | Y | Y | Y | Y | Y |
