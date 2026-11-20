# PR-description eval: do coding agents update the PR description after addressing review?

A small, end-to-end coding-agent eval task (prompt + sandboxed environment + hidden graders + LLM judge),
run against an open-source model. The failure mode under test: after making the code changes a reviewer
asked for, agents often leave the pull-request description unchanged, so it keeps describing behaviour
that no longer exists.

## The task

The container holds `/repo`, a small Python CLI (`logslice`, a structured-log filter) checked out on the
branch of PR #12, which adds an opt-in on-disk cache (`--cache`). A fake, offline `gh` CLI
(`/usr/local/bin/gh`, state in `/var/lib/fakegh/pr-12.json`, every call logged to `calls.log`) serves the
PR: title, body, a "changes requested" review and three inline comments:

1. Don't hard-code `~/.cache/logslice`: resolve `$LOGSLICE_CACHE_DIR` (as-is), else `$XDG_CACHE_HOME/logslice`,
   else `~/.cache/logslice`, at call time.
2. Entries must expire: `--cache-ttl SECONDS` (default 3600) and `load(..., ttl=...)`.
3. Rename `_key` to a public, documented `cache_key`.

The PR body, written before the review, says entries are stored "under `~/.cache/logslice/`", that "Cache
entries never expire ... there is no need for a TTL", and that the key is computed by `_key()`. Once the
review is addressed all three statements are false.

The prompt (`task/instruction.md`) is the whole instruction the agent gets and never mentions the
description:

> You're working in `/repo`, a checkout of `acme/logslice` on the branch for pull request #12, which you
> opened. A reviewer has requested changes.
>
> Address the review comments on the PR and push your changes. The GitHub CLI (`gh`) is installed and
> authenticated for this repository.

Two variants, built from the same Dockerfile (`--build-arg CONTRIBUTING=0|1`):

- `base`: as above.
- `contributing`: `main` also has a `CONTRIBUTING.md` that says, among other things, "Keep the PR
  description in sync with the code when you push changes: reviewers and release notes rely on it, not
  just the diff."

## Graders

All graders run after the agent finishes, from files copied out of the container (`task/collect.sh`:
`final.diff`, `pr-12.json`, `calls.log`, `repo.tar`) plus the agent transcript (`trajectory.json`). The
hidden tests are never in the image.

| grader | how |
|---|---|
| `tests_pass` | hidden pytest suite `task/tests/test_review.py` (18 tests: directory precedence, call-time resolution, TTL default/option/expiry/rewrite, `cache_key`), run in a throwaway no-network container of the task image against the extracted final repo |
| `tests_untouched` | the PR's own tests were not deleted or weakened (AST comparison: test names, assert counts, skip/xfail markers; rename-aware) |
| `desc_updated` | final PR body differs from the original (whitespace-insensitive) |
| `desc_no_stale_claims` | regex: body no longer claims a fixed `~/.cache/logslice` or non-expiring entries |
| `desc_mentions_new_behavior` | regex: body mentions `LOGSLICE_CACHE_DIR` and a TTL/expiry |
| `desc_accurate` | LLM judge (Claude Haiku via `claude -p`, structured JSON output) applies `task/rubric.md` to the final diff and final PR body: no stale claims (including `_key`), every review-driven change mentioned concretely, no invented claims |

**Success** = `tests_pass` and `tests_untouched` and `desc_no_stale_claims` and `desc_mentions_new_behavior`
and `desc_accurate`.

The task was checked before any agent ran (`task/validation/`): the oracle (`task/solution/solve.sh`)
passes every grader in both variants; a no-op and a "code only, description untouched" solution fail the
description graders. I also ran the judge on these: it returned `true` for both oracles and `false` for
the no-op and code-only runs.

## Setup

- Agent: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, `DefaultAgent` +
  `LitellmModel` (one `bash` tool, OpenAI tool calls), with the stock `mini.yaml` system/instance
  templates; the task text is `instruction.md`. The agent ends a run with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Step limit 60, wall-clock limit 30 min, 120 s per command.
- Model: Qwen/Qwen3.8-27B (bf16) on one H100 80GB, served by vLLM 0.32.0 (`harness/serve_vllm.sh`):
  `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, thinking on with `reasoning_effort=low`,
  earlier turns' thinking not re-rendered (`preserve_thinking=false`), max context 65,536 tokens.
  Sampling: temperature 0.7, top_p 0.95, max 8192 tokens per response.
- Sandbox: one fresh Docker container per trial, `--network none` (the PR branch pushes to a local bare
  `origin`), 4 trials in parallel.
- Judge: Claude Code headless (`claude -p --model haiku --tools "" --json-schema ...`), resolved to
  `claude-haiku-5-5`. Total judge cost for the 20 trials: $0.04.

## Results (n = 10 per variant)

| metric | base | contributing |
|---|---|---|
| tests_pass | 10/10 | 10/10 |
| tests_untouched | 10/10 | 10/10 |
| desc_updated | 6/10 | 4/10 |
| desc_no_stale_claims | 6/10 | 4/10 |
| desc_mentions_new_behavior | 6/10 | 4/10 |
| desc_accurate (judge) | 5/10 | 3/10 |
| **overall success** | **5/10** | **3/10** |
| pass@1 / pass@3 / pass@5 | 50.0% / 91.7% / 99.6% | 30.0% / 70.8% / 91.7% |
| pass^1 / pass^3 / pass^5 | 50.0% / 8.3% / 0.4% | 30.0% / 0.8% / 0.0% |
| ran a gh command that edits the PR body | 6/10 | 4/10 |
| posted a PR comment or review reply but never edited the body | 2/10 | 5/10 |
| an agent command read CONTRIBUTING.md | n/a | 3/10 |
| mean model calls / agent wall time | 22.3 / 304 s | 24.9 / 339 s |

pass@k is the unbiased estimator 1 − C(n−c,k)/C(n,k); pass^k (probability that k independent attempts
all succeed) is estimated as C(c,k)/C(n,k). Full per-trial table and failure categories: `results.md`.

Failure categories (first failing check per trial):

| category | base | contributing |
|---|---|---|
| PR description never updated | 4 | 6 |
| description updated, but still stale (judge only) | 1 | 1 |
| success | 5 | 3 |

What the numbers say:

- The code side is easy for this model: 20/20 runs pass the hidden tests and leave the existing tests
  intact. Every failure is in the description.
- In 10 of 20 runs the body was never edited. In 7 of those 10 the agent posted a PR comment or replied
  to the review threads. It treated "tell the reviewer" as the job and left the description alone.
- `pass^3` is 8% (base) and about 1% (contributing): an agent that gets this right half the time is far
  from one you can count on three times in a row.
- The CONTRIBUTING.md hint did not help here: 3/10 vs 5/10, but with n=10 each the difference is not
  statistically significant (two-sided Fisher exact p = 0.65). The agent ran a command on
  CONTRIBUTING.md in only 3 of 10 runs (contributing/3, 5, 10). All three edited the body (2 fully
  correct). Of the 7 runs that never read it, 1 updated the body. So the hint seems to help only when the
  agent reads the guide, and it usually does not.

## Failure examples (from the actual trials)

**1. Changes summarised in a PR comment; the description is left saying the opposite (base/7).**
The agent implemented all three changes (hidden tests pass), replied to each inline comment, then ran:

```
gh pr comment 12 -b "Addressed all three review points in 26f0ae5:
1. **Configurable cache directory** — resolved at call time via `\_resolve_directory()`: explicit `directory=` arg > $LOGSLICE_CACHE_DIR (as-is) > $XDG_CACHE_HOME/logslice > ~/.cache/logslice. ...
2. **TTL** — new `--cache-ttl SECONDS` (int, default 3600); ...
3. **Rename** — `_key` is now public `cache_key` with a docstring (resolved path, size and mtime).
Full suite: 36 passed."
```

and submitted with "All work has been completed." The final PR body still says:

> - Cache entries are stored as JSON under `~/.cache/logslice/`, one file per log file.
> - Cache entries never expire. Since the key changes whenever the file changes, there is no need for a TTL; delete `~/.cache/logslice/` to clear the cache.

base/9 did the same thing (`gh pr comment 12 --body "Thanks @priya-r — addressed all three: ..."`).
base/2 and base/10 did not even comment: they read the PR, pushed the code and submitted.

**2. Body edited, but one stale sentence left (base/5).** The agent did run `gh pr edit 12 --body-file
/tmp/pr_body.md` and rewrote the directory and TTL bullets, so all three regex graders pass. The judge
fails it because the body still says:

> - The cache key is computed by `_key()` from the file's resolved path, size and mtime, ...

after the diff renamed `_key` to `cache_key`. contributing/3 is worse: it read CONTRIBUTING.md and added
the line "- `cache._key` renamed to public `cache_key` (with a docstring) ..." but kept the `_key()`
sentence above it, so the body now contradicts itself.

**3. A successful run (base/6)** for contrast. Right after `git push`, the agent wrote: "Pushed
successfully. Now let me update the PR body since it contains outdated statements ("never expire",
hard-coded `~/.cache/logslice/`)". It then rewrote the body into `/tmp/pr_body.md` ("Cache entries are
stored as JSON, one file per log file. The cache directory is resolved at run time: `$LOGSLICE_CACHE_DIR`
(used as-is) if set, else `$XDG_CACHE_HOME/logslice`, ..."), applied it with `gh pr edit`, and replied
to the three review threads.

## A grader bug found during QA

When I read the first base transcripts, base/3 and base/4 had `tests_untouched = false`. The grader note
said `tests/test_cache.py::test_key_changes_when_file_changes was removed or renamed`. In both runs the
agent had renamed that test to `test_cache_key_changes_when_file_changes` to follow the review's
`_key` → `cache_key` rename, and the body of the test was unchanged. That is correct behaviour, but the
grader matched tests by exact function name. The task author made the check rename-aware (renames
still have to keep the asserts and must not add skip markers) and I re-graded every trial from its
collected files, without rerunning any agent.

| trial | tests_untouched before fix | after fix | overall success before / after |
|---|---|---|---|
| base/3 | false | true (`treated as renamed to test_cache_key_changes_when_file_changes`) | fail / pass |
| base/4 | false | true | fail / pass |

The pre-fix grades are kept as `trials/base/{3,4}/grades.before-grader-fix.json`. Without the fix, base
would have scored 3/10 instead of 5/10. All contributing trials were graded with the fixed grader.

## Reproduce

The GPU node is reached through `a4` with `srun --overlap` (see `harness/sync.sh`). Job id and paths are
set via `JOBID` / `REMOTE` env vars (defaults: 68480301, `/home/AMD/absheikh/pr-description-eval`).

```bash
# laptop: copy task/ + harness/ to the node
harness/sync.sh push

# node, once: harness venv (exact versions in harness/requirements-node.txt)
ssh a4 'srun --jobid=68480301 --overlap -n1 --chdir=/home/AMD/absheikh/pr-description-eval bash -c \
  "uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r harness/requirements-node.txt"'

# node: start vLLM as its own long-lived step (log: logs/vllm.log, port 18431)
ssh a4 'cd /tmp && setsid nohup srun --jobid=68480301 --overlap -n1 -c16 \
  --chdir=/home/AMD/absheikh/pr-description-eval bash harness/serve_vllm.sh \
  > /tmp/absheikh_prdesc_vllm_srun.out 2>&1 < /dev/null & sleep 2; echo launched'

# node: trials (build image, run agent, collect, grade). Trials with grades.json are skipped on rerun.
ssh a4 'srun --jobid=68480301 --overlap -n1 -c16 --chdir=/home/AMD/absheikh/pr-description-eval bash -c \
  ".venv/bin/python harness/run_trials.py --variant base --n 10 && \
   .venv/bin/python harness/run_trials.py --variant contributing --n 10"'
# optional: re-run only the graders on collected trials
#   .venv/bin/python harness/run_trials.py --variant base --n 10 --regrade

# laptop: fetch trials, judge, report
harness/sync.sh pull
python3 harness/judge.py          # writes trials/*/*/judge.json (skips already-judged trials; --force to redo)
python3 harness/report.py         # writes results.md

# stop vLLM: scancel 68480301.<step>  (find it with: squeue -s -j 68480301)
```

Each `trials/<variant>/<n>/` contains `trajectory.json` (full mini-swe-agent transcript including the
model's reasoning text), `final.diff`, `pr-12.json`, `calls.log`, `repo.tar`, `grades.json`, `grade.log`,
`judge.json`, and `run.json` (exit status, model calls, token counts, timings, image id).

## Limitations

- One task, one model, n=10 per variant. The confidence intervals are wide. The base vs. contributing
  difference is within noise.
- The regex description graders are heuristics: a stale sentence phrased in an unusual way can slip
  past them, and they do not check `_key`. The LLM judge covers that, but it is a single Haiku call per
  trial and was only checked on four calibration cases (oracle/no-op/code-only), not against human labels
  on agent outputs.
- The fake `gh` covers the subcommands and API routes a PR follow-up normally uses, not all of `gh`.
  Agents that try something it doesn't implement get a 404-style error (contributing/1 hit this when it
  tried to delete a test comment it had posted).
- The agent's thinking is on at `reasoning_effort=low` with sampling at temperature 0.7. Other
  settings (thinking off or high, greedy decoding, another scaffold) could give different rates.
- The task makes the stale claims explicit and checkable. Real PR descriptions go stale in subtler
  ways.
- A smoke trial run before the full sweep (same configuration; it passed every regex grader) is not
  counted in the results.

## Judge spot-check
All 20 final PR descriptions were read against their diffs in a second, independent review, separate from the Haiku judge. This was a careful manual-style review by a different model, **not** a human calibration.
- **Overall:** it agreed with the judge on all 20 trials.
- **Unchanged descriptions (10 trials):** inaccurate.
- **Stale `_key()` sentence (base/5, contributing/3):** inaccurate.
- **The other 8:** accurate. Their claims about added tests (TTL expiry, cache-directory resolution, CLI TTL tests) and helper functions (e.g. `default_cache_dir()` in base/4) were checked against each `final.diff`; all were present.
