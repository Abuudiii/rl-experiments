#!/usr/bin/env python3
"""LLM judge (runs on the laptop): does the final PR description accurately describe the final diff?

For every trial dir with final.diff + pr-12.json, calls Claude Code headless (`claude -p`) with
task/rubric.md, the final diff and the final PR title/body, and writes judge.json:
  {"desc_accurate": bool, "reason": str, "model": str, "attempts": int}

Usage: python harness/judge.py [--variant base] [--model haiku] [--force]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = {
    "type": "object",
    "properties": {"desc_accurate": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["desc_accurate", "reason"],
    "additionalProperties": False,
}
SYSTEM = (
    "You are a careful code-review grader. You judge whether a pull request description matches the code. "
    "You have no tools; base your judgement only on the material in the user message."
)
MAX_DIFF_CHARS = 60_000


def build_prompt(rubric: str, diff: str, pr: dict) -> str:
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n[... diff truncated, {len(diff) - MAX_DIFF_CHARS} more characters ...]\n"
    return (
        f"# Rubric\n\n{rubric.strip()}\n\n"
        f"# Final diff of the PR branch against `main`\n\n```diff\n{diff}\n```\n\n"
        f"# Final PR title\n\n{pr.get('title', '')}\n\n"
        f"# Final PR description (body)\n\n````markdown\n{pr.get('body', '')}\n````\n\n"
        'Apply the rubric. Answer with JSON {"desc_accurate": <bool>, "reason": "<one or two sentences>"}.'
    )


def call_claude(prompt: str, model: str, timeout: int = 300) -> dict:
    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--tools", "",
        "--no-session-persistence",
        "--system-prompt", SYSTEM,
        "--json-schema", json.dumps(SCHEMA),
    ]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    if proc.returncode != 0:
        raise ValueError(f"claude exited {proc.returncode}: {proc.stderr[-500:] or proc.stdout[-500:]}")
    out = json.loads(proc.stdout)
    if out.get("is_error"):
        raise ValueError(f"claude error: {out.get('result')!r}")
    verdict = out.get("structured_output")
    if verdict is None:  # fall back to parsing the text result
        verdict = json.loads(out.get("result", ""))
    if not isinstance(verdict, dict) or not isinstance(verdict.get("desc_accurate"), bool) \
            or not isinstance(verdict.get("reason"), str):
        raise ValueError(f"malformed verdict: {verdict!r}")
    models = list(out.get("modelUsage", {}))
    return {"desc_accurate": verdict["desc_accurate"], "reason": verdict["reason"],
            "model": models[0] if models else model, "cost_usd": out.get("total_cost_usd")}


def judge_trial(trial: Path, rubric: str, model: str, retries: int) -> dict:
    diff = (trial / "final.diff").read_text(errors="replace")
    pr = json.loads((trial / "pr-12.json").read_text())
    prompt = build_prompt(rubric, diff, pr)
    errors = []
    for attempt in range(1, retries + 1):
        try:
            res = call_claude(prompt, model)
            res["attempts"] = attempt
            return res
        except (ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            errors.append(str(e))
            print(f"  {trial}: attempt {attempt} failed: {e}", file=sys.stderr)
    return {"desc_accurate": None, "reason": "judge failed", "errors": errors, "attempts": retries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append",
                    help="trial root under trials/, e.g. base or qwen35-4b/base (default: all)")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="re-judge trials that already have judge.json")
    args = ap.parse_args()

    rubric = (ROOT / "task" / "rubric.md").read_text()
    # A trial root is any dir under trials/ whose children hold collected trials (trials/base, trials/qwen35-4b/base).
    variants = args.variant or sorted({p.parent.parent.relative_to(ROOT / "trials").as_posix()
                                       for p in (ROOT / "trials").glob("**/final.diff")})
    for v in variants:
        trials = sorted((ROOT / "trials" / v).iterdir(), key=lambda p: (len(p.name), p.name))
        for t in trials:
            if not ((t / "final.diff").exists() and (t / "pr-12.json").exists()):
                print(f"{t}: missing collected files, skipped")
                continue
            if (t / "judge.json").exists() and not args.force:
                continue
            res = judge_trial(t, rubric, args.model, args.retries)
            (t / "judge.json").write_text(json.dumps(res, indent=2) + "\n")
            print(f"{v}/{t.name}: desc_accurate={res['desc_accurate']}  {res['reason'][:120]}")


if __name__ == "__main__":
    main()
