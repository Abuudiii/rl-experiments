#!/usr/bin/env python3
"""Aggregate trials/<root>/<n>/{run,grades,judge}.json into results.md (one column per trial root).

Usage: python harness/report.py [--out results.md] [--variants base contributing qwen35-4b/base ...]
"""

import argparse
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRADERS = ["tests_pass", "tests_untouched", "desc_updated", "desc_no_stale_claims",
           "desc_mentions_new_behavior", "desc_accurate"]
SUCCESS_KEYS = ["tests_pass", "tests_untouched", "desc_no_stale_claims", "desc_mentions_new_behavior",
                "desc_accurate"]
# Failure categories in priority order (first match = primary category).
CATEGORIES = [
    ("agent did not submit", lambda t: t["exit"] != "Submitted"),
    ("review changes: tests fail", lambda t: t["tests_pass"] is not True),
    ("modified the provided tests", lambda t: t["tests_untouched"] is not True),
    ("PR description never updated", lambda t: t["desc_updated"] is not True),
    ("description keeps stale claims", lambda t: t["desc_no_stale_claims"] is not True),
    ("new behavior not described", lambda t: t["desc_mentions_new_behavior"] is not True),
    ("judge: description inaccurate", lambda t: t["desc_accurate"] is not True),
]


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased P(at least one of k samples succeeds) (Chen et al. 2021)."""
    if k > n:
        return None
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float | None:
    """Unbiased estimate of P(all k samples succeed) = p^k, i.e. C(c,k)/C(n,k)."""
    if k > n:
        return None
    return comb(c, k) / comb(n, k)


def is_body_edit(argv: list[str]) -> bool:
    """gh invocation that tries to change the PR body (`gh pr edit --body/-b/--body-file/-F`, or a PATCH via gh api)."""
    if argv[:2] == ["pr", "edit"]:
        return any(a in ("--body", "-b", "--body-file", "-F") or a.startswith(("--body=", "--body-file="))
                   for a in argv)
    if argv[:1] == ["api"]:
        method = any(a in ("PATCH", "--method=PATCH", "-XPATCH") for a in argv)
        return method and any("body=" in a for a in argv)
    return False

def is_comment(argv: list[str]) -> bool:
    """gh invocation that posts a PR comment, review, or review-comment reply (instead of / besides a body edit)."""
    if argv[:2] in (["pr", "comment"], ["pr", "review"]):
        return True
    if argv[:1] == ["api"] and not is_body_edit(argv) and "DELETE" not in argv:
        posts_body = any(a.startswith("body=") for a in argv)
        return posts_body and any(seg in a for a in argv for seg in ("/comments", "/replies", "/reviews"))
    return False



def load_trials(variant_dir: Path) -> list[dict]:
    trials = []
    for d in sorted((p for p in variant_dir.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        grades = json.loads((d / "grades.json").read_text()) if (d / "grades.json").exists() else {}
        judge = json.loads((d / "judge.json").read_text()) if (d / "judge.json").exists() else {}
        calls = []
        if (d / "calls.log").exists():
            for line in (d / "calls.log").read_text(errors="replace").splitlines():
                try:
                    calls.append(json.loads(line).get("argv", []))
                except json.JSONDecodeError:
                    calls.append([])
        t = {"trial": int(d.name), "exit": run.get("agent_exit_status", "missing"),
             "api_calls": run.get("api_calls"), "n_actions": run.get("n_actions"), "prompt_tokens": run.get("prompt_tokens"),
             "completion_tokens": run.get("completion_tokens"), "seconds": run.get("agent_seconds"),
             "gh_calls": len(calls), "body_edits": sum(is_body_edit(c) for c in calls),
             "comments": sum(is_comment(c) for c in calls),
             "read_contributing": agent_read_contributing(d / "trajectory.json"),
             "judge_reason": judge.get("reason", ""),
             "graded": bool(grades), "judged": "desc_accurate" in judge}
        for g in GRADERS[:-1]:
            t[g] = grades.get(g)
        t["desc_accurate"] = judge.get("desc_accurate")
        t["success"] = all(t[k] is True for k in SUCCESS_KEYS)
        t["category"] = next((name for name, f in CATEGORIES if f(t)), "success")
        trials.append(t)
    return trials


def agent_read_contributing(traj: Path) -> bool:
    """Did any agent command mention CONTRIBUTING (e.g. `cat CONTRIBUTING.md`)? Listing the dir does not count."""
    if not traj.exists():
        return False
    msgs = json.loads(traj.read_text()).get("messages", [])
    return any("CONTRIBUTING" in a.get("command", "")
               for m in msgs if m.get("role") == "assistant" for a in m.get("extra", {}).get("actions", []))


def fmt(x, pct=True, digits=0):
    if x is None:
        return "n/a"
    return f"{100 * x:.{digits}f}%" if pct else f"{x:.2f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results.md"))
    ap.add_argument("--variants", nargs="+", default=["base", "contributing"],
                    help="trial roots under trials/ (columns), e.g. base qwen35-4b/base")
    args = ap.parse_args()

    variants = [v for v in args.variants if (ROOT / "trials" / v).is_dir()]
    data = {v: load_trials(ROOT / "trials" / v) for v in variants}
    small = [v for v in variants if v.startswith("qwen35-")]
    L = ["# Results", "",
         "Agent: mini-swe-agent 2.4.6 (stock `mini.yaml` prompts, bash tool calls) driving Qwen/Qwen3.8-27B "
         "served by vLLM 0.32.0 (thinking on, reasoning_effort=low, temperature 0.7, top_p 0.95, "
         "step limit 60, 30 min wall clock). Judge: Claude Haiku via `claude -p`.",
         "", "Success = tests_pass AND tests_untouched AND desc_no_stale_claims AND "
         "desc_mentions_new_behavior AND desc_accurate.", ""]
    if small:
        L[4:4] = ["Columns base/contributing: the run described above. Columns qwen35-*: the same harness, task, sampling and "
                  "limits with Qwen/Qwen3.5-2B / -4B (bf16) served by vLLM 0.33.0 with thinking on (the Qwen3.5 chat "
                  "template has no reasoning_effort), `--language-model-only`, one RTX PRO 4000 Blackwell GPU each. "
                  "qwen35-* trials were not run through the LLM judge (no run edited the PR body).", ""]

    # Main table
    header = ["metric"] + variants
    rows = [["trials (n)"] + [str(len(data[v])) for v in variants]]
    for g in GRADERS:
        cells = []
        for v in variants:
            vals = [t[g] for t in data[v]]
            known = [x for x in vals if x is not None]
            cells.append(f"{sum(known)}/{len(vals)} ({fmt(sum(known) / len(vals))})" if vals else "n/a")
        rows.append([g] + cells)
    rows.append(["**overall success**"] + [
        f"**{sum(t['success'] for t in data[v])}/{len(data[v])}**" for v in variants])
    for k in (1, 3, 5):
        rows.append([f"pass@{k}"] + [fmt(pass_at_k(len(data[v]), sum(t['success'] for t in data[v]), k), digits=1)
                                     for v in variants])
    for k in (1, 3, 5):
        rows.append([f"pass^{k}"] + [fmt(pass_hat_k(len(data[v]), sum(t['success'] for t in data[v]), k), digits=1)
                                     for v in variants])
    rows.append(["agent submitted"] + [f"{sum(t['exit'] == 'Submitted' for t in data[v])}/{len(data[v])}"
                                       for v in variants])
    rows.append(["ran a gh command that edits the PR body"] + [
        f"{sum(t['body_edits'] > 0 for t in data[v])}/{len(data[v])}" for v in variants])
    rows.append(["posted a PR comment / review reply"] + [
        f"{sum(t['comments'] > 0 for t in data[v])}/{len(data[v])}" for v in variants])
    rows.append(["... of which never edited the body"] + [
        f"{sum(t['comments'] > 0 and t['body_edits'] == 0 for t in data[v])}/{len(data[v])}" for v in variants])
    rows.append(["an agent command referenced CONTRIBUTING.md"] + [
        f"{sum(t['read_contributing'] for t in data[v])}/{len(data[v])}" for v in variants])
    rows.append(["mean model calls"] + [fmt(mean(t["api_calls"] for t in data[v]), False) for v in variants])
    rows.append(["mean agent actions (bash commands)"] + [fmt(mean(t["n_actions"] for t in data[v]), False)
                                                          for v in variants])
    rows.append(["mean prompt / completion tokens"] + [
        f"{mean(t['prompt_tokens'] for t in data[v]) or 0:,.0f} / {mean(t['completion_tokens'] for t in data[v]) or 0:,.0f}"
        for v in variants])
    rows.append(["mean agent wall time (s)"] + [fmt(mean(t["seconds"] for t in data[v]), False) for v in variants])
    L += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    L += ["| " + " | ".join(r) + " |" for r in rows]
    L += ["", "pass@k: unbiased estimator 1 - C(n-c,k)/C(n,k). pass^k: C(c,k)/C(n,k), the unbiased estimate "
          "of the probability that k independent attempts all succeed (c = successes out of n trials).", ""]

    # Failure categories
    L += ["## Failure categories", "",
          "Primary = first failing check in the order listed (one per trial); any = trials failing that check.", "",
          "| category | " + " | ".join(f"{v} primary | {v} any" for v in variants) + " |",
          "|---|" + "---|---|" * len(variants)]
    for name, f in CATEGORIES:
        cells = []
        for v in variants:
            cells += [str(sum(t["category"] == name for t in data[v])), str(sum(bool(f(t)) for t in data[v]))]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("| success | " + " | ".join(f"{sum(t['success'] for t in data[v])} | -" for v in variants) + " |")

    # Per-trial table
    L += ["", "## Per-trial", "",
          "| variant | trial | exit | calls | gh calls | body edits | comments/replies | read CONTRIBUTING | "
          + " | ".join(GRADERS) + " | success |",
          "|---|---|---|---|---|---|---|---|" + "---|" * (len(GRADERS) + 1)]
    yn = {True: "Y", False: "N", None: "?"}
    for v in variants:
        for t in data[v]:
            L.append(f"| {v} | {t['trial']} | {t['exit']} | {t['api_calls']} | {t['gh_calls']} | {t['body_edits']} | "
                     f"{t['comments']} | {yn[t['read_contributing']]} | "
                     + " | ".join(yn[t[g]] for g in GRADERS) + f" | {yn[t['success']]} |")
    missing = [f"{v}/{t['trial']}" for v in variants for t in data[v] if not (t["graded"] and t["judged"])]
    if missing:
        L += ["", f"Not fully graded/judged (counted as failures): {', '.join(missing)}"]
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
