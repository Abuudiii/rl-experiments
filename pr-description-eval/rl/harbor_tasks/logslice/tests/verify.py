#!/usr/bin/env python3
"""Harbor verifier for the logslice PR #12 task (runs inside the agent's container after the rollout).

Reuses the graders from grade.py (copied verbatim from task/tests/) and writes /logs/verifier/reward.json:

  reward = 0                                                     if tests_untouched is false
         = 0.5 * hidden_tests_pass_fraction
           + 0.5 * mean(desc_no_stale_claims, desc_mentions_new_behavior)   otherwise

The other keys in reward.json are the grade.py booleans as 0/1, for logging. Harbor (and TRL's HarborEnv)
read the "reward" key.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import grade  # noqa: E402

# grade.py locates the PR-branch test sources and the original PR body relative to task/environment/;
# inside the sandbox those live under /tests/fixtures/.
grade.ENV_DIR = TESTS / "fixtures"
grade.ORIGINAL_PR = TESTS / "fixtures" / "fakegh" / "pr-12.json"

REPO = Path("/repo")
PR_STATE = Path("/var/lib/fakegh/pr-12.json")
OUT = Path("/logs/verifier")


def hidden_tests() -> tuple[int, int, str]:
    """Run test_review.py against /repo (same flags as grade.py's runner); return (passed, total, summary)."""
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "junit.xml"
        env = {**os.environ, "LOGSLICE_REPO": str(REPO), "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=line", "-rfE",
                 "-c", os.devnull, "--rootdir", str(TESTS), f"--junitxml={xml}", str(TESTS / "test_review.py")],
                cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return 0, 1, "hidden tests timed out"
        summary = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["(no output)"]
        if not xml.exists():
            return 0, 1, f"no junit report (exit {proc.returncode}): {summary[0]}"
        root = ET.parse(xml).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        total = sum(int(s.get("tests", 0)) for s in suites)
        bad = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)  # as grade.py: skips pass
        return max(total - bad, 0), max(total, 1), summary[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    passed, total, summary = hidden_tests()
    untouched, problems, info = grade.check_tests_untouched(REPO)

    original_body = json.loads(grade.ORIGINAL_PR.read_text())["body"]
    try:
        final_body = json.loads(PR_STATE.read_text()).get("body") or ""
    except (OSError, ValueError):
        final_body = ""
    desc = grade.check_description(original_body, final_body)
    notes = desc.pop("_notes")

    frac = passed / total
    desc_score = (desc["desc_no_stale_claims"] + desc["desc_mentions_new_behavior"]) / 2
    reward = 0.0 if not untouched else 0.5 * frac + 0.5 * desc_score
    rewards = {
        "reward": round(reward, 4),
        "hidden_tests_frac": round(frac, 4),
        "tests_pass": float(passed == total),
        "tests_untouched": float(untouched),
        "desc_updated": float(desc["desc_updated"]),
        "desc_no_stale_claims": float(desc["desc_no_stale_claims"]),
        "desc_mentions_new_behavior": float(desc["desc_mentions_new_behavior"]),
    }
    (OUT / "reward.json").write_text(json.dumps(rewards) + "\n")
    (OUT / "details.json").write_text(json.dumps(
        {"hidden_tests": f"{passed}/{total}: {summary}", "notes": problems + info + notes}, indent=2) + "\n")
    print(json.dumps(rewards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
