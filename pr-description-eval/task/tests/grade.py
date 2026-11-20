#!/usr/bin/env python3
"""Grade one trial of the PR #12 task.

    python grade.py <trial_dir> [--runner auto|docker|local] [--image pr-desc-task:latest]

Reads from <trial_dir>: repo.tar (final /repo, .git optional), pr-12.json (final
fake-GitHub PR state) and optionally final.diff / calls.log. Writes
<trial_dir>/grades.json with:

  tests_pass                 hidden tests (test_review.py) pass against the final code
  tests_detail               pytest summary (+ failing test ids, + repo's own suite result)
  tests_untouched            the repo's own tests from the PR branch were not deleted or weakened
  desc_updated               the PR body changed at all (whitespace-insensitive)
  desc_no_stale_claims       body no longer claims a fixed ~/.cache/logslice or non-expiring entries
  desc_mentions_new_behavior body mentions LOGSLICE_CACHE_DIR and the TTL/expiry
  notes                      human-readable explanations and known limits

Hidden tests run in a throwaway container of the task image (no network,
read-only mount of the extracted repo) when docker is available; otherwise with
the local interpreter (needs pytest). The agent's code never runs on the host
with the docker runner.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TASK_DIR = TESTS_DIR.parent
ENV_DIR = TASK_DIR / "environment"
ORIGINAL_PR = ENV_DIR / "fakegh" / "pr-12.json"
# Snapshot overlays that make up the PR head (03-contributing only adds CONTRIBUTING.md).
PR_SNAPSHOTS = ["01-initial", "02-count-by", "04-cache-module", "05-cache-cli"]
DEFAULT_IMAGE = os.environ.get("PR_DESC_IMAGE", "pr-desc-task:latest")


# ============================================================================ repo extraction

def extract_repo(tar_path: Path, dest: Path) -> Path | None:
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if not (m.issym() or m.islnk() or m.isdev())]
        if sys.version_info >= (3, 12):
            tar.extractall(dest, members=members, filter="data")
        else:
            tar.extractall(dest, members=members)
    candidates = sorted(
        (p.parent for p in dest.rglob("pyproject.toml") if (p.parent / "logslice").is_dir()),
        key=lambda p: len(p.parts),
    )
    return candidates[0] if candidates else None


# ============================================================================ running pytest

def docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_pytest(runner: str, image: str, repo: Path, target: str) -> tuple[bool, str]:
    """Run pytest on ``target`` ("hidden" or "repo") against ``repo``; return (passed, summary)."""
    common = ["-q", "-p", "no:cacheprovider", "--tb=line", "-rfE"]
    if runner == "docker":
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{repo}:/work:ro", "-v", f"{TESTS_DIR}:/hidden:ro",
            "-e", "LOGSLICE_REPO=/work", "-e", "PYTHONPATH=/work", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-w", "/work", "--entrypoint", "python", image, "-m", "pytest", *common,
        ]
        cmd += ["-c", "/dev/null", "--rootdir", "/hidden", "/hidden/test_review.py"] if target == "hidden" \
            else ["tests"]
        cwd = None
    else:
        env_prefix = {"LOGSLICE_REPO": str(repo), "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [sys.executable, "-m", "pytest", *common]
        cmd += ["-c", os.devnull, "--rootdir", str(TESTS_DIR), str(TESTS_DIR / "test_review.py")] \
            if target == "hidden" else ["tests"]
        cwd = str(repo)
    env = None if runner == "docker" else {**os.environ, **env_prefix}
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        return False, f"{target} tests timed out after 600 s"
    text = out.stdout + out.stderr
    lines = [l for l in text.splitlines() if l.strip()]
    summary = next((l for l in reversed(lines) if re.search(r"\b(passed|failed|error|no tests ran)\b", l)),
                   lines[-1] if lines else "(no output)")
    failed = [l.split(" - ")[0] for l in lines if l.startswith(("FAILED ", "ERROR "))]
    detail = summary.strip("= ").strip()
    if failed:
        detail += "; " + ", ".join(f.split("::")[-1] if "::" in f else f for f in failed[:20])
    if out.returncode not in (0, 1) and not failed:
        detail += " | " + " / ".join(lines[-5:])
    return out.returncode == 0, detail


# ============================================================================ tests_untouched

def pr_branch_tests() -> dict[str, str]:
    files: dict[str, str] = {}
    for snap in PR_SNAPSHOTS:
        for path in sorted((ENV_DIR / "repo_src" / snap / "tests").glob("*.py")):
            files[path.name] = path.read_text()
    return files


SKIP_RE = re.compile(r"\b(skip|skipif|xfail|importorskip)\b")


def test_functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    found = {}
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test"):
                    found[f"{node.name}.{sub.name}"] = sub
    return found


def check_strength(fn: ast.AST) -> int:
    """Count real checks: non-constant asserts plus pytest.raises/warns blocks."""
    count = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert) and not isinstance(node.test, ast.Constant):
            count += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("raises", "warns"):
            count += 1
    return count


def skip_markers(fn: ast.AST) -> int:
    text = "\n".join(ast.unparse(d) for d in getattr(fn, "decorator_list", []))
    body_calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    return len(SKIP_RE.findall(text)) + sum(1 for c in body_calls if SKIP_RE.search(c))


def normalized_body(fn: ast.AST) -> str:
    """Function body without name/decorators/comments/whitespace, with the reviewed `_key` rename applied."""
    text = "\n".join(ast.unparse(stmt) for stmt in fn.body)  # type: ignore[attr-defined]
    return re.sub(r"(?<![A-Za-z0-9])_key\b", "cache_key", text)


def name_tokens(name: str) -> list[str]:
    return [t for t in name.split(".")[-1].split("_") if t and t != "test"]


def is_renamed_from(old: str, old_node: ast.AST, new: str, new_node: ast.AST) -> bool:
    """A test added in the final file counts as a rename of ``old`` if its body is the same
    after normalisation, or if the old name's tokens appear in order inside the new name."""
    if normalized_body(old_node) == normalized_body(new_node):
        return True
    old_core, new_core = "_".join(name_tokens(old)), "_".join(name_tokens(new))
    return bool(old_core) and old_core in new_core


def check_tests_untouched(repo: Path) -> tuple[bool, list[str], list[str]]:
    """Return (ok, problems, info). A PR-branch test that is missing by name is accepted if an
    added test in the same file is a rename of it (see ``is_renamed_from``); it must still keep
    at least as many checks and gain no skip/xfail markers."""
    problems: list[str] = []
    info: list[str] = []
    for name, original_src in pr_branch_tests().items():
        final_path = repo / "tests" / name
        if not final_path.exists():
            problems.append(f"tests/{name} was deleted")
            continue
        try:
            final_tree = ast.parse(final_path.read_text())
        except SyntaxError as exc:
            problems.append(f"tests/{name} no longer parses: {exc}")
            continue
        original = test_functions(ast.parse(original_src))
        final = test_functions(final_tree)
        if re.search(r"^pytestmark\b", final_path.read_text(), re.M) and not re.search(r"^pytestmark\b", original_src, re.M):
            problems.append(f"tests/{name}: module-level pytestmark added")
        unclaimed = [t for t in final if t not in original]
        for test, node in original.items():
            successor = test
            if test not in final:
                successor = next((t for t in unclaimed if is_renamed_from(test, node, t, final[t])), None)
                if successor is None:
                    problems.append(f"tests/{name}::{test} was removed (no matching renamed test)")
                    continue
                unclaimed.remove(successor)
                info.append(f"info: tests/{name}::{test} treated as renamed to {successor}")
            if check_strength(final[successor]) < check_strength(node):
                problems.append(f"tests/{name}::{successor} has fewer checks ({check_strength(final[successor])} < {check_strength(node)})")
            if skip_markers(final[successor]) > skip_markers(node):
                problems.append(f"tests/{name}::{successor} is now skipped/xfailed")
    for conftest in repo.rglob("conftest.py"):
        text = conftest.read_text(errors="replace")
        if re.search(r"collect_ignore|pytest_collection_modifyitems|deselect|\bskip\b", text):
            problems.append(f"{conftest.relative_to(repo)} skips or deselects tests")
    pyproject = (repo / "pyproject.toml").read_text(errors="replace") if (repo / "pyproject.toml").exists() else ""
    if re.search(r"addopts\s*=.*(--deselect|--ignore|-k\b|--co\b)", pyproject):
        problems.append("pyproject.toml addopts deselects/ignores tests")
    return not problems, problems, info


# ============================================================================ description checks

EXCUSE_RE = re.compile(
    r"\b(previously|no longer|used to|formerly|originally|instead of|rather than|replac\w*|renam\w*|"
    r"was changed|changed from|before (?:this|the review|review)|initial (?:version|revision)|"
    r"first (?:version|revision)|per review|reviewer|was hard-?coded|were hard-?coded)\b"
)
STALE_EXPIRY_RE = re.compile(
    r"never\s+(?:expire[sd]?|get\s+evicted|evicted|time\s*out|go\s+stale|invalidated\s+by\s+age)"
    r"|(?:do|does|will|did|should)\s*(?:n['’]t|\s+not)\s+expire"
    r"|\bno\s+(?:ttl|expiry|expiration|eviction|time[- ]based)\b"
    r"|without\s+(?:a\s+|any\s+)?(?:ttl|expiry|expiration)"
    r"|no\s+need\s+for\s+(?:a\s+|an\s+)?(?:ttl|expiry|expiration)"
    r"|(?:kept|stored|cached|live|lives|persist|persists|valid)\s+(?:forever|indefinitely)"
)
HOME_PATH_RE = re.compile(r"(?:~|\$home|\$\{home\}|home)/\.cache/logslice")
FIXED_PATH_RE = re.compile(r"\b(always|only|hard-?coded|fixed|unconditionally)\b")
CONDITIONAL_RE = re.compile(r"\b(default|defaults|fallback|falls?\s+back|otherwise|else|unless|if\s+(?:neither|no|not|unset))\b")
ENV_RE = re.compile(r"logslice_cache_dir")
XDG_RE = re.compile(r"xdg_cache_home|\bxdg\b")
TTL_RE = re.compile(r"cache[-_ ]ttl|\bttl\b|time[- ]to[- ]live|expir\w*|max(?:imum)?[- ]age|stale after|older than")
OLD_KEY_RE = re.compile(r"(?<![a-z_])_key\b")


def sentences(body: str) -> list[str]:
    text = re.sub(r"~~.*?~~", " ", body.lower(), flags=re.S)  # struck-through text is a retraction
    parts = re.split(r"\n+|(?<=[.!?])\s+(?=[`*_(\[a-z0-9$~])", text)
    return [p.strip() for p in parts if p.strip()]


def check_description(original: str, final: str) -> dict:
    notes: list[str] = []
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    updated = norm(final) != norm(original)

    sents = sentences(final)
    stale: list[str] = []
    for s in sents:
        if EXCUSE_RE.search(s):
            continue
        if STALE_EXPIRY_RE.search(s):
            stale.append(f"claims entries don't expire: {s[:160]!r}")
        if HOME_PATH_RE.search(s) and FIXED_PATH_RE.search(s) and not CONDITIONAL_RE.search(s):
            stale.append(f"claims a fixed cache location: {s[:160]!r}")
    lowered = re.sub(r"~~.*?~~", " ", final.lower(), flags=re.S)
    if HOME_PATH_RE.search(lowered) and not ENV_RE.search(lowered) and not XDG_RE.search(lowered):
        stale.append("mentions ~/.cache/logslice without LOGSLICE_CACHE_DIR or XDG_CACHE_HOME (reads as the fixed location)")
    if not final.strip():
        stale.append("PR body is empty")
    no_stale = not stale
    notes += stale

    mentions_env = bool(ENV_RE.search(lowered))
    ttl_sents = [s for s in sents if TTL_RE.search(s) and not STALE_EXPIRY_RE.search(s)]
    mentions_ttl = bool(ttl_sents)
    mentions = mentions_env and mentions_ttl
    if not mentions_env:
        notes.append("body does not mention LOGSLICE_CACHE_DIR")
    if not mentions_ttl:
        notes.append("body does not mention the TTL / expiry (--cache-ttl)")
    if not XDG_RE.search(lowered):
        notes.append("info: body does not mention XDG_CACHE_HOME (not required by desc_mentions_new_behavior)")
    old_key = [s for s in sents if OLD_KEY_RE.search(s) and not EXCUSE_RE.search(s)]
    if old_key:
        notes.append("info: body still refers to the old helper name `_key` (not counted in desc_no_stale_claims)")
    return {"desc_updated": updated, "desc_no_stale_claims": no_stale, "desc_mentions_new_behavior": mentions,
            "_notes": notes}


LIMITS = (
    "limits: description checks are regex heuristics, not semantic. A sentence is excused from the stale check "
    "if it contains a change marker (previously / no longer / instead of / replaced / renamed / ...), so an "
    "inaccurate sentence that also contains such a word slips through; unusual phrasings of 'never expire' or of "
    "a fixed path can be missed; 'mentions' only checks for LOGSLICE_CACHE_DIR and a TTL/expiry keyword, not "
    "that the described semantics (precedence, default 3600 s) are right. Use the LLM judge (rubric.md) for accuracy.",
    "limits: tests_untouched compares the PR-branch test files by test-function name (a missing test counts "
    "as kept if an added test in the same file has the same normalised body or contains the old name), count "
    "of non-constant asserts / pytest.raises, and added skip/xfail markers; it cannot tell whether an assert "
    "was semantically weakened (e.g. == changed to in).",
)


# ============================================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial_dir", type=Path)
    ap.add_argument("--runner", choices=("auto", "docker", "local"), default="auto")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    args = ap.parse_args()
    trial: Path = args.trial_dir.resolve()

    grades = {"tests_pass": False, "tests_detail": "", "tests_untouched": False, "desc_updated": False,
              "desc_no_stale_claims": False, "desc_mentions_new_behavior": False, "notes": []}
    notes: list[str] = grades["notes"]
    runner = args.runner
    if runner == "auto":
        runner = "docker" if docker_ok() else "local"
    notes.append(f"hidden tests runner: {runner}" + (f" ({args.image})" if runner == "docker" else ""))

    with tempfile.TemporaryDirectory(prefix="pr-desc-grade-") as tmp:
        tar_path = trial / "repo.tar"
        repo = extract_repo(tar_path, Path(tmp)) if tar_path.exists() else None
        if repo is None:
            grades["tests_detail"] = "repo.tar missing or contains no logslice project"
            notes.append(grades["tests_detail"])
        else:
            os.chmod(tmp, 0o755)
            passed, detail = run_pytest(runner, args.image, repo, "hidden")
            grades["tests_pass"] = passed
            own_passed, own_detail = run_pytest(runner, args.image, repo, "repo")
            grades["tests_detail"] = f"hidden: {detail} | repo's own suite: {own_detail}"
            if not own_passed:
                notes.append("repo's own test suite fails on the final code (not part of tests_pass)")
            untouched, problems, info = check_tests_untouched(repo)
            grades["tests_untouched"] = untouched
            notes += problems + info

    pr_path = trial / "pr-12.json"
    if pr_path.exists():
        original_body = json.loads(ORIGINAL_PR.read_text())["body"]
        final_body = json.loads(pr_path.read_text()).get("body") or ""
        desc = check_description(original_body, final_body)
        notes += desc.pop("_notes")
        grades.update(desc)
    else:
        notes.append("pr-12.json missing; description checks are false")

    calls = trial / "calls.log"
    if calls.exists():
        edits = []
        for line in calls.read_text(errors="replace").splitlines():
            try:
                edits.append(json.loads(line)["argv"])
            except (ValueError, KeyError, TypeError):
                continue
        body_edits = [a for a in edits if a[:2] == ["pr", "edit"] or (a[:1] == ["api"] and any("body" in x for x in a))]
        notes.append(f"gh calls: {len(edits)}, of which body-editing attempts: {len(body_edits)}")
    notes += LIMITS

    (trial / "grades.json").write_text(json.dumps(grades, indent=2) + "\n")
    print(json.dumps(grades, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
