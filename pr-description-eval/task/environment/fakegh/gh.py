#!/usr/bin/env python3
"""Offline stand-in for the GitHub CLI, scoped to acme/logslice PR #12.

State lives in $FAKEGH_STATE_DIR/pr-12.json (default /var/lib/fakegh) and every
invocation is appended to calls.log there as one JSON line. Git information
(diffs, head SHA) comes from the repo at $FAKEGH_REPO (default /repo).
Nothing here touches the network.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path(os.environ.get("FAKEGH_STATE_DIR", "/var/lib/fakegh"))
STATE = STATE_DIR / "pr-12.json"
CALLS = STATE_DIR / "calls.log"
REPO = os.environ.get("FAKEGH_REPO", "/repo")
OWNER, NAME = "acme", "logslice"
FULL = f"{OWNER}/{NAME}"
ME = "mkowalski"
VERSION = "gh version 2.62.0 (2024-11-14)\nhttps://github.com/cli/cli/releases/tag/v2.62.0"

PR_FIELDS = [
    "additions", "assignees", "author", "baseRefName", "body", "changedFiles", "closed",
    "comments", "commits", "createdAt", "deletions", "files", "headRefName", "headRefOid",
    "id", "isDraft", "labels", "latestReviews", "mergeable", "number", "reviewDecision",
    "reviewRequests", "reviews", "state", "title", "updatedAt", "url",
]


class GhError(Exception):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- state

def load() -> dict:
    return json.loads(STATE.read_text())


def save(pr: dict) -> None:
    pr["updatedAt"] = now()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(pr, indent=2))
    os.replace(tmp, STATE)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout


def ref(pr: dict, name: str) -> str:
    """Prefer the pushed (remote-tracking) ref, like GitHub would."""
    for candidate in (f"refs/remotes/origin/{name}", f"refs/heads/{name}"):
        if git("rev-parse", "--verify", "-q", candidate).strip():
            return candidate
    return name


def pr_range(pr: dict) -> str:
    return f"{ref(pr, pr['baseRefName'])}...{ref(pr, pr['headRefName'])}"


def numstat(pr: dict) -> list[tuple[int, int, str]]:
    rows = []
    for line in git("diff", "--numstat", pr_range(pr)).splitlines():
        add, dele, path = line.split("\t", 2)
        rows.append((int(add) if add.isdigit() else 0, int(dele) if dele.isdigit() else 0, path))
    return rows


def computed(pr: dict, field: str):
    if field in ("additions", "deletions", "changedFiles", "files"):
        rows = numstat(pr)
        if field == "additions":
            return sum(r[0] for r in rows)
        if field == "deletions":
            return sum(r[1] for r in rows)
        if field == "changedFiles":
            return len(rows)
        return [{"path": p, "additions": a, "deletions": d} for a, d, p in rows]
    if field == "headRefOid":
        return git("rev-parse", ref(pr, pr["headRefName"])).strip()
    if field == "commits":
        out = git("log", "--reverse", "--format=%H%x00%s%x00%an%x00%aI", pr_range(pr).replace("...", ".."))
        commits = []
        for line in out.splitlines():
            oid, subject, author, date = line.split("\0")
            commits.append({"oid": oid, "messageHeadline": subject, "authors": [{"name": author}],
                            "committedDate": date})
        return commits
    if field == "id":
        return "PR_kwDOLogslice12"
    if field == "closed":
        return pr["state"] != "OPEN"
    if field == "mergeable":
        return "MERGEABLE"
    if field == "latestReviews":
        return pr.get("reviews", [])[-1:]
    return pr.get(field)


# --------------------------------------------------------------------------- arg helpers

class Args:
    """Tiny argv scanner: supports --flag value, --flag=value and short flags."""

    def __init__(self, argv: list[str]):
        self.argv = list(argv)

    def flag(self, *names: str) -> bool:
        for name in names:
            if name in self.argv:
                self.argv.remove(name)
                return True
        return False

    def value(self, *names: str) -> str | None:
        vals = self.values(*names)
        return vals[-1] if vals else None

    def values(self, *names: str) -> list[str]:
        found, rest, i = [], [], 0
        while i < len(self.argv):
            arg = self.argv[i]
            matched = False
            for name in names:
                if arg == name:
                    if i + 1 >= len(self.argv):
                        raise GhError(f"flag needs an argument: {name}")
                    found.append(self.argv[i + 1])
                    i += 2
                    matched = True
                    break
                if name.startswith("--") and arg.startswith(name + "="):
                    found.append(arg[len(name) + 1:])
                    i += 1
                    matched = True
                    break
            if not matched:
                rest.append(arg)
                i += 1
        self.argv = rest
        return found

    def positionals(self) -> list[str]:
        unknown = [a for a in self.argv if a.startswith("-") and a != "-"]
        if unknown:
            raise GhError(f"unknown flag: {unknown[0]}\n\nUsage: see `gh <command> --help`")
        return list(self.argv)


def read_text_arg(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    try:
        return Path(value).read_text()
    except OSError as exc:
        raise GhError(f"failed to read file {value}: {exc}") from None


def select_pr(selector: list[str]) -> dict:
    pr = load()
    if not selector:
        branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
        if branch and branch != pr["headRefName"]:
            raise GhError(f'no pull requests found for branch "{branch}"')
        return pr
    if len(selector) > 1:
        raise GhError(f"accepts at most 1 arg(s), received {len(selector)}")
    sel = selector[0]
    m = re.fullmatch(r"#?(\d+)|https?://github\.com/acme/logslice/pull/(\d+)/?", sel)
    if m:
        number = int(m.group(1) or m.group(2))
        if number != pr["number"]:
            raise GhError(f"GraphQL: Could not resolve to a PullRequest with the number of {number}. (repository.pullRequest)")
        return pr
    if sel == pr["headRefName"]:
        return pr
    raise GhError(f'no pull requests found for branch "{sel}"')


def emit_json(data, jq: str | None) -> None:
    text = json.dumps(data, indent=2 if jq is None and sys.stdout.isatty() else None)
    if jq is None:
        print(text)
        return
    if shutil.which("jq") is None:
        raise GhError("--jq requires jq, which is not installed")
    out = subprocess.run(["jq", "-r", jq], input=text, capture_output=True, text=True)
    if out.returncode != 0:
        raise GhError(out.stderr.strip() or "jq failed")
    sys.stdout.write(out.stdout)


def json_fields(spec: str) -> list[str]:
    fields = [f.strip() for f in spec.split(",") if f.strip()]
    bad = [f for f in fields if f not in PR_FIELDS]
    if bad:
        raise GhError(f'Unknown JSON field: "{bad[0]}"\nAvailable fields:\n  ' + "\n  ".join(PR_FIELDS))
    return fields


def json_or_fields_error(args: Args) -> list[str] | None:
    if "--json" in args.argv:
        idx = args.argv.index("--json")
        if idx + 1 >= len(args.argv) or args.argv[idx + 1].startswith("-"):
            raise GhError("Specify one or more comma-separated fields for `--json`:\n  " + "\n  ".join(PR_FIELDS))
    spec = args.value("--json")
    return json_fields(spec) if spec is not None else None


# --------------------------------------------------------------------------- gh pr

def review_state_label(state: str) -> str:
    return {"CHANGES_REQUESTED": "Changes requested", "APPROVED": "Approved",
            "COMMENTED": "Commented"}.get(state, state.title())


def pr_view(argv: list[str]) -> None:
    args = Args(argv)
    if args.flag("-w", "--web"):
        raise GhError("--web is not available in this environment (no browser)")
    show_comments = args.flag("-c", "--comments")
    jq = args.value("-q", "--jq")
    if args.value("-t", "--template") is not None:
        raise GhError("--template is not supported here; use --json with --jq")
    fields = json_or_fields_error(args)
    args.values("-R", "--repo")
    pr = select_pr(args.positionals())
    if jq is not None and fields is None:
        raise GhError("cannot use `--jq` without specifying `--json`")
    if fields is not None:
        emit_json({f: computed(pr, f) for f in fields}, jq)
        return

    reviewers = ", ".join(f"{r['author']['login']} ({review_state_label(r['state'])})" for r in pr.get("reviews", []))
    if not show_comments:
        print(f"title:\t{pr['title']}")
        print(f"state:\t{pr['state']}")
        print(f"author:\t{pr['author']['login']}")
        print(f"labels:\t{', '.join(l['name'] for l in pr.get('labels', []))}")
        print("assignees:\t")
        print(f"reviewers:\t{reviewers}")
        print("projects:\t")
        print("milestone:\t")
        print(f"number:\t{pr['number']}")
        print(f"url:\t{pr['url']}")
        print(f"additions:\t{computed(pr, 'additions')}")
        print(f"deletions:\t{computed(pr, 'deletions')}")
        print("auto-merge:\tdisabled")
        print("--")
        print(pr["body"])
        return

    # --comments: issue comments and reviews in chronological order. Unlike the
    # real gh, inline review comments are listed under their review so they are
    # discoverable without `gh api`.
    items = [("comment", c["createdAt"], c) for c in pr.get("comments", [])]
    items += [("review", r["submittedAt"], r) for r in pr.get("reviews", [])]
    for kind, _, item in sorted(items, key=lambda x: x[1]):
        print(f"author:\t{item['author']['login']}")
        print(f"association:\t{item.get('authorAssociation', 'none').lower()}")
        print("edited:\tfalse")
        print(f"status:\t{review_state_label(item['state']).lower() if kind == 'review' else 'none'}")
        print("--")
        print(item["body"])
        if kind == "review":
            for rc in pr.get("review_comments", []):
                if rc.get("pull_request_review_id") == item["id"]:
                    print()
                    print(f"[inline comment #{rc['id']} on {rc['path']}:{rc['line']}]")
                    print(rc["body"])
        print("--")


def pr_edit(argv: list[str]) -> None:
    args = Args(argv)
    body = args.value("-b", "--body")
    body_file = args.value("-F", "--body-file")
    title = args.value("-t", "--title")
    add_labels = [l for v in args.values("--add-label") for l in v.split(",") if l]
    remove_labels = [l for v in args.values("--remove-label") for l in v.split(",") if l]
    for unsupported in ("--add-reviewer", "--remove-reviewer", "--add-assignee", "--remove-assignee",
                        "--milestone", "--add-project", "--remove-project", "--base", "-B"):
        if args.values(unsupported):
            raise GhError(f"{unsupported} is not supported in this environment")
    args.values("-R", "--repo")
    pr = select_pr(args.positionals())
    if body is not None and body_file is not None:
        raise GhError("specify only one of `--body` or `--body-file`")
    if body_file is not None:
        body = read_text_arg(body_file)
    if body is None and title is None and not add_labels and not remove_labels:
        raise GhError("no changes specified; use --title, --body, --body-file, --add-label or --remove-label")
    if body is not None:
        pr["body"] = body
    if title is not None:
        pr["title"] = title
    names = [l["name"] for l in pr.get("labels", [])]
    names = [n for n in names if n not in remove_labels] + [l for l in add_labels if l not in names]
    pr["labels"] = [{"name": n} for n in names]
    save(pr)
    print(pr["url"])


def pr_comment(argv: list[str]) -> None:
    args = Args(argv)
    body = args.value("-b", "--body")
    body_file = args.value("-F", "--body-file")
    args.flag("--edit-last")
    args.values("-R", "--repo")
    pr = select_pr(args.positionals())
    if body_file is not None:
        body = read_text_arg(body_file)
    if not body:
        raise GhError("--body or --body-file required when not running interactively")
    comment_id = 3000 + len(pr.get("comments", [])) + 1
    pr.setdefault("comments", []).append({"id": comment_id, "author": {"login": ME}, "authorAssociation": "MEMBER",
                                          "body": body, "createdAt": now()})
    save(pr)
    print(f"{pr['url']}#issuecomment-{comment_id}")


def pr_review(argv: list[str]) -> None:
    args = Args(argv)
    body = args.value("-b", "--body")
    body_file = args.value("-F", "--body-file")
    approve = args.flag("-a", "--approve")
    request = args.flag("-r", "--request-changes")
    comment = args.flag("-c", "--comment")
    args.values("-R", "--repo")
    pr = select_pr(args.positionals())
    if approve or request:
        raise GhError("failed to create review: GraphQL: Can not approve your own pull request" if approve
                      else "failed to create review: GraphQL: Can not request changes on your own pull request")
    if not comment:
        raise GhError("--approve, --request-changes, or --comment required when not running interactively")
    if body_file is not None:
        body = read_text_arg(body_file)
    if not body:
        raise GhError("body cannot be blank for comment review")
    pr.setdefault("reviews", []).append({"id": 2000 + len(pr["reviews"]) + 1, "author": {"login": ME},
                                         "authorAssociation": "MEMBER", "state": "COMMENTED",
                                         "body": body, "submittedAt": now()})
    save(pr)
    print(f"- Reviewed pull request {FULL}#{pr['number']}")


def pr_list(argv: list[str]) -> None:
    args = Args(argv)
    jq = args.value("-q", "--jq")
    fields = json_or_fields_error(args)
    state = (args.value("-s", "--state") or "open").upper()
    for flag in ("-L", "--limit", "-A", "--author", "-B", "--base", "-H", "--head", "-l", "--label",
                 "-S", "--search", "-R", "--repo"):
        args.values(flag)
    args.flag("-d", "--draft")
    args.positionals()
    pr = load()
    prs = [pr] if state in ("ALL", pr["state"]) else []
    if fields is not None:
        emit_json([{f: computed(p, f) for f in fields} for p in prs], jq)
        return
    if not prs:
        print(f"no {state.lower()} pull requests in {FULL}", file=sys.stderr)
        return
    for p in prs:
        print(f"{p['number']}\t{p['title']}\t{p['headRefName']}\t{p['state']}\t{p['createdAt']}")


def pr_diff(argv: list[str]) -> None:
    args = Args(argv)
    name_only = args.flag("--name-only")
    args.flag("--patch")
    args.values("--color", "-R", "--repo")
    pr = select_pr(args.positionals())
    if name_only:
        sys.stdout.write(git("diff", "--name-only", pr_range(pr)))
    else:
        sys.stdout.write(git("diff", pr_range(pr)))


def pr_status(argv: list[str]) -> None:
    Args(argv).positionals()
    pr = load()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    print(f"\nRelevant pull requests in {FULL}\n")
    print("Current branch")
    if branch == pr["headRefName"]:
        decision = review_state_label(pr.get("reviewDecision") or "REVIEW_REQUIRED")
        print(f"  #{pr['number']}  {pr['title']} [{pr['headRefName']}]")
        print(f"  - {decision}")
    else:
        print(f"  There is no pull request associated with [{branch}]")
    print("\nCreated by you")
    print(f"  #{pr['number']}  {pr['title']} [{pr['headRefName']}]")
    print("\nRequesting a code review from you")
    print("  You have no pull requests to review\n")


def pr_checks(argv: list[str]) -> None:
    select_pr(Args(argv).positionals())
    raise GhError("no checks reported on the 'feature/result-cache' branch")


def pr_state_change(new_state: str):
    def run(argv: list[str]) -> None:
        args = Args(argv)
        args.values("-c", "--comment")
        args.flag("-d", "--delete-branch")
        pr = select_pr(args.positionals())
        pr["state"] = new_state
        save(pr)
        verb = "Closed" if new_state == "CLOSED" else "Reopened"
        print(f"✓ {verb} pull request {FULL}#{pr['number']} ({pr['title']})")
    return run


def pr_merge(argv: list[str]) -> None:
    raise GhError(f"X Pull request {FULL}#12 is not mergeable: the base branch policy requires an approving review.")


def pr_checkout(argv: list[str]) -> None:
    pr = select_pr(Args(argv).positionals())
    out = subprocess.run(["git", "-C", REPO, "checkout", pr["headRefName"]], capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    if out.returncode:
        raise GhError("checkout failed", out.returncode)


PR_COMMANDS = {
    "view": pr_view, "edit": pr_edit, "comment": pr_comment, "review": pr_review, "list": pr_list,
    "ls": pr_list, "diff": pr_diff, "status": pr_status, "checks": pr_checks, "merge": pr_merge,
    "close": pr_state_change("CLOSED"), "reopen": pr_state_change("OPEN"), "checkout": pr_checkout,
    "co": pr_checkout,
}

PR_HELP = """Work with GitHub pull requests.

USAGE
  gh pr <command> [flags]

AVAILABLE COMMANDS (this sandbox: acme/logslice, PR #12)
  checkout:   Check out a pull request in git
  checks:     Show CI status for a single pull request
  close:      Close a pull request
  comment:    Add a comment to a pull request
  diff:       View changes in a pull request
  edit:       Edit a pull request (--title, --body, --body-file, --add-label, --remove-label)
  list:       List pull requests in a repository
  merge:      Merge a pull request
  reopen:     Reopen a pull request
  review:     Add a review to a pull request
  status:     Show status of relevant pull requests
  view:       View a pull request (--comments, --json FIELDS, --jq EXPR)
"""


# --------------------------------------------------------------------------- gh api

def rest_pull(pr: dict) -> dict:
    return {
        "url": f"https://api.github.com/repos/{FULL}/pulls/{pr['number']}",
        "html_url": pr["url"], "number": pr["number"], "state": pr["state"].lower(),
        "title": pr["title"], "body": pr["body"], "user": {"login": pr["author"]["login"]},
        "labels": pr.get("labels", []), "draft": pr.get("isDraft", False),
        "created_at": pr["createdAt"], "updated_at": pr["updatedAt"],
        "head": {"ref": pr["headRefName"], "sha": computed(pr, "headRefOid"), "repo": {"full_name": FULL}},
        "base": {"ref": pr["baseRefName"], "repo": {"full_name": FULL}},
        "additions": computed(pr, "additions"), "deletions": computed(pr, "deletions"),
        "changed_files": computed(pr, "changedFiles"), "comments": len(pr.get("comments", [])),
        "review_comments": len(pr.get("review_comments", [])),
    }


def rest_issue_comment(c: dict) -> dict:
    return {"id": c["id"], "user": {"login": c["author"]["login"]}, "body": c["body"],
            "created_at": c["createdAt"], "author_association": c.get("authorAssociation", "NONE"),
            "html_url": f"https://github.com/{FULL}/pull/12#issuecomment-{c['id']}"}


def rest_review(r: dict) -> dict:
    return {"id": r["id"], "user": {"login": r["author"]["login"]}, "body": r["body"], "state": r["state"],
            "submitted_at": r["submittedAt"]}


def api(argv: list[str]) -> None:
    args = Args(argv)
    method = args.value("-X", "--method")
    jq = args.value("-q", "--jq")
    raw_fields = args.values("-f", "--raw-field")
    typed_fields = args.values("-F", "--field")
    input_file = args.value("--input")
    args.values("-H", "--header", "-p", "--preview", "--hostname", "--cache", "-t", "--template")
    args.flag("--paginate", "--slurp", "-i", "--include", "--silent", "--verbose")
    pos = args.positionals()
    if len(pos) != 1:
        raise GhError("usage: gh api <endpoint> [flags]")
    endpoint = pos[0]

    payload: dict = {}
    if input_file is not None:
        try:
            payload.update(json.loads(read_text_arg(input_file)))
        except ValueError as exc:
            raise GhError(f"invalid JSON in --input: {exc}") from None
    for item in raw_fields:
        key, _, val = item.partition("=")
        payload[key] = val
    for item in typed_fields:
        key, _, val = item.partition("=")
        if val.startswith("@"):
            payload[key] = read_text_arg(val[1:])
        elif val in ("true", "false"):
            payload[key] = val == "true"
        elif val.isdigit():
            payload[key] = int(val)
        else:
            payload[key] = val
    method = (method or ("POST" if payload else "GET")).upper()

    if endpoint == "graphql":
        raise GhError("graphql is not available in this offline sandbox; use the REST endpoints "
                      "(e.g. repos/{owner}/{repo}/pulls/12, .../pulls/12/comments) or `gh pr view`")
    path = endpoint.split("?", 1)[0].strip("/")
    path = path.replace("{owner}", OWNER).replace("{repo}", NAME).replace(":owner", OWNER).replace(":repo", NAME)
    pr = load()
    status_404 = GhError(f"gh: Not Found (HTTP 404)\n{{\"message\":\"Not Found\",\"status\":\"404\"}}")

    if path == "user":
        return emit_json({"login": ME, "name": "Marta Kowalski"}, jq)
    if path == f"repos/{FULL}":
        return emit_json({"full_name": FULL, "name": NAME, "owner": {"login": OWNER}, "default_branch": "main",
                          "private": False, "html_url": f"https://github.com/{FULL}"}, jq)
    if path == f"repos/{FULL}/pulls":
        if method != "GET":
            raise GhError("creating pull requests is not supported in this sandbox")
        return emit_json([rest_pull(pr)] if pr["state"] == "OPEN" else [], jq)

    m = re.fullmatch(rf"repos/{FULL}/(pulls|issues)/(\d+)(/.*)?", path)
    if not m or int(m.group(2)) != pr["number"]:
        raise status_404
    kind, rest = m.group(1), m.group(3) or ""

    if rest == "":
        if method == "GET":
            return emit_json(rest_pull(pr), jq)
        if method in ("PATCH", "POST"):
            unknown = set(payload) - {"title", "body", "state", "base"}
            if unknown:
                raise GhError(f"gh: Validation Failed (HTTP 422): unsupported fields {sorted(unknown)}")
            if "title" in payload:
                pr["title"] = str(payload["title"])
            if "body" in payload:
                pr["body"] = str(payload["body"])
            if "state" in payload:
                pr["state"] = str(payload["state"]).upper()
            save(pr)
            return emit_json(rest_pull(pr), jq)
        raise status_404

    if kind == "issues" and rest == "/comments":
        if method == "GET":
            return emit_json([rest_issue_comment(c) for c in pr.get("comments", [])], jq)
        if method == "POST":
            if not payload.get("body"):
                raise GhError("gh: Validation Failed (HTTP 422): body is required")
            comment_id = 3000 + len(pr.get("comments", [])) + 1
            comment = {"id": comment_id, "author": {"login": ME}, "authorAssociation": "MEMBER",
                       "body": str(payload["body"]), "createdAt": now()}
            pr.setdefault("comments", []).append(comment)
            save(pr)
            return emit_json(rest_issue_comment(comment), jq)
        raise status_404
    if kind == "pulls" and rest == "/reviews" and method == "GET":
        return emit_json([rest_review(r) for r in pr.get("reviews", [])], jq)
    if kind == "pulls" and rest == "/commits" and method == "GET":
        return emit_json([{"sha": c["oid"], "commit": {"message": c["messageHeadline"]}}
                          for c in computed(pr, "commits")], jq)
    if kind == "pulls" and rest == "/files" and method == "GET":
        return emit_json([{"filename": f["path"], "additions": f["additions"], "deletions": f["deletions"]}
                          for f in computed(pr, "files")], jq)
    if kind == "pulls" and rest == "/comments" and method == "GET":
        return emit_json(pr.get("review_comments", []), jq)

    reply = re.fullmatch(r"/comments/(\d+)/replies", rest) if kind == "pulls" else None
    if (reply and method == "POST") or (kind == "pulls" and rest == "/comments" and method == "POST"):
        if not payload.get("body"):
            raise GhError("gh: Validation Failed (HTTP 422): body is required")
        parent_id = int(reply.group(1)) if reply else payload.get("in_reply_to")
        parent = next((c for c in pr.get("review_comments", []) if c["id"] == parent_id), None)
        if parent_id is not None and parent is None:
            raise status_404
        new = {"id": 4000 + len(pr.get("review_comments", [])) + 1, "pull_request_review_id": None,
               "user": {"login": ME}, "path": parent["path"] if parent else payload.get("path"),
               "line": parent["line"] if parent else payload.get("line"), "side": "RIGHT", "commit_id": None,
               "diff_hunk": parent["diff_hunk"] if parent else "", "body": str(payload["body"]),
               "created_at": now(), "in_reply_to_id": parent_id}
        pr.setdefault("review_comments", []).append(new)
        save(pr)
        return emit_json(new, jq)
    raise status_404


# --------------------------------------------------------------------------- misc

def repo_cmd(argv: list[str]) -> None:
    if not argv or argv[0] != "view":
        raise GhError("only `gh repo view` is available in this sandbox")
    args = Args(argv[1:])
    jq = args.value("-q", "--jq")
    spec = args.value("--json")
    if spec is not None:
        data = {"name": NAME, "owner": {"login": OWNER}, "nameWithOwner": FULL,
                "url": f"https://github.com/{FULL}", "defaultBranchRef": {"name": "main"},
                "description": "Filter and summarise structured log files"}
        return emit_json({k: data.get(k) for k in spec.split(",")}, jq)
    print(f"name:\t{FULL}\ndescription:\tFilter and summarise structured log files\n--\n")
    print(Path(REPO, "README.md").read_text() if Path(REPO, "README.md").exists() else "")


def auth_cmd(argv: list[str]) -> None:
    if argv[:1] == ["status"]:
        print("github.com\n  ✓ Logged in to github.com account mkowalski (GH_TOKEN)\n"
              "  - Active account: true\n  - Git operations protocol: https\n  - Token: ghp_************************************")
        return
    if argv[:1] == ["token"]:
        print("ghp_fake_sandbox_token")
        return
    raise GhError("only `gh auth status` is available in this sandbox")


ROOT_HELP = """Work seamlessly with GitHub from the command line.

USAGE
  gh <command> <subcommand> [flags]

COMMANDS AVAILABLE IN THIS SANDBOX
  pr:          Manage pull requests (view, edit, comment, review, list, diff, status, checks, ...)
  api:         Make an authenticated GitHub REST API request
  repo view:   View repository information
  auth status: View authentication status

Run `gh pr --help` for pull request subcommands.
"""


def dispatch(argv: list[str]) -> None:
    if not argv or argv[0] in ("help", "-h", "--help"):
        if len(argv) > 1 and argv[1] == "pr":
            print(PR_HELP)
        else:
            print(ROOT_HELP)
        return
    if argv[0] in ("--version", "version"):
        print(VERSION)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "pr":
        if not rest or rest[0] in ("-h", "--help", "help"):
            print(PR_HELP)
            return
        sub = rest[0]
        if sub not in PR_COMMANDS:
            raise GhError(f'unknown command "{sub}" for "gh pr"\n\n{PR_HELP}')
        if any(a in ("-h", "--help") for a in rest[1:]):
            print(PR_HELP)
            return
        PR_COMMANDS[sub](rest[1:])
        return
    if cmd == "api":
        api(rest)
        return
    if cmd == "repo":
        repo_cmd(rest)
        return
    if cmd == "auth":
        auth_cmd(rest)
        return
    raise GhError(f'unknown command "{cmd}" for "gh"\n\n{ROOT_HELP}')


def main() -> int:
    argv = sys.argv[1:]
    code, error = 0, None
    try:
        dispatch(argv)
    except GhError as exc:
        code, error = exc.code, str(exc)
        print(error, file=sys.stderr)
    except BrokenPipeError:
        code = 0
    except Exception as exc:  # never crash with a traceback; still log the call
        code, error = 1, f"internal error: {exc!r}"
        print(f"gh: {error}", file=sys.stderr)
    finally:
        try:
            with open(CALLS, "a") as log:
                log.write(json.dumps({"ts": now(), "argv": argv, "cwd": os.getcwd(), "exit": code,
                                      "error": error}) + "\n")
        except OSError:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
