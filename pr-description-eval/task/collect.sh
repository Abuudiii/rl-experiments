#!/usr/bin/env bash
# Copy a finished trial out of a (still running) task container, in the layout
# grade.py expects:
#   final.diff  `git diff main` of /repo including uncommitted and untracked files
#   pr-12.json  final fake-GitHub PR state
#   calls.log   every `gh` invocation (JSON lines)
#   repo.tar    tar of /repo without .git (top-level directory `repo/`)
# The agent transcript (trajectory.json) is written by the harness, not here.
#
# Usage: collect.sh <container> <out_dir>
set -euo pipefail
C=${1:?usage: collect.sh <container> <out_dir>}
OUT=${2:?usage: collect.sh <container> <out_dir>}
mkdir -p "$OUT"

# Temporary index so untracked files show up without touching the agent's index.
docker exec "$C" sh -c 'cd /repo && idx=$(mktemp) && cp .git/index "$idx" \
  && GIT_INDEX_FILE="$idx" git add -A \
  && GIT_INDEX_FILE="$idx" git diff --cached --binary main; rc=$?; rm -f "$idx"; exit $rc' > "$OUT/final.diff"
docker cp "$C:/var/lib/fakegh/pr-12.json" "$OUT/pr-12.json"
docker cp "$C:/var/lib/fakegh/calls.log" "$OUT/calls.log"
docker exec "$C" tar -C / --exclude=repo/.git -cf - repo > "$OUT/repo.tar"
echo "collected into $OUT"
