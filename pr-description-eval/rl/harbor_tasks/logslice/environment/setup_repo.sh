#!/usr/bin/env bash
# Build /repo with a realistic history: two commits on main (plus an optional
# CONTRIBUTING.md commit), then PR #12's two commits on feature/result-cache.
# Each snapshot directory in repo_src/ is an overlay of files for one commit.
# Also creates a local bare "origin" so `git push` works without network.
#
# Usage: CONTRIBUTING=0|1 setup_repo.sh <repo_src_dir>
set -euo pipefail

SRC=${1:?usage: setup_repo.sh <repo_src_dir>}
REPO=${REPO:-/repo}
REMOTE=${REMOTE:-/srv/git/logslice.git}
CONTRIBUTING=${CONTRIBUTING:-0}

commit() { # <snapshot> <author name> <author email> <date> <message>
  cp -r "$SRC/$1/." "$REPO/"
  git -C "$REPO" add -A
  GIT_AUTHOR_NAME="$2" GIT_AUTHOR_EMAIL="$3" GIT_AUTHOR_DATE="$4" \
  GIT_COMMITTER_NAME="$2" GIT_COMMITTER_EMAIL="$3" GIT_COMMITTER_DATE="$4" \
    git -C "$REPO" commit -q -m "$5"
}

PRIYA=("Priya Raman" "priya.raman@acme.dev")
MARTA=("Marta Kowalski" "marta.kowalski@acme.dev")

mkdir -p "$REPO"
git -C "$REPO" init -q -b main

commit 01-initial "${PRIYA[@]}" "2024-04-02T10:12:00+00:00" "Initial logslice: parse and filter structured logs"
commit 02-count-by "${PRIYA[@]}" "2024-04-18T15:40:00+00:00" "Add --count-by summaries"
if [ "$CONTRIBUTING" = "1" ]; then
  commit 03-contributing "${PRIYA[@]}" "2024-05-03T09:05:00+00:00" "Add contributing guide"
fi

git -C "$REPO" checkout -q -b feature/result-cache
commit 04-cache-module "${MARTA[@]}" "2024-05-06T11:20:00+00:00" "Add on-disk cache for parsed records"
commit 05-cache-cli "${MARTA[@]}" "2024-05-06T11:48:00+00:00" "Add --cache flag to the CLI"

mkdir -p "$(dirname "$REMOTE")"
git init -q --bare -b main "$REMOTE"
git -C "$REPO" remote add origin "$REMOTE"
git -C "$REPO" push -q origin main
git -C "$REPO" push -q -u origin feature/result-cache
