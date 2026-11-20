#!/usr/bin/env bash
# Laptop <-> GPU node sync via tar over `ssh a4 srun --overlap` (compute nodes are not directly reachable).
#   harness/sync.sh push [paths]    # project files (default: task/ harness/ README.md) -> node
#   harness/sync.sh pull [variant]  # node trials/ (or trials/<variant>) + logs/ -> laptop
set -euo pipefail
JOBID=${JOBID:-68480301}
REMOTE=${REMOTE:-/home/AMD/absheikh/pr-description-eval}
LOCAL=$(cd "$(dirname "$0")/.." && pwd)
SRUN="srun --jobid=$JOBID --overlap -n1"

case "${1:-}" in
  push)
    shift
    if [ $# -gt 0 ]; then items="$*"; else items=$(cd "$LOCAL" && ls -d task harness README.md 2>/dev/null || true); fi
    ssh -o BatchMode=yes a4 "$SRUN mkdir -p $REMOTE" </dev/null
    # shellcheck disable=SC2086
    tar -C "$LOCAL" -cz --exclude=__pycache__ --exclude=.venv --exclude=.pytest_cache $items \
      | ssh -o BatchMode=yes a4 "$SRUN --chdir=$REMOTE tar xz"
    echo "pushed [$(echo $items)] to $REMOTE"
    ;;
  pull)
    what=trials${2:+/$2}
    mkdir -p "$LOCAL/trials"
    ssh -o BatchMode=yes a4 "$SRUN --chdir=$REMOTE tar cz $what logs" </dev/null | tar xz -C "$LOCAL"
    echo "pulled $what and logs/ into $LOCAL"
    ;;
  *) echo "usage: $0 push [paths] | pull [variant]" >&2; exit 2 ;;
esac
