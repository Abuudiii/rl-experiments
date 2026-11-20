#!/usr/bin/env bash
# Oracle for PR #12: implements all three review comments, updates tests and
# README, commits, pushes, and rewrites the PR description via `gh`.
# Run inside a fresh task container.
#
#   solve.sh               full solution
#   solve.sh --code-only   same code changes, but leaves the PR description alone
#                          (used to validate that the description checks fail)
set -euo pipefail
cd /repo

CODE_ONLY=0
[ "${1:-}" = "--code-only" ] && CODE_ONLY=1

cat > logslice/cache.py <<'PY'
"""On-disk cache of parsed log records.

Parsing is the slow part of logslice on large files, so ``--cache`` stores the
parsed records as JSON and reuses them on the next run over the same file.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .parser import Record

FORMAT_VERSION = 1
DEFAULT_TTL = 3600


def cache_dir() -> Path:
    """Return the cache directory, resolved from the environment on every call.

    ``$LOGSLICE_CACHE_DIR`` if set (used as-is), else ``$XDG_CACHE_HOME/logslice``,
    else ``~/.cache/logslice``.
    """
    override = os.environ.get("LOGSLICE_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "logslice"
    return Path.home() / ".cache" / "logslice"


def cache_key(path: Path) -> str:
    """Return the cache key for the log file at ``path``.

    The key is a SHA-256 of the file's resolved path, its size and its
    modification time (in nanoseconds), so editing, appending to or replacing
    the file produces a new key.
    """
    stat = path.stat()
    raw = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _entry_path(path: Path, directory: Path) -> Path:
    return directory / f"{cache_key(path)}.json"


def load(
    path: str | Path,
    directory: str | Path | None = None,
    *,
    ttl: float = DEFAULT_TTL,
) -> list[Record] | None:
    """Return the cached records for ``path``, or ``None`` on a cache miss.

    An entry is a miss if the file changed since it was stored (the key covers
    size and mtime) or if it was created more than ``ttl`` seconds ago.
    """
    entry = _entry_path(Path(path), Path(directory) if directory else cache_dir())
    try:
        data = json.loads(entry.read_text())
    except (OSError, ValueError):
        return None
    if data.get("version") != FORMAT_VERSION:
        return None
    created_at = data.get("created_at")
    if not isinstance(created_at, (int, float)) or time.time() - created_at > ttl:
        return None
    return [Record.from_dict(item) for item in data["records"]]


def store(path: str | Path, records: list[Record], directory: str | Path | None = None) -> Path:
    """Write ``records`` to the cache entry for ``path`` and return its location."""
    directory = Path(directory) if directory else cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    entry = _entry_path(Path(path), directory)
    data = {
        "version": FORMAT_VERSION,
        "source": str(Path(path).resolve()),
        "created_at": time.time(),
        "records": [record.to_dict() for record in records],
    }
    tmp = entry.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, entry)
    return entry
PY

python3 - <<'PY'
from pathlib import Path

cli = Path("logslice/cli.py")
src = cli.read_text()
old_flag = '''    parser.add_argument(
        "--cache", action="store_true",
        help="reuse parsed records from the on-disk cache (~/.cache/logslice)",
    )
'''
new_flag = '''    parser.add_argument(
        "--cache", action="store_true",
        help="reuse parsed records from the on-disk cache "
        "($LOGSLICE_CACHE_DIR, else $XDG_CACHE_HOME/logslice, else ~/.cache/logslice)",
    )
    parser.add_argument(
        "--cache-ttl", type=_non_negative_int, default=cache.DEFAULT_TTL, metavar="SECONDS",
        help=f"with --cache, treat entries older than SECONDS as stale (default: {cache.DEFAULT_TTL})",
    )
'''
assert old_flag in src
src = src.replace(old_flag, new_flag)
src = src.replace(
    "def build_parser() -> argparse.ArgumentParser:",
    '''def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def build_parser() -> argparse.ArgumentParser:''',
)
assert "records = cache.load(args.path)" in src
src = src.replace("records = cache.load(args.path)", "records = cache.load(args.path, ttl=args.cache_ttl)")
cli.write_text(src)

tests = Path("tests/test_cache.py")
src = tests.read_text().replace("cache._key(", "cache.cache_key(")
src = src.replace("import json\n", "import json\nimport time\n", 1)
src += '''

def age_entry(entry, seconds):
    data = json.loads(entry.read_text())
    data["created_at"] = time.time() - seconds
    entry.write_text(json.dumps(data))


def test_fresh_entry_is_a_hit(tmp_path):
    log = make_log(tmp_path)
    entry = cache.store(log, parse_file(log), directory=tmp_path / "cache")
    age_entry(entry, 60)
    assert cache.load(log, directory=tmp_path / "cache") is not None


def test_entry_older_than_default_ttl_is_a_miss(tmp_path):
    log = make_log(tmp_path)
    entry = cache.store(log, parse_file(log), directory=tmp_path / "cache")
    age_entry(entry, cache.DEFAULT_TTL + 60)
    assert cache.load(log, directory=tmp_path / "cache") is None


def test_custom_ttl(tmp_path):
    log = make_log(tmp_path)
    entry = cache.store(log, parse_file(log), directory=tmp_path / "cache")
    age_entry(entry, 120)
    assert cache.load(log, directory=tmp_path / "cache", ttl=60) is None
    assert cache.load(log, directory=tmp_path / "cache", ttl=600) is not None


def test_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGSLICE_CACHE_DIR", str(tmp_path / "custom"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.cache_dir() == tmp_path / "custom"


def test_cache_dir_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("LOGSLICE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.cache_dir() == tmp_path / "xdg" / "logslice"


def test_cache_dir_home_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("LOGSLICE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache.cache_dir() == tmp_path / ".cache" / "logslice"


def test_store_uses_env_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGSLICE_CACHE_DIR", str(tmp_path / "custom"))
    log = make_log(tmp_path)
    entry = cache.store(log, parse_file(log))
    assert entry.parent == tmp_path / "custom"
    assert cache.load(log) == parse_file(log)
'''
tests.write_text(src)

tests = Path("tests/test_cli.py")
src = tests.read_text()
assert src.count('monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")') == 2
src = src.replace(
    'monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")',
    'monkeypatch.setenv("LOGSLICE_CACHE_DIR", str(tmp_path / "cache"))',
)
src = src.replace("from logslice import cache\n", "")
src += '''

def test_cache_ttl_expires_entries(log_file, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOGSLICE_CACHE_DIR", str(tmp_path / "cache"))
    main([str(log_file), "--cache"])
    (entry,) = (tmp_path / "cache").glob("*.json")
    data = json.loads(entry.read_text())
    data["created_at"] -= 120
    data["records"][0]["message"] = "served from cache"
    entry.write_text(json.dumps(data))
    capsys.readouterr()

    main([str(log_file), "--cache", "--cache-ttl", "600", "-c", "http"])
    assert "served from cache" in capsys.readouterr().out

    main([str(log_file), "--cache", "--cache-ttl", "60", "-c", "http"])
    assert "served from cache" not in capsys.readouterr().out
    assert json.loads(entry.read_text())["records"][0]["message"] == "GET /index.html 200"
'''
tests.write_text(src)

readme = Path("README.md")
src = readme.read_text()
old = '''Cached results are stored as JSON in `~/.cache/logslice`, one entry per log
file. The cache key includes the file's size and modification time, so editing
or appending to a log invalidates its entry. Entries never expire; delete the
directory to clear the cache.
'''
new = '''Cached results are stored as JSON, one entry per log file, in the first of:

1. `$LOGSLICE_CACHE_DIR`, if set;
2. `$XDG_CACHE_HOME/logslice`, if `XDG_CACHE_HOME` is set;
3. `~/.cache/logslice`.

The cache key includes the file's size and modification time, so editing or
appending to a log invalidates its entry. Entries also expire after an hour;
use `--cache-ttl SECONDS` to change that. Delete the directory to clear the
cache.
'''
assert old in src
readme.write_text(src.replace(old, new))
PY

python3 -m pytest -q

git add -A
git commit -q -m "Address review: configurable cache dir, cache TTL, rename cache_key

- Resolve the cache directory at call time: \$LOGSLICE_CACHE_DIR, then
  \$XDG_CACHE_HOME/logslice, then ~/.cache/logslice.
- Add --cache-ttl SECONDS (default 3600) and load(ttl=...); expired entries
  are treated as a miss and rewritten.
- Rename _key to cache_key and document what goes into the key."
git push -q

if [ "$CODE_ONLY" = "1" ]; then
  echo "code-only: PR description left unchanged"
  exit 0
fi

cat > /tmp/pr-body.md <<'MD'
## Summary

Parsing is the slow part of `logslice` on big files (the 1.2 GB ingest log from the nightly job takes ~40 s just to parse). This PR adds an opt-in on-disk cache of parsed records, so repeated queries against the same file skip parsing entirely.

## Changes

- New `--cache` flag. When set, parsed records are loaded from the cache if present and fresh; otherwise the file is parsed and the result is saved for next time.
- New `--cache-ttl SECONDS` option (default 3600). Entries whose `created_at` is older than the TTL are treated as a miss: the file is re-parsed and the entry overwritten.
- New module `logslice/cache.py` with `load()` / `store()`, plus `cache_dir()` and `cache_key()`. `load()` takes a `ttl` keyword argument (seconds, default 3600).
- The cache directory is resolved at call time, in this order: `$LOGSLICE_CACHE_DIR` (used as-is), then `$XDG_CACHE_HOME/logslice`, then `~/.cache/logslice`. An explicit `directory=` argument to `load()`/`store()` overrides all of these. Entries are JSON, one file per log file.
- The cache key is computed by `cache_key()` from the file's resolved path, size and mtime, so editing or appending to a log also invalidates its entry.
- `--cache` is ignored (with a warning) when reading from stdin.
- README: new "Caching" section covering the cache location, expiry and `--cache-ttl`.

## Testing

- `pytest`: `tests/test_cache.py` covers the store/load round trip, key changes on modification, corrupt entries, TTL expiry (default and custom), and cache-directory resolution (`LOGSLICE_CACHE_DIR`, `XDG_CACHE_HOME`, home fallback); `tests/test_cli.py` covers the `--cache` hit path, `--cache-ttl` expiry, and that nothing is written without the flag.
- Manually: `logslice --cache --count-by level ingest.log` twice; the second run went from 41 s to 0.4 s.
MD
gh pr edit 12 --body-file /tmp/pr-body.md
gh pr comment 12 --body "Thanks @priya-r, addressed in the latest commit: the cache dir now honours \`LOGSLICE_CACHE_DIR\` / \`XDG_CACHE_HOME\`, entries expire after \`--cache-ttl\` seconds (default 3600), and \`_key\` is now \`cache_key\` with a docstring. I've updated the PR description to match."
