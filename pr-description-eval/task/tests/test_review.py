"""Hidden tests for PR #12 review follow-up (never shipped in the task image).

Run against a candidate checkout of logslice whose root is $LOGSLICE_REPO:

    LOGSLICE_REPO=/path/to/repo PYTHONPATH=/path/to/repo pytest -p no:cacheprovider test_review.py

Most checks drive the CLI in a subprocess (``python -m logslice``) with a
controlled environment, so they don't depend on implementation details. Cache
entries are located by globbing ``*.json`` in the expected directory; expiry is
simulated by back-dating each entry's ``created_at`` field (which the PR
already writes) *and* its file mtime, so either implementation strategy passes.
Cache hits are detected by tampering with the cached records.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(os.environ.get("LOGSLICE_REPO", "/repo")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SAMPLE = """\
2024-05-01T12:00:00Z INFO [http] GET /index.html 200
2024-05-01T12:00:01Z DEBUG [db] query took 3ms
2024-05-01T12:00:02Z WARN [http] slow response /search 1200ms
2024-05-01T13:15:00Z ERROR [db] connection reset by peer
2024-05-02T08:00:00Z INFO [auth] user alice logged in
"""
MARKER = "TAMPERED-BY-HIDDEN-TEST"


# ----------------------------------------------------------------------------- helpers

@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "logs" / "app.log"
    path.parent.mkdir()
    path.write_text(SAMPLE)
    return path


@pytest.fixture
def home(tmp_path):
    path = tmp_path / "home"
    path.mkdir()
    return path


def run_cli(args, home, extra_env=None):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(home),
    }
    env.update(extra_env or {})
    result = subprocess.run(
        [sys.executable, "-m", "logslice", *map(str, args)],
        capture_output=True, text=True, env=env, cwd=str(home), timeout=60,
    )
    return result


def entries(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.json") if p.is_file()) if directory.exists() else []


def all_cache_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix == ".json"]


def doctor(directory: Path, *, age: float, tamper: bool = True) -> None:
    """Back-date every cache entry by ``age`` seconds and optionally tamper with its records."""
    found = entries(directory)
    assert found, f"no cache entries found under {directory}"
    stamp = time.time() - age
    for entry in found:
        data = json.loads(entry.read_text())
        assert "created_at" in data, "cache entries must keep their created_at timestamp"
        data["created_at"] = stamp
        if tamper:
            for record in data["records"]:
                record["message"] = f"{MARKER} {record['message']}"
        entry.write_text(json.dumps(data))
        os.utime(entry, (stamp, stamp))


def cache_module():
    import logslice.cache

    return importlib.reload(logslice.cache)


# ----------------------------------------------------------------------------- sanity

def test_candidate_repo_is_imported():
    import logslice

    assert Path(logslice.__file__).resolve().is_relative_to(REPO), logslice.__file__


# ----------------------------------------------------------------------------- original behaviour

def test_filters_and_summaries_still_work(log_file, home):
    result = run_cli([log_file, "--level", "warning"], home)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "2024-05-01T12:00:02Z WARNING [http] slow response /search 1200ms",
        "2024-05-01T13:15:00Z ERROR [db] connection reset by peer",
    ]
    result = run_cli([log_file, "--count-by", "component", "--json"], home)
    assert json.loads(result.stdout) == {"db": 2, "http": 2, "auth": 1}


def test_no_cache_written_without_flag(log_file, home, tmp_path):
    custom = tmp_path / "custom"
    result = run_cli([log_file], home, {"LOGSLICE_CACHE_DIR": str(custom), "XDG_CACHE_HOME": str(tmp_path / "xdg")})
    assert result.returncode == 0, result.stderr
    assert not all_cache_files(tmp_path / "custom") and not all_cache_files(tmp_path / "xdg")
    assert not all_cache_files(home)


def test_cache_hit_still_served_from_default_location(log_file, home):
    first = run_cli([log_file, "--cache"], home)
    assert first.returncode == 0, first.stderr
    default_dir = home / ".cache" / "logslice"
    assert entries(default_dir), "with no env vars set, the cache should live in ~/.cache/logslice"
    doctor(default_dir, age=5)
    second = run_cli([log_file, "--cache"], home)
    assert second.returncode == 0, second.stderr
    assert MARKER in second.stdout, "a fresh cache entry should be used"


def test_changed_file_is_still_a_miss(log_file, home):
    run_cli([log_file, "--cache"], home)
    doctor(home / ".cache" / "logslice", age=5)
    with open(log_file, "a") as handle:
        handle.write("2024-05-02T09:00:00Z INFO [auth] user bob logged in\n")
    result = run_cli([log_file, "--cache"], home)
    assert MARKER not in result.stdout
    assert "user bob logged in" in result.stdout


# ----------------------------------------------------------------------------- (a) configurable cache dir

def test_logslice_cache_dir_env_var_used_as_is(log_file, home, tmp_path):
    custom = tmp_path / "custom-cache"
    result = run_cli([log_file, "--cache"], home, {"LOGSLICE_CACHE_DIR": str(custom)})
    assert result.returncode == 0, result.stderr
    assert [p.parent for p in entries(custom)] == [custom], "entries go directly into $LOGSLICE_CACHE_DIR"
    assert not all_cache_files(home)
    doctor(custom, age=5)
    again = run_cli([log_file, "--cache"], home, {"LOGSLICE_CACHE_DIR": str(custom)})
    assert MARKER in again.stdout, "the cache in $LOGSLICE_CACHE_DIR should be read back"


def test_xdg_cache_home_default(log_file, home, tmp_path):
    xdg = tmp_path / "xdg"
    result = run_cli([log_file, "--cache"], home, {"XDG_CACHE_HOME": str(xdg)})
    assert result.returncode == 0, result.stderr
    assert entries(xdg / "logslice")
    assert not all_cache_files(home)


def test_env_var_wins_over_xdg(log_file, home, tmp_path):
    custom, xdg = tmp_path / "custom", tmp_path / "xdg"
    run_cli([log_file, "--cache"], home, {"LOGSLICE_CACHE_DIR": str(custom), "XDG_CACHE_HOME": str(xdg)})
    assert entries(custom)
    assert not all_cache_files(xdg)


def test_empty_env_vars_are_ignored(log_file, home):
    run_cli([log_file, "--cache"], home, {"LOGSLICE_CACHE_DIR": "", "XDG_CACHE_HOME": ""})
    assert entries(home / ".cache" / "logslice")


def test_directory_resolved_at_call_time(log_file, tmp_path, monkeypatch):
    cache = cache_module()
    from logslice.parser import parse_file

    monkeypatch.setenv("LOGSLICE_CACHE_DIR", str(tmp_path / "late"))
    records = parse_file(log_file)
    cache.store(log_file, records)
    assert entries(tmp_path / "late"), "LOGSLICE_CACHE_DIR set after import must be honoured"
    assert cache.load(log_file) == records


def test_explicit_directory_argument_wins(log_file, tmp_path, monkeypatch):
    cache = cache_module()
    from logslice.parser import parse_file

    monkeypatch.setenv("LOGSLICE_CACHE_DIR", str(tmp_path / "env"))
    cache.store(log_file, parse_file(log_file), directory=tmp_path / "explicit")
    assert entries(tmp_path / "explicit")
    assert not entries(tmp_path / "env")


# ----------------------------------------------------------------------------- (b) TTL

def test_default_ttl_expires_old_entries(log_file, home, tmp_path):
    env = {"LOGSLICE_CACHE_DIR": str(tmp_path / "c")}
    run_cli([log_file, "--cache"], home, env)
    doctor(tmp_path / "c", age=3600 + 300)
    result = run_cli([log_file, "--cache"], home, env)
    assert result.returncode == 0, result.stderr
    assert MARKER not in result.stdout, "entries older than the default TTL (3600 s) must be a miss"


def test_default_ttl_keeps_recent_entries(log_file, home, tmp_path):
    env = {"LOGSLICE_CACHE_DIR": str(tmp_path / "c")}
    run_cli([log_file, "--cache"], home, env)
    doctor(tmp_path / "c", age=3600 - 300)
    result = run_cli([log_file, "--cache"], home, env)
    assert MARKER in result.stdout, "entries younger than the default TTL must still be used"


def test_cache_ttl_option(log_file, home, tmp_path):
    env = {"LOGSLICE_CACHE_DIR": str(tmp_path / "c")}
    run_cli([log_file, "--cache"], home, env)
    doctor(tmp_path / "c", age=7200)
    long_ttl = run_cli([log_file, "--cache", "--cache-ttl", "86400"], home, env)
    assert long_ttl.returncode == 0, long_ttl.stderr
    assert MARKER in long_ttl.stdout

    doctor(tmp_path / "c", age=120, tamper=False)
    short_ttl = run_cli([log_file, "--cache", "--cache-ttl", "60"], home, env)
    assert short_ttl.returncode == 0, short_ttl.stderr
    assert MARKER not in short_ttl.stdout


def test_expired_entry_is_rewritten(log_file, home, tmp_path):
    env = {"LOGSLICE_CACHE_DIR": str(tmp_path / "c")}
    run_cli([log_file, "--cache"], home, env)
    doctor(tmp_path / "c", age=7200)
    before = time.time()
    run_cli([log_file, "--cache"], home, env)
    fresh = [json.loads(p.read_text()) for p in entries(tmp_path / "c")]
    assert any(d.get("created_at", 0) >= before - 5 for d in fresh), "a miss should store a fresh entry"
    assert all(MARKER not in r["message"] for d in fresh if d.get("created_at", 0) >= before - 5 for r in d["records"])


def test_load_accepts_ttl_keyword(log_file, tmp_path):
    cache = cache_module()
    from logslice.parser import parse_file

    records = parse_file(log_file)
    directory = tmp_path / "c"
    cache.store(log_file, records, directory=directory)
    doctor(directory, age=120, tamper=False)
    assert cache.load(log_file, directory=directory, ttl=60) is None
    assert cache.load(log_file, directory=directory, ttl=600) == records
    doctor(directory, age=7200, tamper=False)
    assert cache.load(log_file, directory=directory) is None, "load() should default to a 3600 s TTL"


# ----------------------------------------------------------------------------- (c) rename nit

def test_cache_key_is_public_and_documented(log_file):
    cache = cache_module()
    assert callable(getattr(cache, "cache_key", None)), "_key should be renamed to cache_key"
    doc = (cache.cache_key.__doc__ or "").lower()
    assert len(doc.strip()) >= 20, "cache_key needs a docstring"
    assert "size" in doc and ("mtime" in doc or "modif" in doc) and "path" in doc, \
        "docstring should say what goes into the key"


def test_cache_key_behaviour(log_file):
    cache = cache_module()
    first = cache.cache_key(Path(log_file))
    assert first == cache.cache_key(Path(log_file))
    with open(log_file, "a") as handle:
        handle.write("2024-05-02T09:00:00Z INFO [auth] more\n")
    assert cache.cache_key(Path(log_file)) != first
