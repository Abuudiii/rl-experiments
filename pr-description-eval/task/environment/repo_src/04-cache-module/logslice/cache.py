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

CACHE_DIR = Path.home() / ".cache" / "logslice"
FORMAT_VERSION = 1


def _key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _entry_path(path: Path, directory: Path) -> Path:
    return directory / f"{_key(path)}.json"


def load(path: str | Path, directory: str | Path | None = None) -> list[Record] | None:
    """Return the cached records for ``path``, or ``None`` on a cache miss.

    Entries never expire: the key changes whenever the file is modified, so a
    stale entry is simply never looked up again.
    """
    entry = _entry_path(Path(path), Path(directory) if directory else CACHE_DIR)
    try:
        data = json.loads(entry.read_text())
    except (OSError, ValueError):
        return None
    if data.get("version") != FORMAT_VERSION:
        return None
    return [Record.from_dict(item) for item in data["records"]]


def store(path: str | Path, records: list[Record], directory: str | Path | None = None) -> Path:
    """Write ``records`` to the cache entry for ``path`` and return its location."""
    directory = Path(directory) if directory else CACHE_DIR
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
