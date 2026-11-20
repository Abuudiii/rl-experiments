"""Record filters used by the CLI."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Iterator

from .parser import LEVELS, Record, normalize_level


def level_rank(level: str) -> int:
    return LEVELS.index(normalize_level(level))


def filter_records(
    records: Iterable[Record],
    *,
    min_level: str | None = None,
    components: Iterable[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    pattern: str | None = None,
) -> Iterator[Record]:
    """Yield the records that pass every given filter.

    ``since`` is inclusive and ``until`` is exclusive. ``pattern`` is a regular
    expression searched for in the message.
    """
    threshold = level_rank(min_level) if min_level else None
    wanted = set(components) if components else None
    regex = re.compile(pattern) if pattern else None

    for record in records:
        if threshold is not None and level_rank(record.level) < threshold:
            continue
        if wanted is not None and record.component not in wanted:
            continue
        if since is not None or until is not None:
            when = record.time
            if since is not None and when < since:
                continue
            if until is not None and when >= until:
                continue
        if regex is not None and not regex.search(record.message):
            continue
        yield record
