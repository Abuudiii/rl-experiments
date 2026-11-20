"""Parse log lines of the form::

    2024-05-01T12:00:00Z INFO [http] GET /index.html 200

i.e. an ISO-8601 timestamp, a level, a component in square brackets and a
free-form message. Lines that don't match are skipped.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_ALIASES = {"WARN": "WARNING", "ERR": "ERROR", "FATAL": "CRITICAL"}

_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
    r"\s+(?P<level>[A-Za-z]+)"
    r"\s+\[(?P<component>[^\]]+)\]"
    r"\s?(?P<message>.*)$"
)


@dataclass(frozen=True)
class Record:
    timestamp: str
    level: str
    component: str
    message: str
    line_no: int = 0

    @property
    def time(self) -> datetime:
        return parse_timestamp(self.timestamp)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        return cls(**data)

    def format(self) -> str:
        return f"{self.timestamp} {self.level} [{self.component}] {self.message}"


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive values are treated as UTC."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def normalize_level(level: str) -> str:
    level = level.upper()
    return _ALIASES.get(level, level)


def parse_line(line: str, line_no: int = 0) -> Record | None:
    match = _LINE_RE.match(line.rstrip("\r\n"))
    if match is None:
        return None
    level = normalize_level(match["level"])
    if level not in LEVELS:
        return None
    return Record(
        timestamp=match["timestamp"],
        level=level,
        component=match["component"].strip(),
        message=match["message"],
        line_no=line_no,
    )


def parse_lines(lines: Iterable[str]) -> Iterator[Record]:
    for line_no, line in enumerate(lines, start=1):
        record = parse_line(line, line_no)
        if record is not None:
            yield record


def parse_file(path: str | Path) -> list[Record]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return list(parse_lines(handle))
