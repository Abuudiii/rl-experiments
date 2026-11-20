"""Aggregations over parsed records."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .parser import Record

FIELDS = ("level", "component", "hour", "day")


def _field_value(record: Record, field: str) -> str:
    if field == "level":
        return record.level
    if field == "component":
        return record.component
    if field == "hour":
        return record.time.strftime("%Y-%m-%dT%H:00Z")
    if field == "day":
        return record.time.strftime("%Y-%m-%d")
    raise ValueError(f"unknown field: {field!r} (expected one of {', '.join(FIELDS)})")


def count_by(records: Iterable[Record], field: str) -> list[tuple[str, int]]:
    """Count records grouped by ``field``, most common first (ties by key)."""
    counts = Counter(_field_value(record, field) for record in records)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def format_counts(counts: list[tuple[str, int]]) -> str:
    if not counts:
        return "(no matching records)"
    width = max(len(key) for key, _ in counts)
    return "\n".join(f"{key:<{width}}  {count}" for key, count in counts)
