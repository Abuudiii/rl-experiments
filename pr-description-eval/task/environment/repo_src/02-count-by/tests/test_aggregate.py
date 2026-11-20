import pytest

from logslice.aggregate import count_by, format_counts
from logslice.parser import parse_lines

SAMPLE = """\
2024-05-01T12:00:00Z INFO [http] GET /index.html 200
2024-05-01T12:00:01Z DEBUG [db] query took 3ms
2024-05-01T12:00:02Z WARN [http] slow response /search 1200ms
this line is not a log record
2024-05-01T13:15:00Z ERROR [db] connection reset by peer
2024-05-02T08:00:00Z INFO [auth] user alice logged in
"""

RECORDS = list(parse_lines(SAMPLE.splitlines()))


def test_count_by_level_most_common_first():
    assert count_by(RECORDS, "level") == [("INFO", 2), ("DEBUG", 1), ("ERROR", 1), ("WARNING", 1)]


def test_count_by_hour():
    assert count_by(RECORDS, "hour")[0] == ("2024-05-01T12:00Z", 3)


def test_count_by_day():
    assert count_by(RECORDS, "day") == [("2024-05-01", 4), ("2024-05-02", 1)]


def test_unknown_field():
    with pytest.raises(ValueError):
        count_by(RECORDS, "colour")


def test_format_counts_aligns_keys():
    assert format_counts([("db", 2), ("http", 1)]) == "db    2\nhttp  1"
    assert format_counts([]) == "(no matching records)"
