from logslice.filters import filter_records
from logslice.parser import parse_lines, parse_timestamp

SAMPLE = """\
2024-05-01T12:00:00Z INFO [http] GET /index.html 200
2024-05-01T12:00:01Z DEBUG [db] query took 3ms
2024-05-01T12:00:02Z WARN [http] slow response /search 1200ms
this line is not a log record
2024-05-01T13:15:00Z ERROR [db] connection reset by peer
2024-05-02T08:00:00Z INFO [auth] user alice logged in
"""

RECORDS = list(parse_lines(SAMPLE.splitlines()))


def messages(records):
    return [r.message for r in records]


def test_no_filters_keeps_everything():
    assert list(filter_records(RECORDS)) == RECORDS


def test_min_level():
    kept = list(filter_records(RECORDS, min_level="warning"))
    assert [r.level for r in kept] == ["WARNING", "ERROR"]


def test_components():
    kept = list(filter_records(RECORDS, components=["db"]))
    assert {r.component for r in kept} == {"db"}
    assert len(kept) == 2


def test_time_window_is_half_open():
    kept = list(
        filter_records(
            RECORDS,
            since=parse_timestamp("2024-05-01T12:00:01Z"),
            until=parse_timestamp("2024-05-01T13:15:00Z"),
        )
    )
    assert messages(kept) == ["query took 3ms", "slow response /search 1200ms"]


def test_pattern():
    assert messages(filter_records(RECORDS, pattern=r"\d+ms")) == [
        "query took 3ms",
        "slow response /search 1200ms",
    ]
