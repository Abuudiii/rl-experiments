from datetime import datetime, timezone

from logslice.parser import Record, parse_file, parse_line, parse_lines, parse_timestamp

SAMPLE = """\
2024-05-01T12:00:00Z INFO [http] GET /index.html 200
2024-05-01T12:00:01Z DEBUG [db] query took 3ms
2024-05-01T12:00:02Z WARN [http] slow response /search 1200ms
this line is not a log record
2024-05-01T13:15:00Z ERROR [db] connection reset by peer
2024-05-02T08:00:00Z INFO [auth] user alice logged in
"""


def test_parse_line_fields():
    record = parse_line("2024-05-01T12:00:00Z INFO [http] GET /index.html 200", 7)
    assert record == Record("2024-05-01T12:00:00Z", "INFO", "http", "GET /index.html 200", 7)


def test_level_aliases_are_normalised():
    assert parse_line("2024-05-01T12:00:00Z warn [x] hi").level == "WARNING"
    assert parse_line("2024-05-01T12:00:00Z FATAL [x] hi").level == "CRITICAL"


def test_unparseable_lines_are_skipped():
    records = list(parse_lines(SAMPLE.splitlines()))
    assert len(records) == 5
    assert [r.line_no for r in records] == [1, 2, 3, 5, 6]


def test_unknown_level_is_rejected():
    assert parse_line("2024-05-01T12:00:00Z LOUD [x] hi") is None


def test_parse_timestamp_assumes_utc():
    assert parse_timestamp("2024-05-01T12:00:00") == datetime(2024, 5, 1, 12, tzinfo=timezone.utc)
    assert parse_timestamp("2024-05-01T12:00:00Z") == datetime(2024, 5, 1, 12, tzinfo=timezone.utc)


def test_record_round_trips_through_dict():
    record = parse_line("2024-05-01T12:00:00Z INFO [http] ok", 3)
    assert Record.from_dict(record.to_dict()) == record


def test_parse_file(tmp_path):
    log = tmp_path / "app.log"
    log.write_text(SAMPLE)
    assert len(parse_file(log)) == 5
