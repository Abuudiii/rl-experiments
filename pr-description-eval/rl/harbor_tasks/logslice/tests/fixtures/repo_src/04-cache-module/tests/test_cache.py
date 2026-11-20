import json

from logslice import cache
from logslice.parser import parse_file

SAMPLE = """\
2024-05-01T12:00:00Z INFO [http] GET /index.html 200
2024-05-01T12:00:05Z ERROR [db] connection reset by peer
"""


def make_log(tmp_path, text=SAMPLE):
    path = tmp_path / "app.log"
    path.write_text(text)
    return path


def test_round_trip(tmp_path):
    log = make_log(tmp_path)
    records = parse_file(log)
    entry = cache.store(log, records, directory=tmp_path / "cache")
    assert entry.parent == tmp_path / "cache"
    assert cache.load(log, directory=tmp_path / "cache") == records


def test_miss_on_empty_cache(tmp_path):
    log = make_log(tmp_path)
    assert cache.load(log, directory=tmp_path / "cache") is None


def test_key_changes_when_file_changes(tmp_path):
    log = make_log(tmp_path)
    before = cache._key(log)
    with open(log, "a") as handle:
        handle.write("2024-05-01T12:00:09Z INFO [http] GET /about 200\n")
    assert cache._key(log) != before


def test_modified_file_is_a_miss(tmp_path):
    log = make_log(tmp_path)
    cache.store(log, parse_file(log), directory=tmp_path / "cache")
    with open(log, "a") as handle:
        handle.write("2024-05-01T12:00:09Z INFO [http] GET /about 200\n")
    assert cache.load(log, directory=tmp_path / "cache") is None


def test_corrupt_entry_is_a_miss(tmp_path):
    log = make_log(tmp_path)
    entry = cache.store(log, parse_file(log), directory=tmp_path / "cache")
    entry.write_text("{not json")
    assert cache.load(log, directory=tmp_path / "cache") is None


def test_entry_records_metadata(tmp_path):
    log = make_log(tmp_path)
    entry = cache.store(log, parse_file(log), directory=tmp_path / "cache")
    data = json.loads(entry.read_text())
    assert data["version"] == cache.FORMAT_VERSION
    assert data["source"] == str(log.resolve())
    assert isinstance(data["created_at"], float)
