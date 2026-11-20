import json

import pytest

from logslice.cli import main

SAMPLE = """\
2024-05-01T12:00:00Z INFO [http] GET /index.html 200
2024-05-01T12:00:01Z DEBUG [db] query took 3ms
2024-05-01T12:00:02Z WARN [http] slow response /search 1200ms
this line is not a log record
2024-05-01T13:15:00Z ERROR [db] connection reset by peer
2024-05-02T08:00:00Z INFO [auth] user alice logged in
"""


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "app.log"
    path.write_text(SAMPLE)
    return path


def test_prints_matching_records(log_file, capsys):
    assert main([str(log_file), "--level", "error"]) == 0
    assert capsys.readouterr().out == "2024-05-01T13:15:00Z ERROR [db] connection reset by peer\n"


def test_json_output(log_file, capsys):
    main([str(log_file), "-c", "auth", "--json"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows == [
        {
            "timestamp": "2024-05-02T08:00:00Z",
            "level": "INFO",
            "component": "auth",
            "message": "user alice logged in",
            "line_no": 6,
        }
    ]


def test_count_by_component(log_file, capsys):
    main([str(log_file), "--count-by", "component"])
    assert capsys.readouterr().out == "db    2\nhttp  2\nauth  1\n"


def test_count_by_json(log_file, capsys):
    main([str(log_file), "--count-by", "level", "--json"])
    assert json.loads(capsys.readouterr().out) == {"INFO": 2, "DEBUG": 1, "ERROR": 1, "WARNING": 1}


def test_missing_file_exit_code(tmp_path, capsys):
    assert main([str(tmp_path / "nope.log")]) == 2
    assert "nope.log" in capsys.readouterr().err
