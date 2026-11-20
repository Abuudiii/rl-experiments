"""Command-line interface for logslice."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .filters import filter_records
from .parser import LEVELS, Record, parse_file, parse_lines, parse_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logslice",
        description="Filter and summarise structured log files.",
    )
    parser.add_argument("path", help="log file to read, or - for stdin")
    parser.add_argument(
        "-l", "--level", choices=LEVELS, type=str.upper,
        help="only show records at this level or above",
    )
    parser.add_argument(
        "-c", "--component", action="append", dest="components", metavar="NAME",
        help="only show records from this component (repeatable)",
    )
    parser.add_argument("--since", type=parse_timestamp, help="ISO timestamp (inclusive)")
    parser.add_argument("--until", type=parse_timestamp, help="ISO timestamp (exclusive)")
    parser.add_argument("-g", "--grep", metavar="REGEX", help="regex to search for in the message")
    parser.add_argument("--json", action="store_true", help="emit JSON lines instead of text")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def read_records(args: argparse.Namespace) -> list[Record]:
    if args.path == "-":
        return list(parse_lines(sys.stdin))
    return parse_file(args.path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = read_records(args)
    except OSError as exc:
        print(f"logslice: {exc}", file=sys.stderr)
        return 2

    selected = filter_records(
        records,
        min_level=args.level,
        components=args.components,
        since=args.since,
        until=args.until,
        pattern=args.grep,
    )
    for record in selected:
        if args.json:
            print(json.dumps(record.to_dict()))
        else:
            print(record.format())
    return 0
