"""Dataset management from the command line (the Streamlit upload wizard comes in M7).

    python -m app.datasets.cli load sample_data/pagila/pagila.sql.gz --name Pagila
    python -m app.datasets.cli load mydb.sql --name "My Shop" --hide customers.phone --expose users.password_hint
    python -m app.datasets.cli list
    python -m app.datasets.cli delete pagila
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.datasets import registry
from app.datasets.loader import DatasetLoadError, delete_dataset, load_pg_dump
from app.datasets.registry import DatasetExists


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.datasets.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    load = sub.add_parser("load", help="load a plain-SQL PostgreSQL dump (.sql or .sql.gz)")
    load.add_argument("file", type=Path)
    load.add_argument("--name", required=True, help="display name, e.g. 'Pagila'")
    load.add_argument("--slug", help="dataset id (default: derived from --name)")
    load.add_argument("--hide", action="append", default=[], metavar="TABLE[.COLUMN]",
                      help="hide a table or column from the bot (repeatable)")
    load.add_argument("--expose", action="append", default=[], metavar="TABLE.COLUMN",
                      help="expose a column hidden by the default rules (repeatable)")

    sub.add_parser("list", help="list datasets")
    delete = sub.add_parser("delete", help="delete a dataset and its database")
    delete.add_argument("slug")

    args = parser.parse_args(argv)
    try:
        if args.command == "load":
            return _load(args)
        if args.command == "list":
            return _list()
        delete_dataset(args.slug)
        print(f"Deleted dataset '{args.slug}'.")
        return 0
    except (DatasetLoadError, DatasetExists, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _load(args: argparse.Namespace) -> int:
    print(f"Loading {args.file} ...")
    start = time.perf_counter()
    report = load_pg_dump(args.file, args.name, slug=args.slug, hide=args.hide, expose=args.expose)
    elapsed = time.perf_counter() - start

    visible = [r for r in report.relations if any(not c.hidden for c in r.columns)]
    print(f"Loaded dataset '{report.slug}' into database {report.db_name} in {elapsed:.1f}s "
          f"({report.statements_run} statements run, {report.statements_skipped} skipped).")
    print(f"\nVisible to the bot ({len(visible)}):")
    for r in visible:
        print(f"  {r.kind:<17} {r.qualified}")
    hidden = [(r, c) for r in report.relations for c in r.columns if c.hidden]
    if hidden:
        print(f"\nHidden from the bot ({len(hidden)} columns; use --expose to override):")
        for r, c in hidden:
            print(f"  {r.qualified + '.' + c.name:<35} ({c.hidden_reason})")
    return 0


def _list() -> int:
    datasets = registry.list_all()
    if not datasets:
        print("No datasets loaded. Try: python -m app.datasets.cli load sample_data/pagila/pagila.sql.gz --name Pagila")
        return 0
    for d in datasets:
        print(f"{d.slug:<20} {d.status:<8} {d.name:<25} {d.source_file or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
