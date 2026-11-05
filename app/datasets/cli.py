"""Dataset management from the command line (the Streamlit upload wizard comes in M7).

    python -m app.datasets.cli load sample_data/pagila/pagila.sql.gz --name Pagila
    python -m app.datasets.cli load mydb.sql --name "My Shop" --hide customers.phone --expose users.password_hint
    python -m app.datasets.cli describe pagila sample_data/pagila/descriptions.yaml
    python -m app.datasets.cli index pagila --embedder local
    python -m app.datasets.cli list
    python -m app.datasets.cli delete pagila
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.database.catalog import load_catalog
from app.datasets import descriptions as descriptions_store
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
    load.add_argument("--descriptions", type=Path, metavar="YAML",
                      help="business descriptions to import after loading")

    describe = sub.add_parser("describe", help="import business descriptions (YAML) for a dataset")
    describe.add_argument("slug")
    describe.add_argument("file", type=Path)

    index = sub.add_parser("index", help="(re)build the schema-retrieval index of a dataset")
    index.add_argument("slug")
    index.add_argument("--embedder", default=None, help="local | gemini (default: RETRIEVAL_EMBEDDER)")

    sub.add_parser("list", help="list datasets")
    delete = sub.add_parser("delete", help="delete a dataset and its database")
    delete.add_argument("slug")

    args = parser.parse_args(argv)
    try:
        if args.command == "load":
            return _load(args)
        if args.command == "list":
            return _list()
        if args.command == "describe":
            return _describe(args.slug, args.file)
        if args.command == "index":
            return _index(args.slug, args.embedder)
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
    if args.descriptions:
        print()
        return _describe(report.slug, args.descriptions)
    return 0


def _describe(slug: str, path: Path) -> int:
    ds = registry.get(slug)
    if ds is None:
        raise ValueError(f"No dataset named '{slug}'.")
    desc = descriptions_store.parse_yaml(path)
    problems = descriptions_store.validate(desc, load_catalog(ds.db_name))
    if problems:
        raise ValueError("Descriptions don't match the dataset: " + "; ".join(problems))
    descriptions_store.save(slug, desc)
    n_cols = sum(len(c) for c in desc.columns.values())
    print(f"Imported descriptions for '{slug}': {len(desc.tables)} tables, {n_cols} columns, "
          f"{len(desc.glossary)} glossary terms.")
    print("Rebuild the retrieval index to use them: python -m app.datasets.cli index " + slug)
    return 0


def _index(slug: str, embedder: str | None) -> int:
    from app.config import get_settings
    from app.retrieval.indexer import build_index

    name = embedder or get_settings().retrieval_embedder
    if name == "none":
        raise ValueError("RETRIEVAL_EMBEDDER is 'none'; pass --embedder local|gemini.")
    start = time.perf_counter()
    n = build_index(slug, name)
    print(f"Indexed {n} schema documents for '{slug}' with the {name} embedder "
          f"in {time.perf_counter() - start:.1f}s.")
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
