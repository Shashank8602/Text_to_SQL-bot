"""Dataset management from the command line (the Streamlit upload page does the same).

    python -m app.datasets.cli load sample_data/pagila/pagila.sql.gz --name Pagila
    python -m app.datasets.cli load mydb.sql --name "My Shop" --hide customers.phone --expose users.password_hint
    python -m app.datasets.cli load customers.csv orders.csv --name "My Shop" --yes-relationships
    python -m app.datasets.cli load book.xlsx --name Sales
    python -m app.datasets.cli describe pagila --auto
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

    load = sub.add_parser("load", help="load a plain-SQL dump (.sql/.sql.gz), or CSV / Excel files")
    load.add_argument("files", type=Path, nargs="+", metavar="file")
    load.add_argument("--yes-relationships", action="store_true",
                      help="CSV/Excel: accept every detected relationship without asking")
    load.add_argument("--no-relationships", action="store_true", help="CSV/Excel: skip relationship detection")
    load.add_argument("--type", action="append", default=[], metavar="TABLE.COLUMN=TYPE",
                      help="CSV/Excel: override an inferred column type (repeatable)")
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
    describe.add_argument("file", type=Path, nargs="?", help="YAML file; omit with --auto")
    describe.add_argument("--auto", action="store_true", help="draft descriptions with the LLM")

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
            return _describe(args.slug, args.file, auto=args.auto)
        if args.command == "index":
            return _index(args.slug, args.embedder)
        delete_dataset(args.slug)
        print(f"Deleted dataset '{args.slug}'.")
        return 0
    except (DatasetLoadError, DatasetExists, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _load(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    is_dump = all(f.name.lower().endswith((".sql", ".sql.gz", ".dump")) for f in args.files)
    if is_dump:
        if len(args.files) != 1:
            raise ValueError("Load one dump at a time.")
        print(f"Loading {args.files[0]} ...")
        report = load_pg_dump(args.files[0], args.name, slug=args.slug, hide=args.hide, expose=args.expose)
    else:
        report = _load_tabular(args)
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


def _load_tabular(args: argparse.Namespace):
    from app.datasets.relationships import detect
    from app.datasets.tabular_loader import load_tabular, parse_files

    print("Reading " + ", ".join(f.name for f in args.files) + " ...")
    tables = parse_files(args.files)
    for t in tables:
        print(f"  {t.name}: {len(t.rows)} rows; " + ", ".join(f"{c.name} {c.type}" for c in t.columns))
    overrides = {}
    for item in args.type:
        key, _, typ = item.partition("=")
        overrides[key] = typ
    accepted = []
    candidates = [] if args.no_relationships else detect(tables)
    for rel in candidates:
        if args.yes_relationships or (sys.stdin.isatty() and input(f"Relationship {rel.label()} [Y/n]? ").strip().lower() in ("", "y", "yes")):
            accepted.append(rel)
    if candidates:
        print(f"Relationships applied: {len(accepted)} of {len(candidates)} detected"
              + ("" if args.yes_relationships or sys.stdin.isatty() else " (use --yes-relationships to accept)"))
    source = "excel" if any(f.suffix.lower() == ".xlsx" for f in args.files) else "csv"
    return load_tabular(tables, args.name, slug=args.slug, type_overrides=overrides, relationships=accepted,
                        hide=args.hide, expose=args.expose, source_type=source)


def _describe(slug: str, path: Path | None, auto: bool = False) -> int:
    ds = registry.get(slug)
    if ds is None:
        raise ValueError(f"No dataset named '{slug}'.")
    if auto:
        from app.datasets.describer import describe_and_save
        desc = describe_and_save(slug)
        print(f"Drafted descriptions for '{slug}': {len(desc.tables)} tables, "
              f"{sum(len(c) for c in desc.columns.values())} columns, {len(desc.glossary)} glossary terms. "
              "Review them in the UI or replace them with a YAML file.")
        return _index(slug, None)
    if path is None:
        raise ValueError("Give a YAML file, or use --auto.")
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
