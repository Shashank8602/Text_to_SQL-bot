"""Load parsed CSV / Excel tables as a new dataset (docs/DESIGN.md section 15.3).

Same hardening as dump loading: own database ds_<slug>, created by data_loader, sensitive columns
hidden, bot_reader granted SELECT on the exposed columns only. Any failure leaves nothing behind.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import psycopg
from psycopg import sql

from app.config import get_settings
from app.database.connection import conninfo
from app.datasets import registry
from app.datasets.loader import (
    DatasetLoadError, LoadReport, _apply_visibility, _create_database, _drop_database, _harden_and_grant,
    _inspect, slugify,
)
from app.datasets.relationships import Relationship
from app.datasets.tabular import PG_TYPES, ParsedTable, TabularError, coerces, parse_file, unique_table_names

MAX_TABLES = 50


def parse_files(paths: Sequence[Path]) -> list[ParsedTable]:
    """Parse every file; sheets of a workbook become separate tables. Table names are made unique."""
    settings = get_settings()
    tables: list[ParsedTable] = []
    for p in paths:
        tables += parse_file(Path(p), max_bytes=settings.max_upload_mb * 1024 * 1024)
    if not tables:
        raise TabularError("No tables found.")
    if len(tables) > MAX_TABLES:
        raise TabularError(f"More than {MAX_TABLES} tables in one dataset.")
    unique_table_names(tables)
    return tables


def load_tabular(
    tables: list[ParsedTable],
    name: str,
    *,
    slug: str | None = None,
    type_overrides: dict[str, str] | None = None,
    relationships: Sequence[Relationship] = (),
    hide: list[str] = (),
    expose: list[str] = (),
    source_type: str = "csv",
) -> LoadReport:
    """type_overrides: {'table.column': 'integer', ...} applied before loading."""
    for key, t in (type_overrides or {}).items():
        table_name, _, col_name = key.partition(".")
        table = next((x for x in tables if x.name == table_name), None)
        if table is None or col_name not in [c.name for c in table.columns]:
            raise DatasetLoadError(f"Type override for unknown column '{key}'.")
        if t not in PG_TYPES:
            raise DatasetLoadError(f"Unsupported type '{t}' for {key}. Use one of: {', '.join(PG_TYPES)}.")
        if not coerces(table.values(col_name), t):
            raise DatasetLoadError(f"Some values in {key} cannot be stored as {t}.")
        table.column(col_name).type = t

    slug = slug or slugify(name)
    db_name = f"ds_{slug}"
    registry.reserve(slug=slug, name=name, db_name=db_name, source_type=source_type,
                     source_file=", ".join(sorted({t.source.split(" / ")[0] for t in tables}))[:250])
    try:
        _create_database(db_name)
        with psycopg.connect(conninfo("data_loader", db_name)) as conn:
            for t in tables:
                _create_and_fill(conn, t)
            for rel in relationships:
                _apply_relationship(conn, rel)
            relations = _inspect(conn)
            _apply_visibility(relations, hide, expose)
            if not any(not c.hidden for r in relations for c in r.columns):
                raise DatasetLoadError("Every column was hidden; the bot would have nothing to query.")
            _harden_and_grant(conn, db_name, relations)
            conn.execute("ANALYZE")
            conn.commit()
        report = LoadReport(slug, db_name, relations, statements_run=len(tables), statements_skipped=0)
        registry.mark_ready(slug, hidden_columns=report.hidden_columns)
        return report
    except BaseException:
        _drop_database(db_name)
        registry.remove(slug)
        raise


def _create_and_fill(conn: psycopg.Connection, t: ParsedTable) -> None:
    ident = sql.Identifier(t.name)
    cols = sql.SQL(", ").join(sql.SQL("{} {}").format(sql.Identifier(c.name), sql.SQL(c.type))
                              for c in t.columns)
    try:
        conn.execute(sql.SQL("CREATE TABLE {} ({})").format(ident, cols))
        with conn.cursor() as cur, cur.copy(
            sql.SQL("COPY {} ({}) FROM STDIN").format(
                ident, sql.SQL(", ").join(sql.Identifier(c.name) for c in t.columns))
        ) as copy:
            for row in t.rows:
                copy.write_row(row)
    except psycopg.Error as e:
        raise DatasetLoadError(f"Could not load {t.source}: {e.diag.message_primary or e}") from e


def _apply_relationship(conn: psycopg.Connection, rel: Relationship) -> None:
    """Make the target column a key and add a foreign key. NOT VALID keeps the constraint (which the
    catalog reads) without failing on the few source rows that have no match."""
    tgt = sql.Identifier(rel.to_table)
    try:
        conn.execute(sql.SQL("ALTER TABLE {} ADD UNIQUE ({})").format(tgt, sql.Identifier(rel.to_column)))
        conn.execute(sql.SQL("ALTER TABLE {} ADD FOREIGN KEY ({}) REFERENCES {} ({}) NOT VALID").format(
            sql.Identifier(rel.from_table), sql.Identifier(rel.from_column), tgt, sql.Identifier(rel.to_column)))
    except psycopg.Error as e:
        raise DatasetLoadError(f"Could not apply relationship {rel.label()}: {e.diag.message_primary or e}") from e
