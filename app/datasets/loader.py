"""Load a PostgreSQL plain-SQL dump as a new dataset.

Pipeline (docs/DESIGN.md §15):
  create ds_<slug> database -> restore dump as data_loader -> hide sensitive columns
  -> grant bot_reader SELECT on the exposed columns -> mark dataset ready.
Any failure drops the database and removes the registry entry.
"""

from __future__ import annotations

import gzip
import io
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
from psycopg import sql

from app.config import get_settings
from app.database.connection import conninfo
from app.datasets import registry
from app.datasets.dump_parser import DumpRejected, classify, parse_dump, rewrite

# Columns hidden from the bot by default (the user can override per column).
_SENSITIVE_NAME = re.compile(
    r"pass(word|wd)?|pwd|secret|token|api_?key|salt|hash|ssn|social_security"
    r"|credit_?card|card_?number|cvv|iban|private_?key",
    re.IGNORECASE,
)
# Types that are useless to the LLM and expensive to return.
_OPAQUE_TYPES = {"bytea", "vector", "tsvector"}


class DatasetLoadError(Exception):
    pass


@dataclass
class Column:
    name: str
    type: str
    hidden: bool
    hidden_reason: str | None = None


@dataclass
class Relation:
    schema: str
    name: str
    kind: str  # table | view | materialized view
    columns: list[Column] = field(default_factory=list)

    @property
    def qualified(self) -> str:
        return self.name if self.schema == "public" else f"{self.schema}.{self.name}"


@dataclass
class LoadReport:
    slug: str
    db_name: str
    relations: list[Relation]
    statements_run: int
    statements_skipped: int

    @property
    def hidden_columns(self) -> list[str]:
        return [f"{r.qualified}.{c.name}" for r in self.relations for c in r.columns if c.hidden]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"d_{slug}"
    return slug[:40]


def load_pg_dump(
    path: Path,
    name: str,
    *,
    slug: str | None = None,
    hide: list[str] = (),
    expose: list[str] = (),
    restore_timeout_s: int | None = None,
) -> LoadReport:
    """hide / expose: 'table.column' (or 'schema.table.column') overrides of the default
    sensitive-column rules. hide also accepts a bare table name to hide the whole table."""
    settings = get_settings()
    path = Path(path)
    if not path.is_file():
        raise DatasetLoadError(f"File not found: {path}")
    if path.stat().st_size > settings.max_upload_mb * 1024 * 1024:
        raise DatasetLoadError(f"File is larger than {settings.max_upload_mb} MB.")
    if path.suffix.lower() == ".dump" or _is_custom_format(path):
        raise DatasetLoadError(
            "Custom-format dumps are not supported yet. Export a plain SQL dump instead: "
            "pg_dump --format=plain --no-owner --no-privileges -f mydb.sql mydb"
        )

    slug = slug or slugify(name)
    db_name = f"ds_{slug}"
    registry.reserve(slug=slug, name=name, db_name=db_name, source_type="pg_dump", source_file=path.name)
    try:
        _create_database(db_name)
        with psycopg.connect(conninfo("data_loader", db_name)) as conn:
            ran, skipped = _restore(conn, path, restore_timeout_s or settings.restore_timeout_s)
            relations = _inspect(conn)
            _apply_visibility(relations, hide, expose)
            if not any(not c.hidden for r in relations for c in r.columns):
                raise DatasetLoadError("The dump contains no tables the bot could query.")
            _harden_and_grant(conn, db_name, relations)
            conn.commit()
        report = LoadReport(slug, db_name, relations, ran, skipped)
        registry.mark_ready(slug, hidden_columns=report.hidden_columns)
        return report
    except BaseException:
        _drop_database(db_name)
        registry.remove(slug)
        raise


def delete_dataset(slug: str) -> None:
    ds = registry.get(slug)
    if ds is None:
        raise DatasetLoadError(f"No dataset named '{slug}'.")
    _drop_database(ds.db_name)
    registry.remove(slug)


# --------------------------------------------------------------------------- restore

def _restore(conn: psycopg.Connection, path: Path, timeout_s: int) -> tuple[int, int]:
    """Run the dump in ONE transaction (committed by the caller). Returns (run, skipped)."""
    ran = skipped = 0
    # Wall-clock limit for the whole restore; per-statement limit comes from the role.
    watchdog = threading.Timer(timeout_s, conn.cancel_safe)
    watchdog.start()
    try:
        with _open_text(path) as lines, conn.cursor() as cur:
            for stmt in parse_dump(lines):
                verdict, reason = classify(stmt.sql)
                if verdict == "reject":
                    raise DumpRejected(stmt.line, f"{reason}: {_preview(stmt.sql)}")
                if verdict == "skip":
                    skipped += 1
                    continue
                try:
                    if stmt.is_copy:
                        with cur.copy(stmt.sql) as copy:
                            for row in stmt.copy_data:
                                copy.write(row)
                    else:
                        cur.execute(rewrite(stmt.sql))
                except psycopg.errors.QueryCanceled as e:
                    raise DatasetLoadError(f"Restore exceeded the time limit at line {stmt.line}.") from e
                except psycopg.Error as e:
                    raise DatasetLoadError(
                        f"Dump failed at line {stmt.line}: {e.diag.message_primary or e} "
                        f"[{_preview(stmt.sql)}]"
                    ) from e
                ran += 1
    except DumpRejected as e:
        raise DatasetLoadError(f"Dump rejected at {e}") from e
    finally:
        watchdog.cancel()
    return ran, skipped


@contextmanager
def _open_text(path: Path) -> Iterator[io.TextIOBase]:
    if path.suffix.lower() == ".gz":
        f = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        f = open(path, "r", encoding="utf-8", newline="")
    with f:
        yield f


def _is_custom_format(path: Path) -> bool:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as f:
        return f.read(5) == b"PGDMP"


def _preview(s: str, n: int = 80) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- inspect

_INSPECT_SQL = """
SELECT n.nspname, c.relname,
       CASE c.relkind WHEN 'v' THEN 'view' WHEN 'm' THEN 'materialized view' ELSE 'table' END,
       a.attname, format_type(a.atttypid, a.atttypmod), bt.typname
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
JOIN pg_type t ON t.oid = a.atttypid
JOIN pg_type bt ON bt.oid = CASE WHEN t.typtype = 'd' THEN t.typbasetype ELSE t.oid END
WHERE c.relkind IN ('r', 'p', 'v', 'm')
  AND NOT c.relispartition
  AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg\\_%'
  AND NOT EXISTS (SELECT 1 FROM pg_depend d
                  WHERE d.classid = 'pg_class'::regclass AND d.objid = c.oid AND d.deptype = 'e')
ORDER BY n.nspname, c.relname, a.attnum
"""


def _inspect(conn: psycopg.Connection) -> list[Relation]:
    relations: dict[tuple[str, str], Relation] = {}
    for schema, rel, kind, col, col_type, base_type in conn.execute(_INSPECT_SQL):
        r = relations.setdefault((schema, rel), Relation(schema, rel, kind))
        if _SENSITIVE_NAME.search(col):
            r.columns.append(Column(col, col_type, hidden=True, hidden_reason="sensitive name"))
        elif base_type in _OPAQUE_TYPES:
            r.columns.append(Column(col, col_type, hidden=True, hidden_reason=f"{base_type} column"))
        else:
            r.columns.append(Column(col, col_type, hidden=False))
    return list(relations.values())


def _apply_visibility(relations: list[Relation], hide: list[str], expose: list[str]) -> None:
    def matches(ref: str, r: Relation, c: Column | None) -> bool:
        parts = ref.lower().split(".")
        target = [r.schema, r.name] + ([c.name] if c else [])
        return parts == target[-len(parts):] and len(parts) >= (2 if c else 1)

    unmatched = set(hide) | set(expose)
    for r in relations:
        for c in r.columns:
            for ref in expose:
                if matches(ref, r, c):
                    c.hidden, c.hidden_reason = False, None
                    unmatched.discard(ref)
            for ref in hide:
                if matches(ref, r, c) or (ref.count(".") <= 1 and matches(ref, r, None)):
                    c.hidden, c.hidden_reason = True, "hidden by user"
                    unmatched.discard(ref)
    if unmatched:
        raise DatasetLoadError(f"No such table/column in the dump: {', '.join(sorted(unmatched))}")


# --------------------------------------------------------------------------- grants

def _harden_and_grant(conn: psycopg.Connection, db_name: str, relations: list[Relation]) -> None:
    schemas = sorted({r.schema for r in relations})
    with conn.cursor() as cur:
        # Functions from the dump: not callable by the bot by default.
        for s in schemas:
            cur.execute(sql.SQL("REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA {} FROM PUBLIC")
                        .format(sql.Identifier(s)))

        cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO bot_reader").format(sql.Identifier(db_name)))
        for s in {r.schema for r in relations if any(not c.hidden for c in r.columns)}:
            cur.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO bot_reader").format(sql.Identifier(s)))

        for r in relations:
            visible = [c.name for c in r.columns if not c.hidden]
            table = sql.Identifier(r.schema, r.name)
            if not visible:
                continue
            if len(visible) == len(r.columns):
                cur.execute(sql.SQL("GRANT SELECT ON {} TO bot_reader").format(table))
            else:
                cols = sql.SQL(", ").join(map(sql.Identifier, visible))
                cur.execute(sql.SQL("GRANT SELECT ({}) ON {} TO bot_reader").format(cols, table))


# --------------------------------------------------------------------------- databases

def _create_database(db_name: str) -> None:
    with psycopg.connect(conninfo("data_loader", "postgres"), autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        conn.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(db_name)))


def _drop_database(db_name: str) -> None:
    with psycopg.connect(conninfo("data_loader", "postgres"), autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
