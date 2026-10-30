"""Schema catalog of a dataset, as seen by bot_reader (docs/DESIGN.md §6.1).

Built from pg_catalog with privilege checks, so columns hidden at load time never reach
the LLM. Rendered as compact DDL for prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import psycopg
from psycopg import sql

from app.database.connection import conninfo

SAMPLE_MAX_DISTINCT = 50   # only columns with at most this many distinct values get samples
SAMPLE_SHOW = 10           # values shown per column
_SAMPLE_TYPES = {"text", "varchar", "bpchar", "citext"}


@dataclass
class Column:
    name: str
    type: str
    nullable: bool
    sample_values: list[str] | None = None


@dataclass
class ForeignKey:
    columns: list[str]
    ref_table: str  # qualified like Table.qualified
    ref_columns: list[str]


@dataclass
class Table:
    schema: str
    name: str
    kind: str  # table | view | materialized view
    row_estimate: int
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    @property
    def qualified(self) -> str:
        return self.name if self.schema == "public" else f"{self.schema}.{self.name}"


@dataclass
class Catalog:
    db_name: str
    tables: list[Table]

    @property
    def table_names(self) -> set[str]:
        return {t.name for t in self.tables}

    @property
    def schemas(self) -> set[str]:
        return {t.schema for t in self.tables}

    def to_ddl(self, tables: list[str] | None = None) -> str:
        selected = [t for t in self.tables if tables is None or t.qualified in tables]
        return "\n\n".join(_render(t) for t in selected)


_RELATIONS_SQL = """
SELECT c.oid, n.nspname, c.relname,
       CASE c.relkind WHEN 'v' THEN 'view' WHEN 'm' THEN 'materialized view' ELSE 'table' END,
       CASE WHEN c.relkind = 'p'
            THEN (SELECT COALESCE(sum(GREATEST(ch.reltuples, 0)), 0) FROM pg_inherits i
                  JOIN pg_class ch ON ch.oid = i.inhrelid WHERE i.inhparent = c.oid)
            ELSE GREATEST(c.reltuples, 0) END::bigint
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p', 'v', 'm')
  AND NOT c.relispartition
  AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg\\_%'
  AND has_schema_privilege(n.oid, 'USAGE')
  AND EXISTS (SELECT 1 FROM pg_attribute a
              WHERE a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
                AND has_column_privilege(c.oid, a.attnum, 'SELECT'))
ORDER BY n.nspname, c.relname
"""

_COLUMNS_SQL = """
SELECT a.attrelid, a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull,
       bt.typname, t.typtype = 'e' AS is_enum, a.atttypid
FROM pg_attribute a
JOIN pg_type t ON t.oid = a.atttypid
JOIN pg_type bt ON bt.oid = CASE WHEN t.typtype = 'd' THEN t.typbasetype ELSE t.oid END
WHERE a.attrelid = ANY(%s) AND a.attnum > 0 AND NOT a.attisdropped
  AND has_column_privilege(a.attrelid, a.attnum, 'SELECT')
ORDER BY a.attrelid, a.attnum
"""

_CONSTRAINTS_SQL = """
SELECT con.conrelid, con.contype,
       ARRAY(SELECT attname FROM unnest(con.conkey) WITH ORDINALITY k(n, i)
             JOIN pg_attribute ON attrelid = con.conrelid AND attnum = k.n ORDER BY k.i),
       con.confrelid,
       ARRAY(SELECT attname FROM unnest(con.confkey) WITH ORDINALITY k(n, i)
             JOIN pg_attribute ON attrelid = con.confrelid AND attnum = k.n ORDER BY k.i)
FROM pg_constraint con
WHERE con.conrelid = ANY(%s) AND con.contype IN ('p', 'f')
"""


@lru_cache(maxsize=16)
def load_catalog(db_name: str) -> Catalog:
    with psycopg.connect(conninfo("bot_reader", db_name), autocommit=True) as conn:
        relations = conn.execute(_RELATIONS_SQL).fetchall()
        by_oid = {oid: Table(schema, name, kind, rows) for oid, schema, name, kind, rows in relations}
        oids = list(by_oid)

        enum_labels = dict(conn.execute(
            "SELECT enumtypid, array_agg(enumlabel ORDER BY enumsortorder) FROM pg_enum GROUP BY 1"
        ).fetchall())

        sample_candidates = []
        for relid, name, type_, nullable, base_type, is_enum, type_oid in conn.execute(_COLUMNS_SQL, (oids,)):
            col = Column(name, type_, nullable)
            if is_enum:
                col.sample_values = list(enum_labels.get(type_oid, []))
            elif base_type in _SAMPLE_TYPES:
                sample_candidates.append((by_oid[relid], col))
            by_oid[relid].columns.append(col)

        for relid, contype, cols, ref_oid, ref_cols in conn.execute(_CONSTRAINTS_SQL, (oids,)):
            table = by_oid[relid]
            if contype == "p":
                table.primary_key = cols
            elif ref_oid in by_oid:  # only FKs to tables the bot can see
                table.foreign_keys.append(ForeignKey(cols, by_oid[ref_oid].qualified, ref_cols))

        for table, col in sample_candidates:
            col.sample_values = _sample_values(conn, table, col.name)

    return Catalog(db_name, list(by_oid.values()))


def _sample_values(conn: psycopg.Connection, table: Table, column: str) -> list[str] | None:
    """Distinct values for low-cardinality text columns, else None."""
    query = sql.SQL("SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL LIMIT {n}").format(
        col=sql.Identifier(column),
        tbl=sql.Identifier(table.schema, table.name),
        n=sql.Literal(SAMPLE_MAX_DISTINCT + 1),
    )
    try:
        values = [r[0] for r in conn.execute(query)]
    except psycopg.Error:
        return None
    if not values or len(values) > SAMPLE_MAX_DISTINCT:
        return None
    return sorted(str(v) for v in values)[:SAMPLE_SHOW] + (["…"] if len(values) > SAMPLE_SHOW else [])


def _render(t: Table) -> str:
    keyword = {"table": "TABLE", "view": "VIEW", "materialized view": "MATERIALIZED VIEW"}[t.kind]
    single_pk = t.primary_key[0] if len(t.primary_key) == 1 else None
    fk_by_col = {fk.columns[0]: fk for fk in t.foreign_keys if len(fk.columns) == 1}

    items: list[tuple[str, str]] = []  # (definition, trailing comment)
    for c in t.columns:
        line = f"  {c.name} {c.type}"
        if c.name == single_pk:
            line += " PRIMARY KEY"
        elif not c.nullable and t.kind == "table":
            line += " NOT NULL"
        if fk := fk_by_col.get(c.name):
            line += f" REFERENCES {fk.ref_table}({fk.ref_columns[0]})"
        comment = ""
        if c.sample_values:
            comment = "  -- values: " + ", ".join(f"'{v}'" if v != "…" else v for v in c.sample_values)
        items.append((line, comment))
    if len(t.primary_key) > 1:
        items.append((f"  PRIMARY KEY ({', '.join(t.primary_key)})", ""))
    for fk in t.foreign_keys:
        if len(fk.columns) > 1:
            items.append((f"  FOREIGN KEY ({', '.join(fk.columns)}) "
                          f"REFERENCES {fk.ref_table}({', '.join(fk.ref_columns)})", ""))

    body = "\n".join(
        definition + ("," if i < len(items) - 1 else "") + comment
        for i, (definition, comment) in enumerate(items)
    )
    header = f"-- {t.kind}, ~{t.row_estimate:,} rows" if t.kind != "view" else "-- view"
    return f"{header}\nCREATE {keyword} {t.qualified} (\n{body}\n);"
