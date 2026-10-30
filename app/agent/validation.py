"""Deterministic checks that run before the LLM judge (docs/DESIGN.md §5.2).

They are free (no LLM call) and give the generator very specific feedback, e.g.
"'NY' not found in city.city; closest values: 'New York', ...".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
import sqlglot
from psycopg import sql as pgsql
from sqlglot import exp

from app.database.catalog import Catalog
from app.database.connection import conninfo
from app.database.executor import QueryResult

logging.getLogger("sqlglot").setLevel(logging.ERROR)

LOOKUP_TIMEOUT_MS = 2000
SUGGESTIONS = 5


@dataclass
class FilterLiteral:
    schema: str
    table: str
    column: str
    value: str
    is_pattern: bool  # LIKE/ILIKE pattern rather than an exact value


def check_result(sql: str, result: QueryResult, catalog: Catalog) -> str | None:
    """Return a critique if the result is clearly broken, else None."""
    if _effectively_empty(result) and (problem := _missing_filter_values(sql, catalog)):
        what = "no rows" if not result.rows else "only zero/NULL values"
        return f"The query returned {what}. {problem}"
    if result.rows and all(v is None for row in result.rows for v in row):
        return ("Every value in the result is NULL. This usually means a wrong join (e.g. a LEFT JOIN on "
                "the wrong key) or aggregating a column that is never set.")
    return None


def _effectively_empty(result: QueryResult) -> bool:
    """No rows, or a single aggregate row of zeros/NULLs (e.g. COALESCE(SUM(...), 0) over nothing)."""
    if not result.rows:
        return True
    return len(result.rows) == 1 and all(v is None or v == 0 for v in result.rows[0])


def _missing_filter_values(sql: str, catalog: Catalog) -> str | None:
    problems = []
    for lit in extract_filter_literals(sql, catalog):
        if _value_exists(catalog.db_name, lit):
            continue
        close = _closest_values(catalog.db_name, lit)
        where = f"{lit.table}.{lit.column}"
        msg = f"The filter value '{lit.value}' does not occur in {where}."
        if close:
            msg += " Closest existing values: " + ", ".join(f"'{v}'" for v in close) + "."
        problems.append(msg)
    return " ".join(problems) or None


def extract_filter_literals(sql: str, catalog: Catalog) -> list[FilterLiteral]:
    """String literals compared with =, IN, LIKE or ILIKE against a known table column."""
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except sqlglot.errors.ParseError:
        return []

    tables_by_name = {t.name.lower(): t for t in catalog.tables}
    alias_to_table = {}
    for t in tree.find_all(exp.Table):
        if t.name.lower() in tables_by_name:
            alias_to_table[t.alias_or_name.lower()] = tables_by_name[t.name.lower()]
    referenced = list({id(v): v for v in alias_to_table.values()}.values())

    def resolve(col: exp.Column):
        name = col.name.lower()
        if col.table:
            table = alias_to_table.get(col.table.lower())
            return table if table and any(c.name == name for c in table.columns) else None
        owners = [t for t in referenced if any(c.name == name for c in t.columns)]
        return owners[0] if len(owners) == 1 else None

    found: list[FilterLiteral] = []

    def add(col: exp.Expression, lit: exp.Expression, pattern: bool):
        if isinstance(col, exp.Column) and isinstance(lit, exp.Literal) and lit.is_string:
            table = resolve(col)
            if table is not None:
                found.append(FilterLiteral(table.schema, table.name, col.name.lower(), lit.this, pattern))

    for node in tree.find_all(exp.EQ, exp.In, exp.Like, exp.ILike):
        if isinstance(node, exp.EQ):
            add(node.this, node.expression, False)
            add(node.expression, node.this, False)
        elif isinstance(node, exp.In):
            for value in node.expressions:
                add(node.this, value, False)
        else:
            add(node.this, node.expression, True)
    return found


def pattern_match_notes(sql: str, catalog: Catalog, limit: int = 5) -> list[str]:
    """What each LIKE/ILIKE filter actually matched, e.g. "country.country ILIKE '%uk%' matched:
    'Ukraine'". Given to the judge, which otherwise can't see that a pattern hit the wrong values."""
    notes = []
    for lit in extract_filter_literals(sql, catalog):
        if not lit.is_pattern:
            continue
        query = pgsql.SQL("SELECT DISTINCT {c}::text FROM {t} WHERE {c}::text ILIKE %s ORDER BY 1 LIMIT {n}").format(
            t=pgsql.Identifier(lit.schema, lit.table), c=pgsql.Identifier(lit.column), n=pgsql.Literal(limit + 1))
        rows = _lookup(catalog.db_name, query, (lit.value,))
        if rows is None:
            continue
        values = [r[0] for r in rows]
        shown = ", ".join(f"'{v}'" for v in values[:limit]) + (", …" if len(values) > limit else "")
        notes.append(f"{lit.table}.{lit.column} LIKE '{lit.value}' matched: {shown or 'nothing'}")
    return notes


def _value_exists(db_name: str, lit: FilterLiteral) -> bool:
    op = "ILIKE" if lit.is_pattern else "="
    query = pgsql.SQL("SELECT 1 FROM {t} WHERE {c}::text " + op + " %s LIMIT 1").format(
        t=pgsql.Identifier(lit.schema, lit.table), c=pgsql.Identifier(lit.column))
    rows = _lookup(db_name, query, (lit.value,))
    return rows is None or bool(rows)  # on lookup failure, don't accuse the query


def _closest_values(db_name: str, lit: FilterLiteral) -> list[str]:
    needle = lit.value.strip("%") if lit.is_pattern else lit.value
    query = pgsql.SQL(
        "SELECT v FROM (SELECT DISTINCT {c}::text AS v FROM {t} WHERE {c} IS NOT NULL) d "
        "ORDER BY similarity(v, %s) DESC, v LIMIT {n}"
    ).format(t=pgsql.Identifier(lit.schema, lit.table), c=pgsql.Identifier(lit.column),
             n=pgsql.Literal(SUGGESTIONS))
    rows = _lookup(db_name, query, (needle,))
    return [r[0] for r in rows or []]


def _lookup(db_name: str, query, params) -> list[tuple] | None:
    try:
        with psycopg.connect(conninfo("bot_reader", db_name), autocommit=True) as conn:
            conn.execute(f"SET statement_timeout = {LOOKUP_TIMEOUT_MS}")
            return conn.execute(query, params).fetchall()
    except psycopg.Error:
        return None


def column_hint(sql: str, catalog: Catalog) -> str | None:
    """For undefined-column/table errors: list the real columns of the tables the query uses."""
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        names = {t.name.lower() for t in tree.find_all(exp.Table)}
    except sqlglot.errors.ParseError:
        names = set()
    tables = [t for t in catalog.tables if t.name in names]
    if not tables:
        return "Available tables: " + ", ".join(sorted(t.qualified for t in catalog.tables)) + "."
    return " ".join(f"Columns of {t.qualified}: {', '.join(c.name for c in t.columns)}." for t in tables)
