"""Static safety check for LLM-generated SQL.

This is layer 2 of the defense in depth (see docs/DESIGN.md §7). Layer 1, the read-only
`bot_reader` role, is the real guarantee; this layer gives the LLM a precise, early
rejection reason it can learn from on the next attempt, and blocks things the role
alone would allow (e.g. reading pg_catalog, pg_sleep).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# sqlglot warns when it falls back to parsing as Command; we reject those anyway.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

DIALECT = "postgres"

# Any of these node types anywhere in the tree means the statement can modify state
# or run something other than a plain query.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,
    exp.Create, exp.Drop, exp.Alter, exp.TruncateTable,
    exp.Copy, exp.Command, exp.Grant, exp.Revoke,
    exp.Transaction, exp.Commit, exp.Rollback, exp.Set,
    exp.Into, exp.Lock, exp.Analyze, exp.Execute, exp.Declare,
)

_FORBIDDEN_SCHEMAS = frozenset({"pg_catalog", "information_schema", "pg_toast", "rag"})

_FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "txid_")
_FORBIDDEN_FUNCTIONS = frozenset({
    "set_config", "current_setting",
    "nextval", "setval", "currval", "lastval",
    # Execute an arbitrary query string, bypassing this checker.
    "query_to_xml", "query_to_xml_and_xmlschema", "query_to_xmlschema",
    "cursor_to_xml", "cursor_to_xmlschema",
})


@dataclass(frozen=True)
class SafetyResult:
    ok: bool
    reason: str | None = None

    @classmethod
    def reject(cls, reason: str) -> "SafetyResult":
        return cls(ok=False, reason=reason)


_OK = SafetyResult(ok=True)


def check_sql(
    sql: str,
    allowed_tables: set[str] | None = None,
    allowed_schemas: set[str] = frozenset({"public"}),
) -> SafetyResult:
    """Accept only a single read-only SELECT (CTEs and set operations allowed).

    allowed_tables: if given, every referenced table must be in it (or be a CTE name).
    Names are compared lowercase and unqualified.
    allowed_schemas: schemas a table may be qualified with (the active dataset's schemas).
    """
    if not sql or not sql.strip():
        return SafetyResult.reject("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except ParseError as e:
        return SafetyResult.reject(f"SQL could not be parsed: {e}")

    if len(statements) != 1:
        return SafetyResult.reject(
            f"Exactly one statement is allowed, got {len(statements)}."
        )
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return SafetyResult.reject(
            f"Only SELECT queries are allowed, got {_describe(tree)}."
        )

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return SafetyResult.reject(
                f"Only read-only SELECT queries are allowed; found {_describe(node)}."
            )

    if (result := _check_functions(tree)) is not _OK:
        return result
    return _check_tables(tree, allowed_tables, allowed_schemas)


def _check_functions(tree: exp.Expression) -> SafetyResult:
    for func in tree.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_FUNCTION_PREFIXES):
            return SafetyResult.reject(f"Function '{name}' is not allowed.")
    return _OK


def _check_tables(
    tree: exp.Expression, allowed_tables: set[str] | None, allowed_schemas: set[str]
) -> SafetyResult:
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} if allowed_tables is not None else None
    schemas = {s.lower() for s in allowed_schemas}

    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = table.db.lower()
        if not name:  # table-valued function, e.g. FROM generate_series(...)
            continue
        if schema in _FORBIDDEN_SCHEMAS or name.startswith("pg_"):
            return SafetyResult.reject(f"System table '{table.sql(dialect=DIALECT)}' is not allowed.")
        if schema and schema not in schemas:
            return SafetyResult.reject(f"Schema '{schema}' is not allowed.")
        if name in cte_names and not schema:
            continue
        if allowed is not None and name not in allowed:
            return SafetyResult.reject(
                f"Unknown table '{name}'. Use only tables from the provided schema."
            )
    return _OK


def _describe(node: exp.Expression) -> str:
    if isinstance(node, exp.Command):
        return f"command '{node.name.upper()}'"
    if isinstance(node, exp.Into):
        return "SELECT ... INTO"
    if isinstance(node, exp.Lock):
        return "row locking clause (FOR UPDATE/SHARE)"
    return node.key.upper()
