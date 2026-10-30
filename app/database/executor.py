"""Executes already safety-checked SQL as bot_reader.

Defense layers applied here (docs/DESIGN.md §7):
- READ ONLY transaction, always rolled back.
- Per-statement timeout.
- Server-side cursor (DECLARE ... CURSOR FOR) — Postgres itself rejects anything that
  is not a query, and we fetch at most max_rows + 1 rows without rewriting the SQL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import psycopg
from psycopg import errors

from app.config import get_settings
from app.database.connection import conninfo


@dataclass
class QueryResult:
    columns: list[str]
    column_types: list[str]  # Postgres type names, e.g. "numeric", "timestamptz"
    rows: list[tuple[Any, ...]]
    truncated: bool  # True if more than max_rows rows exist
    duration_ms: int


@dataclass
class QueryError(Exception):
    kind: Literal["db_error", "timeout"]
    message: str
    hint: str | None = None
    sqlstate: str | None = None
    duration_ms: int = 0
    detail: str | None = field(default=None, repr=False)

    def __str__(self) -> str:
        return self.message if not self.hint else f"{self.message} (HINT: {self.hint})"


def execute_readonly(
    sql: str,
    *,
    db_name: str,
    max_rows: int | None = None,
    timeout_ms: int | None = None,
) -> QueryResult:
    """Run sql as bot_reader against a dataset database (ds_<slug>)."""
    settings = get_settings()
    max_rows = max_rows or settings.max_result_rows
    timeout_ms = timeout_ms or settings.query_timeout_ms

    sql = sql.strip().rstrip(";").strip()
    start = time.perf_counter()

    # Connection failures are infrastructure errors and propagate as psycopg.OperationalError.
    with psycopg.connect(conninfo("bot_reader", db_name), autocommit=False) as conn:
        conn.read_only = True
        try:
            conn.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
            with conn.cursor(name="bot_query") as cur:
                cur.execute(sql)
                rows = cur.fetchmany(max_rows + 1)
                description = cur.description or []
        except errors.QueryCanceled as e:
            raise QueryError(
                kind="timeout",
                message=f"Query exceeded the {timeout_ms} ms time limit.",
                sqlstate=e.sqlstate,
                duration_ms=_elapsed_ms(start),
            ) from e
        except psycopg.Error as e:
            diag = e.diag
            raise QueryError(
                kind="db_error",
                message=diag.message_primary or str(e).strip(),
                hint=diag.message_hint,
                detail=diag.message_detail,
                sqlstate=e.sqlstate,
                duration_ms=_elapsed_ms(start),
            ) from e
        finally:
            conn.rollback()

        types = conn.adapters.types
        column_types = []
        for col in description:
            info = types.get(col.type_code)
            column_types.append(info.name if info else "unknown")

    return QueryResult(
        columns=[col.name for col in description],
        column_types=column_types,
        rows=rows[:max_rows],
        truncated=len(rows) > max_rows,
        duration_ms=_elapsed_ms(start),
    )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
