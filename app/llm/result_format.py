"""Compact text rendering of a query result for LLM prompts.

The LLM never sees the full result (docs/DESIGN.md §8): it gets the row count, the first
rows, and summary statistics for numeric columns when rows were left out.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from app.database.executor import QueryResult

PREVIEW_ROWS = 20
MAX_CELL_CHARS = 80
_NUMERIC_TYPES = {"int2", "int4", "int8", "numeric", "float4", "float8", "money"}


def format_result(result: QueryResult, preview_rows: int = PREVIEW_ROWS) -> str:
    n = len(result.rows)
    count = f"{n}+ rows (truncated at {n})" if result.truncated else f"{n} row{'s' if n != 1 else ''}"
    if n == 0:
        return f"Columns: {', '.join(result.columns)}\n0 rows"

    shown = result.rows[:preview_rows]
    lines = [
        f"Columns: {', '.join(f'{c} ({t})' for c, t in zip(result.columns, result.column_types))}",
        f"Total: {count}" + (f"; showing the first {len(shown)}" if len(shown) < n else ""),
        "",
        " | ".join(result.columns),
    ]
    lines += [" | ".join(_cell(v) for v in row) for row in shown]

    if len(shown) < n:
        stats = _numeric_stats(result)
        if stats:
            lines += ["", "Statistics over all returned rows:"] + stats
    return "\n".join(lines)


def _numeric_stats(result: QueryResult) -> list[str]:
    out = []
    for i, (col, typ) in enumerate(zip(result.columns, result.column_types)):
        if typ not in _NUMERIC_TYPES or _is_identifier_like(col):
            continue
        values = [row[i] for row in result.rows if row[i] is not None]
        if not values:
            continue
        total = sum(Decimal(str(v)) for v in values)
        out.append(
            f"- {col}: min {_cell(min(values))}, max {_cell(max(values))}, "
            f"sum {_cell(total)}, avg {_cell(round(total / len(values), 2))}"
        )
    return out


def _is_identifier_like(column: str) -> bool:
    """ids and years are numbers, but their sum/avg is meaningless."""
    c = column.lower()
    return c == "id" or c.endswith("_id") or c in {"year", "month", "day", "quarter", "week"}


def _cell(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, datetime):
        s = value.isoformat(sep=" ", timespec="seconds")
    elif isinstance(value, date):
        s = value.isoformat()
    elif isinstance(value, (list, tuple)):
        s = "{" + ", ".join(_cell(v) for v in value) + "}"
    else:
        s = str(value)
    s = s.replace("\n", " ").replace("|", "/")
    return s if len(s) <= MAX_CELL_CHARS else s[: MAX_CELL_CHARS - 1] + "…"
