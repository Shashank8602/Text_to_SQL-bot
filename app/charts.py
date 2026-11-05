"""Pick and build a chart for a query result (docs/DESIGN.md section 9).

Deterministic rules over column types, not an LLM call: no quota, no failure mode, and the spec is
valid by construction. Returns None when a table is the better presentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from app.database.executor import QueryResult

MAX_CHART_ROWS = 40
_ID = re.compile(r"(^id$|_id$|^id_|code$|zip|postal)", re.I)
_NUMERIC = ("int", "numeric", "decimal", "float", "double", "real", "money", "serial")
_TEMPORAL = ("date", "timestamp")


@dataclass(frozen=True)
class ChartSpec:
    type: Literal["bar", "line", "scatter"]
    x: str
    y: str
    title: str


def _is_numeric(pg_type: str) -> bool:
    t = pg_type.lower()
    return any(k in t for k in _NUMERIC) and "interval" not in t


def _is_temporal(pg_type: str) -> bool:
    t = pg_type.lower()
    return any(k in t for k in _TEMPORAL)


def suggest_chart(result: QueryResult) -> ChartSpec | None:
    cols = list(zip(result.columns, result.column_types))
    if len(cols) < 2 or not 2 <= len(result.rows) <= MAX_CHART_ROWS:
        return None
    measures = [c for c, t in cols if _is_numeric(t) and not _ID.search(c)]
    categories = [(c, t) for c, t in cols if not _is_numeric(t)]
    if categories and measures:
        x, xt = categories[0]
        y = measures[0]
        if _is_temporal(xt):
            return ChartSpec("line", x, y, f"{y} by {x}")
        distinct = len({row[result.columns.index(x)] for row in result.rows})
        if distinct == len(result.rows):  # one bar per category
            return ChartSpec("bar", x, y, f"{y} by {x}")
        return None
    if len(measures) >= 2 and len(result.rows) >= 5:
        return ChartSpec("scatter", measures[0], measures[1], f"{measures[1]} vs {measures[0]}")
    return None


def _plain(value):
    return float(value) if isinstance(value, Decimal) else value


def build_figure(spec: ChartSpec, result: QueryResult):
    """A Plotly figure for the spec. Imported lazily so the agent never needs plotly."""
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame([[_plain(v) for v in row] for row in result.rows], columns=result.columns)
    if spec.type == "line":
        fig = px.line(df.sort_values(spec.x), x=spec.x, y=spec.y, markers=True, title=spec.title)
    elif spec.type == "bar":
        fig = px.bar(df, x=spec.x, y=spec.y, title=spec.title)
    else:
        fig = px.scatter(df, x=spec.x, y=spec.y, title=spec.title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
