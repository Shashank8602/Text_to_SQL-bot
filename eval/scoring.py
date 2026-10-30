"""Execution-accuracy scoring: does the bot's result contain the gold answer?

Lenient on presentation, strict on content:
- column names and column order don't matter; extra columns in the bot's result are fine
- every gold column must match some bot column value-for-value, with rows aligned
- a gold text column may also match two adjacent bot text columns joined by a space
  (first_name + last_name)
- numbers compare after rounding to 2 decimals; text compares case-insensitively
- row order matters only when the question asks for a ranking (order_matters)
"""

from __future__ import annotations

import itertools
from collections import Counter
from datetime import date, datetime
from decimal import Decimal

from app.database.executor import QueryResult

MAX_COMBINATIONS = 2000


def results_match(gold: QueryResult, pred: QueryResult | None, *, order_matters: bool = False) -> bool:
    if pred is None or len(gold.rows) != len(pred.rows):
        return False
    if not gold.rows:
        return True

    gold_cols = _columns(gold.rows)
    pred_cols = _columns(pred.rows)
    pred_cols += [merged[i] for merged, i in _merged_name_pairs(pred_cols)]
    # The gold side may split a name that the bot returned as one column, or vice versa.
    variants = [gold_cols] + [merged for merged, _ in _merged_name_pairs(gold_cols)]
    return any(_match(g, pred_cols, order_matters) for g in variants)


def _match(gold_cols: list[list], pred_cols: list[list], order_matters: bool) -> bool:
    candidates = []
    for g in gold_cols:
        key = sorted(g, key=_sort_key)
        matches = [k for k, p in enumerate(pred_cols) if sorted(p, key=_sort_key) == key]
        if not matches:
            return False
        candidates.append(matches)

    gold_rows = list(zip(*gold_cols))
    for combo in itertools.islice(itertools.product(*candidates), MAX_COMBINATIONS):
        pred_rows = list(zip(*(pred_cols[k] for k in combo)))
        if order_matters and pred_rows == gold_rows:
            return True
        if not order_matters and Counter(pred_rows) == Counter(gold_rows):
            return True
    return False


def normalize(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return round(float(value), 2)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return tuple(normalize(v) for v in value)
    return str(value).strip().lower()


def _columns(rows) -> list[list]:
    return [[normalize(v) for v in col] for col in zip(*rows)]


def _merged_name_pairs(cols: list[list]) -> list[tuple[list[list], int]]:
    """For each adjacent pair of text columns: (columns with that pair joined by a space, index of the
    joined column)."""
    out = []
    for i, (a, b) in enumerate(zip(cols, cols[1:])):
        if all(isinstance(x, str) for x in a + b):
            joined = [f"{x} {y}" for x, y in zip(a, b)]
            out.append((cols[:i] + [joined] + cols[i + 2:], i))
    return out


def _sort_key(v):
    return (v is None, type(v).__name__, v if v is not None else 0)
