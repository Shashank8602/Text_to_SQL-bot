"""Detect likely foreign keys between uploaded tables (docs/DESIGN.md section 15.3 step 4).

A candidate `orders.customer_id -> customers.customer_id` needs: a plausible name match, compatible
types, a target column that is unique and never null, and >= 95% of the source's non-null values
present in the target. The user confirms or rejects each candidate before anything is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.datasets.tabular import ParsedColumn, ParsedTable

MIN_OVERLAP = 0.95
_KEYLIKE = ("id", "code", "key", "no", "num", "number", "sku")
_INTS = {"integer", "bigint"}


@dataclass(frozen=True)
class Relationship:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    match_pct: float = 100.0

    def label(self) -> str:
        return (f"{self.from_table}.{self.from_column} → {self.to_table}.{self.to_column} "
                f"({self.match_pct:.1f}% match)")


def detect(tables: list[ParsedTable]) -> list[Relationship]:
    found = []
    for src in tables:
        for col in src.columns:
            best = _best_target(src, col, tables)
            if best:
                found.append(best)
    return found


def _best_target(src: ParsedTable, col: ParsedColumn, tables: list[ParsedTable]) -> Relationship | None:
    src_vals = [v for v in src.values(col.name) if v is not None]
    if not src_vals:
        return None
    best: tuple[tuple[int, float], Relationship] | None = None
    for tgt in tables:
        if tgt is src:
            continue
        for d in tgt.columns:
            name_score = _name_score(src.name, col.name, tgt.name, d.name)
            if name_score == 0 or not _compatible(col.type, d.type):
                continue
            tgt_vals = tgt.values(d.name)
            if not tgt_vals or None in tgt_vals or len(set(tgt_vals)) != len(tgt_vals):
                continue  # target must be a unique, non-null key
            keys = set(tgt_vals)
            overlap = sum(v in keys for v in src_vals) / len(src_vals)
            if overlap < MIN_OVERLAP:
                continue
            rel = Relationship(src.name, col.name, tgt.name, d.name, round(overlap * 100, 1))
            rank = (name_score, overlap)
            if best is None or rank > best[0]:
                best = (rank, rel)
    return best[1] if best else None


def _compatible(a: str, b: str) -> bool:
    return a == b and a in _INTS | {"text"} or (a in _INTS and b in _INTS)


def _name_score(src_table: str, src_col: str, tgt_table: str, tgt_col: str) -> int:
    """0 = no plausible link; 2 = the column is named after the target table; 1 = same key-like name."""
    singular = _singular(tgt_table)
    if tgt_col == "id" and src_col in {f"{singular}_id", f"{tgt_table}_id", f"{singular}id", f"id_{singular}"}:
        return 2
    if src_col == tgt_col and src_col != "id" and (src_col.endswith(_KEYLIKE) or src_col in _KEYLIKE):
        return 2 if src_col.startswith(singular) else 1
    return 0


def _singular(name: str) -> str:
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("ses") or name.endswith("xes"):
        return name[:-2]
    return name[:-1] if name.endswith("s") and not name.endswith("ss") else name
