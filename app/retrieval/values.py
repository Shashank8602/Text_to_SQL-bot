"""Value index: link words in the question to actual values in the data (docs/DESIGN.md §14 #6).

"How many customers live in Canada?" mentions no table, but 'Canada' is a value of country.country.
Matching values (a) forces those tables into the retrieved schema and (b) gives the SQL generator
the exact stored spelling, e.g. 'Varanasi' -> city.city = 'Varanasi (Benares)'.

Built in memory per dataset from what bot_reader can see: distinct values of text columns of base
tables, skipping high-cardinality columns. Views are skipped so values point to the source tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import psycopg
from psycopg import sql as pgsql

from app.database.catalog import load_catalog
from app.database.connection import conninfo

MAX_DISTINCT_PER_COLUMN = 5000
MAX_MATCHES = 8
_TEXT_TYPES = ("text", "character varying", "character", "citext")
_PARENS = re.compile(r"\([^)]*\)")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "in", "is", "it",
    "many", "much", "of", "on", "or", "the", "to", "was", "were", "what", "which", "who", "with", "all",
    "new", "most", "more", "than", "each", "per", "total", "number", "count", "average", "top",
}


@dataclass(frozen=True)
class ValueMatch:
    table: str   # qualified
    column: str
    value: str

    def hint(self) -> str:
        return f"{self.table}.{self.column} = '{self.value}'"


class ValueIndex:
    def __init__(self, entries: list[tuple[str, str, str]]):
        # first significant word -> [(words, table, column, value)]
        self._by_word: dict[str, list[tuple[frozenset[str], str, str, str]]] = {}
        for table, column, value in entries:
            words = _value_words(value)
            if words:
                self._by_word.setdefault(min(words), []).append((words, table, column, value))

    def find(self, question: str) -> list[ValueMatch]:
        q_words = set(_WORD.findall(question.lower()))
        found = []
        for w in q_words:
            for words, table, column, value in self._by_word.get(w, ()):
                if words <= q_words:
                    found.append((len(words), len(value), ValueMatch(table, column, value)))
        # Prefer multi-word, longer values ("ACADEMY DINOSAUR" over "ACADEMY").
        found.sort(key=lambda x: (-x[0], -x[1], x[2].table, x[2].column))
        out, seen = [], set()
        for _, _, m in found:
            if (m.table, m.column, m.value.lower()) not in seen:
                seen.add((m.table, m.column, m.value.lower()))
                out.append(m)
        return out[:MAX_MATCHES]

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_word.values())


def _value_words(value: str) -> frozenset[str] | None:
    words = _WORD.findall(_PARENS.sub(" ", value.lower()))
    significant = [w for w in words if w not in _STOPWORDS]
    if not significant or all(w.isdigit() for w in words):
        return None
    if len(significant) == 1 and len(significant[0]) < 3:
        return None
    return frozenset(significant)


@lru_cache(maxsize=16)
def load_value_index(db_name: str) -> ValueIndex:
    catalog = load_catalog(db_name)
    entries: list[tuple[str, str, str]] = []
    with psycopg.connect(conninfo("bot_reader", db_name), autocommit=True) as conn:
        conn.execute("SET statement_timeout = 5000")
        for t in catalog.tables:
            if t.kind != "table":
                continue
            for c in t.columns:
                if not c.type.startswith(_TEXT_TYPES) or c.type.endswith("[]"):
                    continue
                query = pgsql.SQL("SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL LIMIT {n}").format(
                    c=pgsql.Identifier(c.name), t=pgsql.Identifier(t.schema, t.name),
                    n=pgsql.Literal(MAX_DISTINCT_PER_COLUMN + 1))
                try:
                    values = [r[0] for r in conn.execute(query)]
                except psycopg.Error:
                    continue
                if len(values) <= MAX_DISTINCT_PER_COLUMN:
                    entries += [(t.qualified, c.name, v) for v in values if len(v) <= 80 and "@" not in v]
    return ValueIndex(entries)
