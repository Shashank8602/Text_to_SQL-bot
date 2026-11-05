"""Select the tables relevant to a question (docs/DESIGN.md §6.2, §14 techniques 1–2).

1. Hybrid ranking: vector similarity over table + column documents, and Postgres full-text keyword
   search, fused with Reciprocal Rank Fusion. A table's score is its best document.
2. Value matches: tables holding a value the question mentions ('Canada' -> country) are always
   included (app/retrieval/values.py). So are tables named in the definition of a glossary term the
   question uses ("revenue" -> "SUM(payment.amount)" -> payment).
3. Top-k tables plus those, then join-path completion: shortest paths over the foreign-key graph
   (declared and inferred) connect the selected tables, adding bridge tables such as film_category.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

import psycopg

from app.database.catalog import Catalog
from app.database.connection import app_conninfo
from app.database.migrations import ensure_docs_table
from app.retrieval.embedders import get_embedder
from app.retrieval.indexer import _vector_literal, build_index, index_exists
from app.retrieval.values import ValueMatch, load_value_index

RRF_K = 60
MAX_JOIN_HOPS = 4
MAX_VALUE_TABLES = 4


@dataclass
class Retrieval:
    ranked: list[tuple[str, float]]         # all tables, best first (fused score)
    selected: list[str]                     # top-k by rank, plus tables of value matches
    join_tables: list[str] = field(default_factory=list)  # added to connect the selected tables
    value_matches: list[ValueMatch] = field(default_factory=list)

    @property
    def tables(self) -> list[str]:
        return self.selected + [t for t in self.join_tables if t not in self.selected]


def retrieve(slug: str, question: str, catalog: Catalog, *, embedder_name: str, k: int,
             glossary: dict[str, str] | None = None) -> Retrieval:
    ranked = rank_tables(slug, question, catalog, embedder_name=embedder_name)
    return select(ranked, catalog, k, value_matches(question, catalog),
                  glossary_tables(question, glossary or {}, catalog))


def glossary_tables(question: str, glossary: dict[str, str], catalog: Catalog) -> list[str]:
    """Tables named in the definitions of glossary terms that the question uses."""
    q_words = {_stem(w) for w in re.findall(r"[a-z0-9]+", question.lower())}
    names = {t.name: t.qualified for t in catalog.tables}
    found: list[str] = []
    for term, definition in glossary.items():
        term_words = {_stem(w) for w in re.findall(r"[a-z0-9]+", term.lower()) if len(w) > 1}
        if term_words and term_words <= q_words:
            found += [names[w] for w in re.findall(r"[a-z_][a-z0-9_]*", definition.lower()) if w in names]
    return list(dict.fromkeys(found))


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def value_matches(question: str, catalog: Catalog) -> list[ValueMatch]:
    return load_value_index(catalog.db_name).find(question)


def select(ranked: list[tuple[str, float]], catalog: Catalog, k: int,
           matches: list[ValueMatch] = (), forced: list[str] = ()) -> Retrieval:
    selected = [t for t, _ in ranked[:k]]
    for table in dict.fromkeys([m.table for m in matches] + list(forced)):
        if table not in selected and len(selected) < k + MAX_VALUE_TABLES:
            selected.append(table)
    return Retrieval(ranked=ranked, selected=selected, join_tables=connect_tables(selected, catalog),
                     value_matches=list(matches))


def rank_tables(slug: str, question: str, catalog: Catalog, *, embedder_name: str) -> list[tuple[str, float]]:
    """All tables the bot can see, most relevant first (hybrid vector + keyword, RRF-fused)."""
    embedder = get_embedder(embedder_name)
    if not index_exists(slug, embedder.name):
        build_index(slug, embedder.name)
    table = ensure_docs_table(embedder.name, embedder.dim)
    vector = _vector_literal(embedder.embed_query(question))

    with psycopg.connect(app_conninfo()) as conn:
        by_vector = [r[0] for r in conn.execute(
            f"SELECT table_name FROM {table} WHERE dataset = %s "
            "GROUP BY table_name ORDER BY max(1 - (embedding <=> %s::vector)) DESC",
            (slug, vector))]
        by_keyword = [r[0] for r in conn.execute(
            f"""WITH q AS (SELECT to_tsquery('english', array_to_string(
                               tsvector_to_array(to_tsvector('english', %s)), ' | ')) AS q)
                SELECT table_name FROM {table}, q
                WHERE dataset = %s AND tsv @@ q.q
                GROUP BY table_name ORDER BY max(ts_rank(tsv, q.q)) DESC""",
            (question, slug))]

    known = {t.qualified for t in catalog.tables}
    return [(t, s) for t, s in fuse([by_vector, by_keyword]) if t in known]


def fuse(rankings: list[list[str]]) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = sum over rankings of 1 / (RRF_K + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def connect_tables(selected: list[str], catalog: Catalog) -> list[str]:
    """Bridge tables on shortest FK paths that connect the selected tables (in rank order)."""
    graph: dict[str, set[str]] = {}
    for t in catalog.tables:
        for fk in t.foreign_keys:
            graph.setdefault(t.qualified, set()).add(fk.ref_table)
            graph.setdefault(fk.ref_table, set()).add(t.qualified)

    in_graph = [t for t in selected if t in graph]
    if len(in_graph) < 2:
        return []
    component = {in_graph[0]}
    added: list[str] = []
    for target in in_graph[1:]:
        if target in component:
            continue
        path = _shortest_path(graph, component, target)
        if path is None or len(path) - 1 > MAX_JOIN_HOPS:
            continue
        for node in path:
            if node not in component and node not in selected:
                added.append(node)
            component.add(node)
        component.add(target)
    return added


def _shortest_path(graph: dict[str, set[str]], sources: set[str], target: str) -> list[str] | None:
    parent: dict[str, str | None] = {s: None for s in sources}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        if node == target:
            path = [node]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        for nxt in sorted(graph.get(node, ())):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    return None
