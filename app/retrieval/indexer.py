"""Build the schema-retrieval index of a dataset: one document per table and per column,
embedded and stored in rag.schema_docs_<embedder> (docs/DESIGN.md §6.2)."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from app.database.catalog import Catalog, Table, load_catalog
from app.database.connection import app_conninfo
from app.database.migrations import ensure_docs_table
from app.datasets import descriptions as descriptions_store
from app.datasets import registry
from app.datasets.descriptions import Descriptions
from app.retrieval.embedders import get_embedder


@dataclass
class SchemaDoc:
    table: str
    column: str | None
    content: str


def schema_documents(catalog: Catalog, desc: Descriptions) -> list[SchemaDoc]:
    docs = []
    for t in catalog.tables:
        docs.append(SchemaDoc(t.qualified, None, _table_text(t, desc)))
        for c in t.columns:
            parts = [f"Column {c.name} ({_words(c.name)}) of {t.kind} {t.qualified} ({_words(t.name)}), type {c.type}."]
            if text := desc.column(t.qualified, c.name):
                parts.append(text)
            if c.sample_values:
                parts.append("Values: " + ", ".join(v for v in c.sample_values if v != "…") + ".")
            docs.append(SchemaDoc(t.qualified, c.name, " ".join(parts)))
    return docs


def _table_text(t: Table, desc: Descriptions) -> str:
    parts = [f"{t.kind.capitalize()} {t.qualified} ({_words(t.name)})."]
    if text := desc.table(t.qualified):
        parts.append(text)
    parts.append("Columns: " + ", ".join(c.name for c in t.columns) + ".")
    related = sorted({fk.ref_table for fk in t.foreign_keys})
    if related:
        parts.append("Linked to: " + ", ".join(related) + ".")
    return " ".join(parts)


def _words(name: str) -> str:
    return name.replace("_", " ")


def build_index(slug: str, embedder_name: str) -> int:
    ds = registry.get(slug)
    if ds is None or ds.status != "ready":
        raise ValueError(f"No ready dataset named '{slug}'.")
    embedder = get_embedder(embedder_name)
    docs = schema_documents(load_catalog(ds.db_name), descriptions_store.load(slug))
    vectors = embedder.embed_documents([d.content for d in docs])
    table = ensure_docs_table(embedder.name, embedder.dim)

    with psycopg.connect(app_conninfo()) as conn:
        conn.execute(f"DELETE FROM {table} WHERE dataset = %s", (slug,))
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {table} (dataset, table_name, column_name, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s::vector)",
                [(slug, d.table, d.column, d.content, _vector_literal(v)) for d, v in zip(docs, vectors)],
            )
        conn.execute(
            "INSERT INTO rag.index_builds (dataset, embedder, doc_count) VALUES (%s, %s, %s) "
            "ON CONFLICT (dataset, embedder) DO UPDATE SET doc_count = EXCLUDED.doc_count, built_at = now()",
            (slug, embedder.name, len(docs)),
        )
    return len(docs)


def index_exists(slug: str, embedder_name: str) -> bool:
    with psycopg.connect(app_conninfo()) as conn:
        try:
            return conn.execute("SELECT 1 FROM rag.index_builds WHERE dataset = %s AND embedder = %s",
                                (slug, embedder_name)).fetchone() is not None
        except psycopg.errors.UndefinedTable:
            return False


def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"
