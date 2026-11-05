"""Idempotent schema migrations for the app database (rag schema), run as rag_writer.

db/init/01_setup.sh only runs on a fresh volume; tables added later are created here so existing
installations (and their loaded datasets) keep working without a reset.
"""

from __future__ import annotations

import re
from functools import lru_cache

import psycopg

from app.database.connection import app_conninfo

_MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS rag.dataset_descriptions (
           dataset    text PRIMARY KEY REFERENCES rag.datasets(slug) ON DELETE CASCADE,
           doc        jsonb NOT NULL,
           updated_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS rag.index_builds (
           dataset    text NOT NULL REFERENCES rag.datasets(slug) ON DELETE CASCADE,
           embedder   text NOT NULL,
           doc_count  int NOT NULL,
           built_at   timestamptz NOT NULL DEFAULT now(),
           PRIMARY KEY (dataset, embedder))""",
]


@lru_cache
def ensure_app_schema() -> None:
    with psycopg.connect(app_conninfo()) as conn:
        for ddl in _MIGRATIONS:
            conn.execute(ddl)


@lru_cache
def ensure_docs_table(embedder: str, dim: int) -> str:
    """Per-embedder table of schema documents (dimensions differ between embedders)."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", embedder):
        raise ValueError(f"Invalid embedder name: {embedder!r}")
    ensure_app_schema()
    table = f"rag.schema_docs_{embedder}"
    with psycopg.connect(app_conninfo()) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id          bigserial PRIMARY KEY,
                dataset     text NOT NULL REFERENCES rag.datasets(slug) ON DELETE CASCADE,
                table_name  text NOT NULL,          -- qualified, as in the catalog
                column_name text,                   -- NULL for the table-level document
                content     text NOT NULL,
                tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                embedding   vector({int(dim)}) NOT NULL)""")
        conn.execute(f"CREATE INDEX IF NOT EXISTS schema_docs_{embedder}_dataset ON {table} (dataset)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS schema_docs_{embedder}_tsv ON {table} USING gin (tsv)")
    return table
