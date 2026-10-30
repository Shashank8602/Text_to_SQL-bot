"""Dataset registry (rag.datasets in the app database, written as rag_writer)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from app.database.connection import app_conninfo


class DatasetExists(Exception):
    pass


@dataclass
class Dataset:
    slug: str
    name: str
    db_name: str
    source_type: str
    source_file: str | None
    status: str
    hidden_columns: list[str]
    created_at: datetime


def reserve(*, slug: str, name: str, db_name: str, source_type: str, source_file: str | None) -> None:
    with psycopg.connect(app_conninfo()) as conn:
        try:
            conn.execute(
                "INSERT INTO rag.datasets (slug, name, db_name, source_type, source_file, status) "
                "VALUES (%s, %s, %s, %s, %s, 'loading')",
                (slug, name, db_name, source_type, source_file),
            )
        except psycopg.errors.UniqueViolation as e:
            raise DatasetExists(f"A dataset named '{slug}' already exists.") from e
        except psycopg.errors.CheckViolation as e:
            raise ValueError(
                f"Invalid dataset id '{slug}': use lowercase letters, digits and _, starting with a letter."
            ) from e


def mark_ready(slug: str, *, hidden_columns: list[str]) -> None:
    with psycopg.connect(app_conninfo()) as conn:
        conn.execute(
            "UPDATE rag.datasets SET status = 'ready', hidden_columns = %s WHERE slug = %s",
            (Jsonb(hidden_columns), slug),
        )


def remove(slug: str) -> None:
    with psycopg.connect(app_conninfo()) as conn:
        conn.execute("DELETE FROM rag.datasets WHERE slug = %s", (slug,))


def get(slug: str) -> Dataset | None:
    with psycopg.connect(app_conninfo()) as conn:
        cur = conn.cursor(row_factory=class_row(Dataset))
        return cur.execute("SELECT * FROM rag.datasets WHERE slug = %s", (slug,)).fetchone()


def list_all() -> list[Dataset]:
    with psycopg.connect(app_conninfo()) as conn:
        cur = conn.cursor(row_factory=class_row(Dataset))
        return cur.execute("SELECT * FROM rag.datasets ORDER BY created_at").fetchall()
