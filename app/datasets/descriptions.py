"""Business descriptions of a dataset: table/column descriptions and a glossary.

Stored per dataset in rag.dataset_descriptions. Imported from YAML (see
sample_data/pagila/descriptions.yaml); M7 adds LLM-drafted descriptions for uploads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import psycopg
import yaml
from psycopg.types.json import Jsonb

from app.database.catalog import Catalog
from app.database.connection import app_conninfo
from app.database.migrations import ensure_app_schema


@dataclass
class Descriptions:
    tables: dict[str, str] = field(default_factory=dict)                 # qualified table -> text
    columns: dict[str, dict[str, str]] = field(default_factory=dict)     # table -> column -> text
    glossary: dict[str, str] = field(default_factory=dict)               # term -> definition

    def table(self, name: str) -> str | None:
        return self.tables.get(name)

    def column(self, table: str, column: str) -> str | None:
        return self.columns.get(table, {}).get(column)

    def to_dict(self) -> dict:
        return {"tables": self.tables, "columns": self.columns, "glossary": self.glossary}

    @classmethod
    def from_dict(cls, d: dict) -> "Descriptions":
        return cls(d.get("tables", {}), d.get("columns", {}), d.get("glossary", {}))


def parse_yaml(path: Path) -> Descriptions:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    desc = Descriptions()
    for table, spec in (raw.get("tables") or {}).items():
        spec = spec or {}
        if spec.get("description"):
            desc.tables[str(table)] = str(spec["description"]).strip()
        cols = {str(c): str(t).strip() for c, t in (spec.get("columns") or {}).items() if t}
        if cols:
            desc.columns[str(table)] = cols
    desc.glossary = {str(k): str(v).strip() for k, v in (raw.get("glossary") or {}).items()}
    return desc


def validate(desc: Descriptions, catalog: Catalog) -> list[str]:
    """Names in the descriptions that don't exist in what the bot can see (typos, hidden columns)."""
    tables = {t.qualified: t for t in catalog.tables}
    problems = [f"unknown table '{t}'" for t in desc.tables if t not in tables]
    for t, cols in desc.columns.items():
        if t not in tables:
            problems.append(f"unknown table '{t}'")
            continue
        names = {c.name for c in tables[t].columns}
        problems += [f"unknown column '{t}.{c}'" for c in cols if c not in names]
    return sorted(set(problems))


def save(slug: str, desc: Descriptions) -> None:
    ensure_app_schema()
    with psycopg.connect(app_conninfo()) as conn:
        conn.execute(
            "INSERT INTO rag.dataset_descriptions (dataset, doc) VALUES (%s, %s) "
            "ON CONFLICT (dataset) DO UPDATE SET doc = EXCLUDED.doc, updated_at = now()",
            (slug, Jsonb(desc.to_dict())),
        )


def load(slug: str) -> Descriptions:
    ensure_app_schema()
    with psycopg.connect(app_conninfo()) as conn:
        row = conn.execute("SELECT doc FROM rag.dataset_descriptions WHERE dataset = %s", (slug,)).fetchone()
    return Descriptions.from_dict(row[0]) if row else Descriptions()
