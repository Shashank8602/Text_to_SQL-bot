"""Draft table/column descriptions and a glossary with the LLM (docs/DESIGN.md section 14 #7, 15.3 step 5).

The draft is a starting point: it is validated against the real schema (unknown names are dropped) and
the user can edit or replace it. Sample values in the prompt come from the catalog the bot already sees.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.database.catalog import load_catalog
from app.datasets import descriptions as store
from app.datasets import registry
from app.llm import client


class ColumnDraft(BaseModel):
    name: str
    description: str


class TableDraft(BaseModel):
    name: str
    description: str = Field(description="One sentence: what one row of this table represents.")
    columns: list[ColumnDraft] = Field(description="Only columns whose meaning is not obvious from the name.")


class GlossaryDraft(BaseModel):
    term: str = Field(description="A business term a user might use, e.g. 'revenue'.")
    definition: str = Field(description="How to compute it from the columns, e.g. 'SUM(orders.total)'.")


class DescriptionDraft(BaseModel):
    tables: list[TableDraft]
    glossary: list[GlossaryDraft] = Field(description="At most 8 terms; only ones the data can define.")


SYSTEM = """\
You document a database so that an analyst (and a text-to-SQL assistant) can use it correctly.
Given the schema below (with sample values), write:
- one sentence per table: what a row represents;
- notes ONLY for columns whose meaning, units, or coding is not obvious from the name
  (e.g. status codes, flags, currency, text stored in a particular case);
- up to 8 glossary entries: business terms a user would ask about, defined precisely in terms of the
  actual columns (e.g. "revenue: SUM(orders.total)").
Rules: describe only what the schema and sample values support; never invent columns, tables or
business rules; keep every text short; use exactly the table and column names shown.

Schema:
{schema}
"""


def draft(slug: str) -> store.Descriptions:
    ds = registry.get(slug)
    if ds is None or ds.status != "ready":
        raise ValueError(f"No ready dataset named '{slug}'.")
    catalog = load_catalog(ds.db_name)
    messages = [{"role": "system", "content": SYSTEM.format(schema=catalog.to_ddl())},
                {"role": "user", "content": "Write the descriptions."}]
    r = client.call("generator", messages, DescriptionDraft)
    return _clean(r.data, catalog)


def _clean(d: DescriptionDraft, catalog) -> store.Descriptions:
    tables = {t.qualified: {c.name for c in t.columns} for t in catalog.tables}
    out = store.Descriptions()
    for t in d.tables:
        if t.name not in tables:
            continue
        if t.description.strip():
            out.tables[t.name] = t.description.strip()
        cols = {c.name: c.description.strip() for c in t.columns if c.name in tables[t.name] and c.description.strip()}
        if cols:
            out.columns[t.name] = cols
    out.glossary = {g.term.strip().lower(): g.definition.strip() for g in d.glossary
                    if g.term.strip() and g.definition.strip()}
    return out


def describe_and_save(slug: str) -> store.Descriptions:
    desc = draft(slug)
    store.save(slug, desc)
    return desc
