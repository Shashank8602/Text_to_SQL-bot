"""Retrieval against the real Pagila dataset with the local embedder (no LLM quota)."""

import psycopg
import pytest

from app.agent import graph
from app.agent.graph import run_agent
from app.database.catalog import load_catalog
from app.database.connection import app_conninfo
from app.datasets import descriptions
from app.datasets.loader import delete_dataset, load_pg_dump
from app.llm.prompts import JudgeVerdict, SQLGeneration
from app.retrieval.retriever import retrieve

from .conftest import PAGILA_DESCRIPTIONS, TEST_SLUG
from .test_agent import GOOD_SQL, llm  # noqa: F401  (fixture)


def test_value_and_join_completion(pagila_index, pagila_db):
    r = retrieve(TEST_SLUG, "How many customers live in Canada?", load_catalog(pagila_db),
                 embedder_name="local", k=5, glossary=descriptions.load(TEST_SLUG).glossary)
    assert {"customer", "address", "city", "country"} <= set(r.tables)
    assert [m.hint() for m in r.value_matches] == ["country.country = 'Canada'"]


def test_glossary_pulls_in_payment(pagila_index, pagila_db):
    r = retrieve(TEST_SLUG, "How much revenue came from R-rated films?", load_catalog(pagila_db),
                 embedder_name="local", k=5, glossary=descriptions.load(TEST_SLUG).glossary)
    # payment is reached through the glossary ("revenue"), film through the ranking,
    # rental/inventory as bridge tables.
    assert {"payment", "rental", "inventory", "film"} <= set(r.tables)


def test_agent_prompt_uses_retrieved_schema_and_hints(pagila_index, llm):  # noqa: F811
    llm.sql("SELECT nope FROM country", GOOD_SQL).judge(("pass", "")).answer("ok")
    r = run_agent("How many customers live in Canada?", TEST_SLUG, retrieval="local")
    first = llm.prompts[SQLGeneration][0]
    assert "CREATE TABLE country (" in first[0]["content"]
    assert "CREATE TABLE staff (" not in first[0]["content"]          # not relevant, not sent
    assert "-- Glossary (business definitions):" in first[0]["content"]
    assert "<value_matches>\ncountry.country = 'Canada'\n</value_matches>" in first[1]["content"]
    retry = llm.prompts[SQLGeneration][1][0]["content"]
    assert retry.count("CREATE ") > first[0]["content"].count("CREATE ")  # wider schema on retry
    assert r.retrieved_tables and "country" in r.retrieved_tables
    judge_prompt = llm.prompts[JudgeVerdict][0][1]["content"]
    assert "-- table, ~1,000 rows: Films in the catalog" in judge_prompt  # descriptions reach the judge


def test_retrieval_failure_falls_back_to_full_schema(pagila_db, llm, monkeypatch):  # noqa: F811
    def broken(*a, **k):
        raise RuntimeError("embedding service down")
    monkeypatch.setattr(graph, "rank_tables", broken)
    llm.sql(GOOD_SQL).judge(("pass", "")).answer("ok")
    r = run_agent("How many films?", TEST_SLUG, retrieval="local")
    assert r.outcome == "answered" and r.retrieved_tables is None
    assert "CREATE TABLE staff (" in llm.prompts[SQLGeneration][0][0]["content"]


def test_deleting_dataset_removes_its_index_and_descriptions(tmp_path):
    slug = "pytest_tiny"
    dump = tmp_path / "tiny.sql"
    dump.write_text("CREATE TABLE things (thing_id int PRIMARY KEY, label text);\n"
                    "COPY things (thing_id, label) FROM stdin;\n1\tWidget\n\\.\n", encoding="utf-8")
    from app.datasets import registry
    if registry.get(slug):
        delete_dataset(slug)
    load_pg_dump(dump, "Tiny", slug=slug)
    descriptions.save(slug, descriptions.Descriptions(tables={"things": "Stuff."}))
    from app.retrieval.indexer import build_index
    assert build_index(slug, "local") == 3  # table doc + 2 column docs
    delete_dataset(slug)
    with psycopg.connect(app_conninfo()) as conn:
        assert conn.execute("SELECT count(*) FROM rag.schema_docs_local WHERE dataset = %s", (slug,)).fetchone() == (0,)
        assert conn.execute("SELECT count(*) FROM rag.dataset_descriptions WHERE dataset = %s", (slug,)).fetchone() == (0,)


def test_pagila_descriptions_match_schema(pagila_db):
    assert descriptions.validate(descriptions.parse_yaml(PAGILA_DESCRIPTIONS), load_catalog(pagila_db)) == []
