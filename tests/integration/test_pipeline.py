"""Pipeline tests against the real Pagila dataset.

Most use a scripted fake LLM (no API quota). Tests marked `llm` call the real providers and
are deselected by default; run them with `pytest -m llm`.
"""

import pytest

from app.database.catalog import load_catalog
from app.database.executor import execute_readonly
from app.llm import client
from app.llm.client import LLMResult
from app.llm.prompts import Answer, SQLGeneration
from app.pipeline import DatasetNotFound, answer_question

from .conftest import TEST_SLUG


@pytest.fixture
def fake_llm(monkeypatch):
    """Script the LLM: queue SQL and answer replies; record the prompts it received."""
    script = {"sql": [], "answers": [], "prompts": []}

    def fake_call(group, messages, schema, **_):
        script["prompts"].append(messages)
        if schema is SQLGeneration:
            data = SQLGeneration(reasoning="plan", sql=script["sql"].pop(0))
        else:
            data = Answer(answer=script["answers"].pop(0))
        return LLMResult(data=data, model="fake/model", fell_back=False, latency_ms=1,
                         prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(client, "call", fake_call)
    return script


def test_answers_question(pagila_db, fake_llm):
    fake_llm["sql"].append("SELECT count(*) AS n FROM film")
    fake_llm["answers"].append("There are 1,000 films.")
    r = answer_question("How many films?", TEST_SLUG)
    assert r.status == "answered"
    assert r.result.rows == [(1000,)]
    assert r.answer == "There are 1,000 films."
    assert [c.step for c in r.llm_calls] == ["generate_sql", "compose_answer"]
    # the answer prompt saw the real result
    assert "1000" in fake_llm["prompts"][1][1]["content"]


def test_schema_prompt_excludes_hidden_columns(pagila_db, fake_llm):
    fake_llm["sql"].append("SELECT 1")
    fake_llm["answers"].append("ok")
    answer_question("anything", TEST_SLUG)
    schema_prompt = fake_llm["prompts"][0][0]["content"]
    assert "CREATE TABLE staff (" in schema_prompt
    assert "password" not in schema_prompt
    assert "picture" not in schema_prompt
    assert "payment_p2022_01" not in schema_prompt


@pytest.mark.parametrize(("sql", "status"), [
    ("DELETE FROM payment", "unsafe"),
    ("SELECT * FROM films", "unsafe"),          # unknown table, caught before execution
    ("SELECT nope FROM film", "db_error"),
    ("", "declined"),
])
def test_failures_reported_without_answer(pagila_db, fake_llm, sql, status):
    fake_llm["sql"].append(sql)
    r = answer_question("q", TEST_SLUG)
    assert r.status == status
    assert r.answer is None
    assert r.error


def test_unknown_dataset():
    with pytest.raises(DatasetNotFound):
        answer_question("q", "no_such_dataset")


def test_catalog_matches_bot_visibility(pagila_db):
    catalog = load_catalog(pagila_db)
    assert "staff" in catalog.table_names
    assert "film_embedding" in catalog.table_names   # table visible, vector column hidden
    film_embedding = next(t for t in catalog.tables if t.name == "film_embedding")
    assert "embedding" not in [c.name for c in film_embedding.columns]
    film = next(t for t in catalog.tables if t.name == "film")
    rating = next(c for c in film.columns if c.name == "rating")
    assert rating.sample_values == ["G", "PG", "PG-13", "R", "NC-17"]
    payment = next(t for t in catalog.tables if t.name == "payment")
    assert payment.row_estimate > 10_000  # summed over partitions


@pytest.mark.llm
def test_live_llm_end_to_end(pagila_db):
    """Real Groq/Gemini call. Checks the answer's number against a hand-written query."""
    expected = execute_readonly("SELECT count(*) FROM rental WHERE return_date IS NULL", db_name=pagila_db).rows[0][0]
    r = answer_question("How many rentals have never been returned?", TEST_SLUG)
    assert r.status == "answered", r.error
    assert r.result.rows[0][0] == expected
    assert str(expected) in r.answer
