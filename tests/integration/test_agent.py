"""Agent graph tests against the real Pagila dataset with a scripted fake LLM (no API quota).

The live end-to-end test is marked `llm` and deselected by default (`pytest -m llm`).
"""

import pytest

from app.agent.graph import run_agent
from app.agent.validation import extract_filter_literals
from app.database.catalog import load_catalog
from app.database.executor import execute_readonly
from app.llm import client
from app.llm.client import LLMError, LLMResult
from app.llm.prompts import Answer, JudgeVerdict, SQLGeneration

GOOD_SQL = "SELECT count(*) AS n FROM film"


class FakeLLM:
    """Replies are queued per output schema. An Exception in a queue is raised instead."""

    def __init__(self):
        self.queues = {SQLGeneration: [], JudgeVerdict: [], Answer: []}
        self.prompts = {SQLGeneration: [], JudgeVerdict: [], Answer: []}

    def sql(self, *sqls):
        self.queues[SQLGeneration] += [s if isinstance(s, Exception) else SQLGeneration(reasoning="plan", sql=s)
                                       for s in sqls]
        return self

    def judge(self, *verdicts):
        self.queues[JudgeVerdict] += [v if isinstance(v, Exception) else
                                      JudgeVerdict(verdict=v[0], critique=v[1]) for v in verdicts]
        return self

    def answer(self, *answers):
        self.queues[Answer] += [Answer(answer=a) for a in answers]
        return self

    def __call__(self, group, messages, schema, **_):
        self.prompts[schema].append(messages)
        reply = self.queues[schema].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(data=reply, model=f"fake/{group}", fell_back=False, latency_ms=1,
                         prompt_tokens=1, completion_tokens=1)


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(client, "call", fake)
    yield fake
    leftover = {s.__name__: q for s, q in fake.queues.items() if q}
    assert not leftover, f"scripted replies not consumed: {leftover}"


def statuses(r):
    return [a.status for a in r.attempts]


def test_first_try_success(pagila_db, llm):
    llm.sql(GOOD_SQL).judge(("pass", "")).answer("There are 1,000 films.")
    r = run_agent("How many films?", pagila_db)
    assert r.outcome == "answered"
    assert statuses(r) == ["passed"]
    assert r.sql == GOOD_SQL and r.result.rows == [(1000,)]
    assert r.answer == "There are 1,000 films."
    assert [c.step for c in r.llm_calls] == ["generate_sql", "validate_result", "compose_answer"]


def test_db_error_is_fed_back_with_real_columns(pagila_db, llm):
    llm.sql("SELECT name FROM film", "SELECT title FROM film LIMIT 1").judge(("pass", "")).answer("ok")
    r = run_agent("Name one film", pagila_db)
    assert statuses(r) == ["db_error", "passed"]
    assert "Columns of film: film_id, title" in r.attempts[0].error
    retry_prompt = llm.prompts[SQLGeneration][1][1]["content"]
    assert "Attempt 1 (db_error)" in retry_prompt
    assert "SELECT name FROM film" in retry_prompt


def test_unsafe_query_is_never_executed_and_retried(pagila_db, llm):
    llm.sql("DELETE FROM film", GOOD_SQL).judge(("pass", "")).answer("ok")
    r = run_agent("How many films?", pagila_db)
    assert statuses(r) == ["unsafe", "passed"]
    assert execute_readonly("SELECT count(*) FROM film", db_name=pagila_db).rows == [(1000,)]


def test_duplicate_query_is_not_rerun(pagila_db, llm):
    bad = "SELECT nope FROM film"
    llm.sql(bad, "select  NOPE from FILM", GOOD_SQL).judge(("pass", "")).answer("ok")
    r = run_agent("q", pagila_db)
    assert statuses(r) == ["db_error", "duplicate", "passed"]


def test_missing_filter_value_caught_without_judge(pagila_db, llm):
    usa = ("SELECT COALESCE(SUM(p.amount), 0) FROM payment p JOIN customer c USING (customer_id) "
           "JOIN address a USING (address_id) JOIN city ci USING (city_id) JOIN country co USING (country_id) "
           "WHERE co.country = '{}'")
    llm.sql(usa.format("USA"), usa.format("United States")).judge(("pass", "")).answer("ok")
    r = run_agent("How much did US customers pay?", pagila_db)
    assert statuses(r) == ["validation_failed", "passed"]
    assert "'United States'" in r.attempts[0].critique
    assert len(llm.prompts[JudgeVerdict]) == 1  # judge only saw the second attempt


def test_judge_rejection_then_success(pagila_db, llm):
    llm.sql("SELECT title FROM film LIMIT 3", GOOD_SQL)
    llm.judge(("fail", "The question asks for a count."), ("pass", "")).answer("1,000")
    r = run_agent("How many films?", pagila_db)
    assert statuses(r) == ["validation_failed", "passed"]
    assert "asks for a count" in llm.prompts[SQLGeneration][1][1]["content"]


def test_best_effort_when_judge_keeps_rejecting(pagila_db, llm):
    llm.sql("SELECT 1 AS a", "SELECT 2 AS a", "SELECT 3 AS a")
    llm.judge(*[("fail", "doubtful")] * 3).answer("Best guess: 3.")
    r = run_agent("q", pagila_db)
    assert r.outcome == "best_effort"
    assert r.result.rows == [(3,)] and r.sql == "SELECT 3 AS a"
    assert r.warning == "doubtful"
    answer_prompt = llm.prompts[Answer][0][1]["content"]
    assert "<reviewer_note>\ndoubtful\n</reviewer_note>" in answer_prompt


def test_gives_up_with_explanation(pagila_db, llm):
    llm.sql("SELECT a FROM film", "SELECT b FROM film", "SELECT c FROM film")
    r = run_agent("q", pagila_db)
    assert r.outcome == "gave_up"
    assert statuses(r) == ["db_error"] * 3
    assert "after 3 attempts" in r.answer and "Attempt 3 (db error)" in r.answer
    assert r.result is None


def test_max_attempts_respected(pagila_db, llm):
    llm.sql("SELECT a FROM film")
    r = run_agent("q", pagila_db, max_attempts=1)
    assert r.outcome == "gave_up" and len(r.attempts) == 1


def test_declined(pagila_db, llm):
    llm.sql("")
    r = run_agent("delete everything", pagila_db)
    assert r.outcome == "declined" and r.attempts == []


def test_llm_outage_reported(pagila_db, llm):
    llm.sql(LLMError("All providers failed"))
    r = run_agent("q", pagila_db)
    assert r.outcome == "llm_error" and "All providers failed" in r.error


def test_judge_outage_does_not_block(pagila_db, llm):
    llm.sql(GOOD_SQL).judge(LLMError("down")).answer("1,000")
    r = run_agent("q", pagila_db)
    assert r.outcome == "answered"
    assert "could not be double-checked" in r.warning


def test_compose_can_be_skipped(pagila_db, llm):
    llm.sql(GOOD_SQL).judge(("pass", ""))
    r = run_agent("q", pagila_db, compose=False)
    assert r.outcome == "answered" and r.answer is None and r.result.rows == [(1000,)]


def test_schema_prompt_excludes_hidden_columns(pagila_db, llm):
    llm.sql(GOOD_SQL).judge(("pass", "")).answer("ok")
    run_agent("q", pagila_db)
    schema_prompt = llm.prompts[SQLGeneration][0][0]["content"]
    assert "CREATE TABLE staff (" in schema_prompt
    assert "password" not in schema_prompt and "payment_p2022_01" not in schema_prompt


class TestFilterLiterals:
    def test_resolves_aliases_and_unqualified_columns(self, pagila_db):
        cat = load_catalog(pagila_db)
        lits = extract_filter_literals(
            "SELECT * FROM film f JOIN language l ON l.language_id = f.language_id "
            "WHERE l.name = 'English' AND rating IN ('PG', 'R') AND title ILIKE '%love%' AND f.length = 90", cat)
        assert {(x.table, x.column, x.value, x.is_pattern) for x in lits} == {
            ("language", "name", "English", False), ("film", "rating", "PG", False),
            ("film", "rating", "R", False), ("film", "title", "%love%", True)}

    def test_pattern_matches_shown_to_judge(self, pagila_db, llm):
        uk = ("SELECT count(*) FROM customer cu JOIN address a USING (address_id) JOIN city ci USING (city_id) "
              "JOIN country co USING (country_id) WHERE co.country ILIKE '%uk%'")
        llm.sql(uk).judge(("pass", "")).answer("ok")
        run_agent("How many customers are from the UK?", pagila_db)
        judge_prompt = llm.prompts[JudgeVerdict][0][1]["content"]
        assert "country.country LIKE '%uk%' matched: 'Ukraine'" in judge_prompt

    def test_ambiguous_column_skipped(self, pagila_db):
        cat = load_catalog(pagila_db)
        # last_update exists in both tables, so it can't be attributed
        sql = "SELECT * FROM actor JOIN film_actor USING (actor_id) WHERE last_update = '2020-01-01'"
        assert extract_filter_literals(sql, cat) == []


@pytest.mark.llm
def test_live_agent_end_to_end(pagila_db):
    """Real Groq/Gemini calls; checks the number against a hand-written query."""
    expected = execute_readonly("SELECT count(*) FROM rental WHERE return_date IS NULL",
                                db_name=pagila_db).rows[0][0]
    r = run_agent("How many rentals have never been returned?", pagila_db)
    assert r.outcome == "answered", r.error
    assert r.result.rows[0][0] == expected
    assert str(expected) in r.answer
