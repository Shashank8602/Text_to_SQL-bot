"""Streamlit UI smoke tests (AppTest) with the agent stubbed out: no LLM quota."""

from decimal import Decimal
from pathlib import Path

from streamlit.testing.v1 import AppTest

import app.agent.graph as graph
from app.agent.graph import AgentResult
from app.database.executor import QueryResult

from .conftest import TEST_SLUG  # noqa: F401  (ensures a ready dataset exists)

SCRIPT = str(Path(__file__).resolve().parents[2] / "app" / "ui" / "streamlit_app.py")


def _result(**kw) -> AgentResult:
    base = dict(question="q", outcome="answered", sql="SELECT 1", reasoning=None, result=None, answer="Done.",
                warning=None, error=None)
    return AgentResult(**(base | kw))


def _app() -> AppTest:
    return AppTest.from_file(SCRIPT, default_timeout=60).run()


def test_renders_with_sidebar_and_samples(pagila_db):
    at = _app()
    assert not at.exception
    assert any("Ask about" in h.value for h in at.header)
    assert len(at.sidebar.button) >= 2


def test_question_shows_answer_table_and_sql(pagila_db, monkeypatch):
    qr = QueryResult(["category", "revenue"], ["text", "numeric"],
                     [("Sports", Decimal("10")), ("Drama", Decimal("7"))], False, 3)
    monkeypatch.setattr(graph, "run_agent", lambda *a, **k: _result(result=qr, answer="Sports wins."))
    at = _app()
    at.chat_input[0].set_value("Which category earned most?").run()
    assert not at.exception
    assert any("Sports wins." in m.value for m in at.markdown)
    assert len(at.dataframe) == 1


def test_clarification_flow(pagila_db, monkeypatch):
    asked = _result(outcome="clarifying", answer=None, sql=None, thread_id="t1",
                    clarification="By spend or rentals?", clarification_options=["Spend", "Rentals"])
    monkeypatch.setattr(graph, "run_agent", lambda *a, **k: asked)
    seen = {}

    def fake_resume(thread_id, answer, **_):
        seen.update(thread_id=thread_id, answer=answer)
        return _result(answer="By spend: KARL SEAL.")

    monkeypatch.setattr(graph, "resume_agent", fake_resume)
    at = _app()
    at.chat_input[0].set_value("best customers?").run()
    assert any("By spend or rentals?" in m.value for m in at.markdown)
    option = next(b for b in at.button if b.label == "Rentals")
    option.click().run()
    assert seen == {"thread_id": "t1", "answer": "Rentals"}
    assert any("KARL SEAL" in m.value for m in at.markdown)


def test_agent_exception_is_shown_not_raised(pagila_db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database down")
    monkeypatch.setattr(graph, "run_agent", boom)
    at = _app()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert any("database down" in e.value for e in at.error)


UPLOAD = str(Path(__file__).resolve().parents[2] / "app" / "ui" / "pages" / "1_Upload_data.py")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sample_csv"


def test_upload_page_renders_and_lists_datasets(pagila_db):
    at = AppTest.from_file(UPLOAD, default_timeout=60).run()
    assert not at.exception
    assert any("Bring your own data" in t.value for t in at.title)
    assert any(TEST_SLUG in m.value for m in at.markdown)


def test_upload_review_step_loads_a_dataset(pagila_db):
    from app.datasets import registry
    from app.datasets.loader import delete_dataset
    from app.datasets.relationships import detect
    from app.datasets.tabular_loader import parse_files

    tables = parse_files([FIXTURES / f"{n}.csv" for n in ("customers", "products", "orders")])
    at = AppTest.from_file(UPLOAD, default_timeout=60)
    at.session_state["wizard"] = {"kind": "tabular", "name": "UI Shop", "tables": tables,
                                  "candidates": detect(tables), "paths": [FIXTURES / "orders.csv"]}
    try:
        at.run()
        assert not at.exception
        assert len(at.checkbox) == 2          # two detected relationships, both pre-ticked
        next(b for b in at.button if b.label == "Load dataset").click().run()
        assert not at.exception, at.exception
        assert registry.get("ui_shop").status == "ready"
        assert any("is loaded" in s.value for s in at.success)
    finally:
        if registry.get("ui_shop"):
            delete_dataset("ui_shop")
