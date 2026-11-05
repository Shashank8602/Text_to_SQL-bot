"""Streamlit chat UI (docs/DESIGN.md section 9).

    .venv/Scripts/python -m streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `app` importable under `streamlit run`

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.agent.graph import DEFAULT_MAX_ATTEMPTS, AgentResult, resume_agent, run_agent  # noqa: E402
from app.charts import build_figure, suggest_chart  # noqa: E402
from app.datasets import registry  # noqa: E402

SAMPLE_QUESTIONS = {
    "pagila": [
        "Which film category earned the most revenue?",
        "Who are our best customers?",
        "How many rentals does each store have per month?",
        "Top 5 most rented films",
        "How many customers are from the UK?",
        "What is the weather in Paris?",
    ]
}
_ICONS = {"passed": "✅", "validation_failed": "⚠️", "db_error": "❌", "timeout": "⏱️", "unsafe": "🚫",
          "duplicate": "♻️"}


# --------------------------------------------------------------------------- rendering

def render_result(r: AgentResult) -> None:
    if r.outcome == "clarifying":
        st.markdown(f"**{r.clarification}**")
        return
    if r.outcome == "out_of_scope":
        st.info(r.answer)
        return
    if r.outcome == "declined":
        st.warning(f"I can't do that with a read-only query. {r.error}")
        return
    if r.outcome in ("gave_up", "llm_error") and not r.answer:
        st.error(r.error or "Something went wrong.")
    if r.answer:
        st.markdown(r.answer)
    if r.outcome == "best_effort":
        st.warning("A reviewer doubted this result fully answers the question. Check the SQL below.")
    elif r.warning and r.outcome == "answered":
        st.caption(f"Note: {r.warning}")

    if r.result is not None:
        spec = suggest_chart(r.result)
        if spec is not None:
            st.plotly_chart(build_figure(spec, r.result), width="stretch")
        label = f"Result table ({len(r.result.rows)} row{'s' if len(r.result.rows) != 1 else ''}" \
                + (", truncated)" if r.result.truncated else ")")
        with st.expander(label, expanded=spec is None):
            st.dataframe(pd.DataFrame(r.result.rows, columns=r.result.columns), width="stretch")
    if r.sql:
        with st.expander("SQL"):
            st.code(r.sql, language="sql")
    if r.attempts or r.llm_calls:
        n = len(r.attempts)
        with st.expander(f"Agent trace ({n} attempt{'s' if n != 1 else ''})"):
            if r.retrieved_tables is not None:
                st.caption(f"Schema retrieval sent {len(r.retrieved_tables)} tables: "
                           + ", ".join(r.retrieved_tables))
            for a in r.attempts:
                via = a.model + (" (fallback)" if a.fell_back else "")
                st.markdown(f"{_ICONS.get(a.status, '•')} **#{a.n} {a.status.replace('_', ' ')}** · {via}"
                            + (f" · {a.row_count} rows" if a.row_count is not None else ""))
                if a.status != "passed":
                    st.code(a.sql, language="sql")
                    st.caption(a.critique or a.error or "")
            if r.llm_calls:
                st.caption(" · ".join(f"{c.step}: {c.latency_ms} ms{' (fallback)' if c.fell_back else ''}"
                                      for c in r.llm_calls))


# --------------------------------------------------------------------------- app

def _reset() -> None:
    st.session_state.messages = []
    st.session_state.pending = None


def _handle(text: str, dataset: str, retrieval: str, max_attempts: int) -> None:
    st.session_state.messages.append({"role": "user", "text": text})
    pending: AgentResult | None = st.session_state.pending
    try:
        with st.spinner("Thinking…"):
            if pending is not None:
                r = resume_agent(pending.thread_id, text, max_attempts=max_attempts)
            else:
                r = run_agent(text, dataset, max_attempts=max_attempts, retrieval=retrieval,
                              thread_id=st.session_state.thread_id)
    except Exception as e:  # keep the chat alive on unexpected errors
        st.session_state.pending = None
        st.session_state.messages.append({"role": "assistant", "error": f"{type(e).__name__}: {e}"})
        return
    st.session_state.pending = r if r.outcome == "clarifying" else None
    if r.outcome != "clarifying":
        st.session_state.thread_id = None  # next question starts a fresh thread
    st.session_state.messages.append({"role": "assistant", "result": r})


def main() -> None:
    st.set_page_config(page_title="Text-to-SQL Bot", page_icon="🗄️", layout="wide")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("queued", None)

    ready = [d for d in registry.list_all() if d.status == "ready"]
    with st.sidebar:
        st.title("🗄️ Text-to-SQL Bot")
        if not ready:
            st.error("No dataset loaded. Run: `python -m app.datasets.cli load <dump> --name <name>`")
            st.stop()
        by_slug = {d.slug: d for d in ready}
        slug = st.selectbox("Dataset", list(by_slug), format_func=lambda s: by_slug[s].name,
                            on_change=_reset)
        retrieval = st.selectbox("Schema retrieval", ["local", "gemini", "none"],
                                 help="Embedding model used to pick relevant tables; 'none' sends the full schema.")
        max_attempts = st.slider("Max attempts", 1, 5, DEFAULT_MAX_ATTEMPTS)
        if st.button("Reset chat", width="stretch"):
            _reset()
            st.rerun()
        samples = SAMPLE_QUESTIONS.get(slug, [])
        if samples:
            st.subheader("Sample questions")
            for q in samples:
                if st.button(q, key=f"sample-{q}", width="stretch"):
                    st.session_state.queued = q

    st.header(f"Ask about **{by_slug[slug].name}**")
    if not st.session_state.messages:
        st.caption("Ask a question in plain English. The bot writes read-only SQL, checks the result and "
                   "retries if something looks wrong.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            if m["role"] == "user":
                st.markdown(m["text"])
            elif "error" in m:
                st.error(m["error"])
            else:
                render_result(m["result"])

    pending: AgentResult | None = st.session_state.pending
    if pending is not None:
        st.caption("Pick an option or type your own answer below.")
        cols = st.columns(max(1, len(pending.clarification_options)))
        for col, option in zip(cols, pending.clarification_options):
            if col.button(option, key=f"opt-{pending.thread_id}-{option}", width="stretch"):
                st.session_state.queued = option

    typed = st.chat_input("Answer the question above…" if pending else "Ask a question about your data")
    text = typed or st.session_state.queued
    st.session_state.queued = None
    if text:
        if pending is None and st.session_state.thread_id is None:
            import uuid
            st.session_state.thread_id = uuid.uuid4().hex
        _handle(text, slug, retrieval, max_attempts)
        st.rerun()


main()
