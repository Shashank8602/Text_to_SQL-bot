"""The self-correcting Text-to-SQL agent (docs/DESIGN.md §4–5).

    retrieve_schema → generate_sql → check_sql → execute_sql → validate_result → compose_answer
                          ▲              │            │               │
                          └──────────────┴────────────┴───────────────┘
                              failure, attempts < max_attempts
                                         │ attempts exhausted
                                         ▼
                                  finish_failure → compose_answer (best effort) | END

Every failure is recorded as an Attempt and fed back to the generator on the next try.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from functools import lru_cache

import sqlglot
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agent.state import AgentState, Attempt, Draft, LLMCall, Outcome
from app.agent.validation import check_result, column_hint, pattern_match_notes
from app.database.catalog import Catalog, load_catalog
from app.database.executor import QueryError, QueryResult, execute_readonly
from app.config import get_settings
from app.database.safety import check_sql
from app.datasets import descriptions as descriptions_store
from app.datasets import registry
from app.llm import client
from app.retrieval.retriever import glossary_tables, rank_tables, select, value_matches
from app.llm.prompts import (
    Answer, JudgeVerdict, SQLGeneration, Understanding, answer_messages, generate_sql_messages,
    judge_messages, understand_messages, with_clarifications,
)

DEFAULT_MAX_ATTEMPTS = 3
MAX_CLARIFICATIONS = 2
RETRY_EXTRA_TABLES = 4
_UNDEFINED_OBJECT_STATES = {"42703", "42P01"}  # undefined_column, undefined_table


# --------------------------------------------------------------------------- nodes

def understand_question(state: AgentState) -> dict:
    """Triage: clear / ambiguous / out of scope (docs/DESIGN.md section 4.3)."""
    if not state.get("understand", True):
        return {"pending_question": None}
    clar = state.get("clarifications", [])
    catalog, desc = _catalog(state), descriptions_store.load(state["dataset"])
    messages = understand_messages(state["question"], catalog.to_ddl(descriptions=desc), desc.glossary, clar)
    try:
        r = client.call("generator", messages, Understanding)
    except client.LLMError:
        return {"pending_question": None}  # triage is an optimisation: go on with the best reading
    u, calls = r.data, [_llm_call("understand_question", r)]
    if u.verdict == "out_of_scope":
        return {"outcome": "out_of_scope", "pending_question": None, "llm_calls": calls,
                "answer": u.reason or "That is outside what this dataset can answer."}
    if u.verdict == "ambiguous" and u.clarifying_question and len(clar) < MAX_CLARIFICATIONS:
        options = [o for o in u.options if o.strip()][:4]
        return {"pending_question": u.clarifying_question, "pending_options": options, "llm_calls": calls}
    update: dict = {"pending_question": None, "llm_calls": calls}
    if u.assumption.strip():
        update["assumptions"] = [u.assumption.strip()]
    return update


def ask_clarification(state: AgentState) -> dict:
    """Pause the graph; the caller resumes it with the user's answer (Command(resume=...))."""
    question = state["pending_question"]
    answer = interrupt({"question": question, "options": state.get("pending_options", [])})
    return {"clarifications": [(question, str(answer).strip())], "pending_question": None}


def retrieve_schema(state: AgentState) -> dict:
    """Pick the relevant tables (docs/DESIGN.md §6.2). Retries get a wider schema."""
    catalog, desc = _catalog(state), descriptions_store.load(state["dataset"])
    mode = state.get("retrieval") or get_settings().retrieval_embedder
    full = {"schema_ddl": catalog.to_ddl(descriptions=desc), "retrieved_tables": None, "value_hints": []}
    if mode == "none":
        return full | {"schema_ddl_retry": full["schema_ddl"]}
    k = get_settings().retrieval_top_k
    try:
        ranked = rank_tables(state["dataset"], _q(state), catalog, embedder_name=mode)
    except Exception as e:  # embedder/index unavailable: degrade to the full schema, don't fail
        return full | {"schema_ddl_retry": full["schema_ddl"], "warning": f"Schema retrieval unavailable ({e})."}
    matches = value_matches(_q(state), catalog)
    forced = glossary_tables(_q(state), desc.glossary, catalog)
    narrow = select(ranked, catalog, k, matches, forced)
    wide = select(ranked, catalog, k + RETRY_EXTRA_TABLES, matches, forced)
    return {
        "schema_ddl": catalog.to_ddl(narrow.tables, descriptions=desc),
        "schema_ddl_retry": catalog.to_ddl(wide.tables, descriptions=desc),
        "retrieved_tables": narrow.tables,
        "value_hints": [m.hint() for m in matches],
    }


def generate_sql(state: AgentState) -> dict:
    attempts = state.get("attempts", [])
    schema = state["schema_ddl_retry"] if attempts else state["schema_ddl"]
    messages = generate_sql_messages(_q(state), schema, previous_attempts=attempts,
                                     value_hints=state.get("value_hints", []))
    try:
        r = client.call("generator", messages, SQLGeneration)
    except client.LLMError as e:
        return {"outcome": "llm_error", "error": str(e), "draft": None}
    calls = [_llm_call("generate_sql", r)]
    sql = r.data.sql.strip().rstrip(";").strip()
    if not sql:
        return {"outcome": "declined", "error": r.data.reasoning, "draft": None, "llm_calls": calls}
    return {"draft": Draft(sql, r.data.reasoning, r.model, r.fell_back), "llm_calls": calls}


def check(state: AgentState) -> dict:
    draft = state["draft"]
    seen = {_normalize(a.sql) for a in state.get("attempts", [])}
    if _normalize(draft.sql) in seen:
        return _fail(state, "duplicate", error="This exact query was already tried and failed.")
    catalog = _catalog(state)
    safety = check_sql(draft.sql, allowed_tables=catalog.table_names, allowed_schemas=catalog.schemas)
    if not safety.ok:
        return _fail(state, "unsafe", error=safety.reason)
    return {}


def execute(state: AgentState) -> dict:
    draft = state["draft"]
    try:
        result = execute_readonly(draft.sql, db_name=state["db_name"])
    except QueryError as e:
        error = str(e)
        if e.sqlstate in _UNDEFINED_OBJECT_STATES and (hint := column_hint(draft.sql, _catalog(state))):
            error += f" {hint}"
        if e.kind == "timeout":
            error += " Simplify the query: avoid cross joins and unnecessary subqueries, filter early."
        return _fail(state, e.kind, error=error, duration_ms=e.duration_ms)
    return {"result": result}


def validate(state: AgentState) -> dict:
    draft, result = state["draft"], state["result"]
    catalog = _catalog(state)

    critique = check_result(draft.sql, result, catalog)
    if critique:
        return _fail(state, "validation_failed", critique=critique, result=result)

    referenced = _referenced_tables(draft.sql, catalog)
    desc = descriptions_store.load(state["dataset"])
    messages = judge_messages(_q(state), draft.sql, result,
                              catalog.to_ddl(referenced or None, descriptions=desc),
                              filter_notes=pattern_match_notes(draft.sql, catalog))
    try:
        r = client.call("judge", messages, JudgeVerdict)
    except client.LLMError:
        # The judge is a safety net, not a gate: if it's unavailable, accept the result.
        return _pass(state, warning="The result could not be double-checked (reviewer unavailable).")
    calls = [_llm_call("validate_result", r)]
    if r.data.verdict == "fail":
        return _fail(state, "validation_failed", critique=r.data.critique, result=result) | {"llm_calls": calls}
    return _pass(state) | {"llm_calls": calls}


def finish_failure(state: AgentState) -> dict:
    last = state["attempts"][-1]
    if last.status == "validation_failed" and last.result is not None and last.result.rows:
        # The query ran, the reviewer doubted it: answer anyway, with a visible warning.
        return {"outcome": "best_effort", "result": last.result, "final_sql": last.sql,
                "warning": last.critique}
    return {"outcome": "gave_up", "answer": _explain_failure(state)}


def compose_answer(state: AgentState) -> dict:
    outcome: Outcome = state.get("outcome") or "answered"
    if not state.get("compose", True):
        return {"outcome": outcome}
    caveat = state.get("warning") if outcome == "best_effort" else None
    assumptions = state.get("assumptions", [])
    try:
        r = client.call("generator", answer_messages(_q(state), state["final_sql"], state["result"],
                                                     caveat=caveat), Answer)
    except client.LLMError as e:
        return {"outcome": outcome, "answer": None, "error": f"Could not compose an answer: {e}"}
    text = r.data.answer
    if assumptions:
        text += "\n\n*Assumption: " + " ".join(assumptions) + "*"
    return {"outcome": outcome, "answer": text, "llm_calls": [_llm_call("compose_answer", r)]}


# --------------------------------------------------------------------------- routing

def _after_understand(state: AgentState) -> str:
    if state.get("outcome") == "out_of_scope":
        return END
    return "ask_clarification" if state.get("pending_question") else "retrieve_schema"


def _after_generate(state: AgentState) -> str:
    return END if state.get("outcome") in ("llm_error", "declined") else "check_sql"


def _after_step(next_node: str):
    def route(state: AgentState) -> str:
        if state.get("draft") is not None:
            return next_node
        return "generate_sql" if len(state["attempts"]) < state["max_attempts"] else "finish_failure"
    return route


def _after_finish(state: AgentState) -> str:
    return "compose_answer" if state["outcome"] == "best_effort" else END


_CHECKPOINTER = MemorySaver()  # per process; one chat session = one thread_id


@lru_cache
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("understand_question", understand_question)
    g.add_node("ask_clarification", ask_clarification)
    g.add_node("retrieve_schema", retrieve_schema)
    g.add_node("generate_sql", generate_sql)
    g.add_node("check_sql", check)
    g.add_node("execute_sql", execute)
    g.add_node("validate_result", validate)
    g.add_node("finish_failure", finish_failure)
    g.add_node("compose_answer", compose_answer)

    g.add_edge(START, "understand_question")
    g.add_conditional_edges("understand_question", _after_understand,
                            ["ask_clarification", "retrieve_schema", END])
    g.add_edge("ask_clarification", "understand_question")
    g.add_edge("retrieve_schema", "generate_sql")
    g.add_conditional_edges("generate_sql", _after_generate, ["check_sql", END])
    g.add_conditional_edges("check_sql", _after_step("execute_sql"),
                            ["execute_sql", "generate_sql", "finish_failure"])
    g.add_conditional_edges("execute_sql", _after_step("validate_result"),
                            ["validate_result", "generate_sql", "finish_failure"])
    g.add_conditional_edges("validate_result", _after_step("compose_answer"),
                            ["compose_answer", "generate_sql", "finish_failure"])
    g.add_conditional_edges("finish_failure", _after_finish, ["compose_answer", END])
    g.add_edge("compose_answer", END)
    return g.compile(checkpointer=_CHECKPOINTER)


# --------------------------------------------------------------------------- entry point

@dataclass
class AgentResult:
    question: str
    outcome: Outcome
    sql: str | None
    reasoning: str | None
    result: QueryResult | None
    answer: str | None
    warning: str | None
    error: str | None
    attempts: list[Attempt] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    retrieved_tables: list[str] | None = None   # None: full schema was used
    value_hints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    thread_id: str | None = None
    clarification: str | None = None            # set when outcome == "clarifying"
    clarification_options: list[str] = field(default_factory=list)


def run_agent(question: str, dataset: str, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
              compose: bool = True, retrieval: str | None = None, understand: bool = True,
              thread_id: str | None = None) -> AgentResult:
    """dataset: slug of a ready dataset. retrieval: "local" | "gemini" | "none" (default: settings).

    If the question is ambiguous the result has outcome "clarifying"; answer it with resume_agent()."""
    ds = registry.get(dataset)
    if ds is None or ds.status != "ready":
        raise ValueError(f"No ready dataset named '{dataset}'.")
    thread_id = thread_id or uuid.uuid4().hex
    final = build_graph().invoke(
        {"question": question, "dataset": dataset, "db_name": ds.db_name, "max_attempts": max_attempts,
         "compose": compose, "retrieval": retrieval, "understand": understand,
         "clarifications": [], "assumptions": [], "pending_question": None,
         "attempts": [], "llm_calls": [], "draft": None, "result": None},
        config=_config(thread_id, max_attempts),
    )
    return _to_result(question, final, thread_id)


def resume_agent(thread_id: str, answer: str, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> AgentResult:
    """Continue a run that stopped with outcome "clarifying", giving the user's answer."""
    config = _config(thread_id, max_attempts)
    snapshot = build_graph().get_state(config)
    if not snapshot.next:
        raise ValueError("This conversation is not waiting for a clarification.")
    final = build_graph().invoke(Command(resume=answer), config=config)
    return _to_result(snapshot.values["question"], final, thread_id)


def _config(thread_id: str, max_attempts: int) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 25 + 5 * max_attempts}


def _to_result(question: str, final: dict, thread_id: str) -> AgentResult:
    attempts = final.get("attempts", [])
    common = dict(question=question, attempts=attempts, llm_calls=final.get("llm_calls", []),
                  assumptions=final.get("assumptions", []), thread_id=thread_id,
                  retrieved_tables=final.get("retrieved_tables"), value_hints=final.get("value_hints", []))
    if intr := final.get("__interrupt__"):
        payload = intr[0].value
        return AgentResult(outcome="clarifying", sql=None, reasoning=None, result=None, answer=None,
                           warning=None, error=None, clarification=payload["question"],
                           clarification_options=payload["options"], **common)
    final_sql = final.get("final_sql")
    reasoning = next((a.reasoning for a in attempts if a.sql == final_sql), None)
    return AgentResult(
        outcome=final["outcome"], sql=final_sql, reasoning=reasoning,
        result=final.get("result") if final["outcome"] in ("answered", "best_effort") else None,
        answer=final.get("answer"), warning=final.get("warning"), error=final.get("error"), **common,
    )


# --------------------------------------------------------------------------- helpers

def _fail(state: AgentState, status: str, *, error: str | None = None, critique: str | None = None,
          result: QueryResult | None = None, duration_ms: int = 0) -> dict:
    d = state["draft"]
    attempt = Attempt(
        n=len(state.get("attempts", [])) + 1, sql=d.sql, reasoning=d.reasoning, status=status,
        error=error, critique=critique, row_count=len(result.rows) if result else None,
        duration_ms=result.duration_ms if result else duration_ms,
        model=d.model, fell_back=d.fell_back, result=result,
    )
    return {"attempts": [attempt], "draft": None, "result": None}


def _pass(state: AgentState, warning: str | None = None) -> dict:
    d, result = state["draft"], state["result"]
    attempt = Attempt(
        n=len(state.get("attempts", [])) + 1, sql=d.sql, reasoning=d.reasoning, status="passed",
        row_count=len(result.rows), duration_ms=result.duration_ms, model=d.model, fell_back=d.fell_back,
    )
    update = {"attempts": [attempt], "final_sql": d.sql, "outcome": "answered"}
    if warning:
        update["warning"] = warning
    return update


def _explain_failure(state: AgentState) -> str:
    lines = [f"I couldn't find a reliable answer after {len(state['attempts'])} attempts."]
    for a in state["attempts"]:
        lines.append(f"- Attempt {a.n} ({a.status.replace('_', ' ')}): {a.critique or a.error}")
    lines.append("Try rephrasing the question, or be more specific about the names, dates or measures "
                 "you mean.")
    return "\n".join(lines)


def _q(state: AgentState) -> str:
    """The question plus any clarifications the user gave."""
    return with_clarifications(state["question"], state.get("clarifications", []))


def _catalog(state: AgentState) -> Catalog:
    return load_catalog(state["db_name"])


def _normalize(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, dialect="postgres").sql(dialect="postgres", normalize=True)
    except sqlglot.errors.ParseError:
        return " ".join(sql.lower().split())


def _referenced_tables(sql: str, catalog: Catalog) -> list[str]:
    try:
        names = {t.name.lower() for t in sqlglot.parse_one(sql, dialect="postgres").find_all(sqlglot.exp.Table)}
    except sqlglot.errors.ParseError:
        return []
    return [t.qualified for t in catalog.tables if t.name in names]


def _llm_call(step: str, r: client.LLMResult) -> LLMCall:
    return LLMCall(step, r.model, r.fell_back, r.latency_ms, r.prompt_tokens + r.completion_tokens)
