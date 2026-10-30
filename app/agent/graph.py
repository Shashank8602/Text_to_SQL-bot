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

from dataclasses import dataclass, field
from functools import lru_cache

import sqlglot
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState, Attempt, Draft, LLMCall, Outcome
from app.agent.validation import check_result, column_hint, pattern_match_notes
from app.database.catalog import Catalog, load_catalog
from app.database.executor import QueryError, QueryResult, execute_readonly
from app.database.safety import check_sql
from app.llm import client
from app.llm.prompts import (
    Answer, JudgeVerdict, SQLGeneration, answer_messages, generate_sql_messages, judge_messages,
)

DEFAULT_MAX_ATTEMPTS = 3
_UNDEFINED_OBJECT_STATES = {"42703", "42P01"}  # undefined_column, undefined_table


# --------------------------------------------------------------------------- nodes

def retrieve_schema(state: AgentState) -> dict:
    # M4 replaces this with embedding retrieval of the relevant tables.
    return {"schema_ddl": _catalog(state).to_ddl()}


def generate_sql(state: AgentState) -> dict:
    messages = generate_sql_messages(
        state["question"], state["schema_ddl"], previous_attempts=state.get("attempts", []))
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
    messages = judge_messages(state["question"], draft.sql, result, catalog.to_ddl(referenced or None),
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
    try:
        r = client.call("generator", answer_messages(state["question"], state["final_sql"], state["result"],
                                                     caveat=caveat), Answer)
    except client.LLMError as e:
        return {"outcome": outcome, "answer": None, "error": f"Could not compose an answer: {e}"}
    return {"outcome": outcome, "answer": r.data.answer, "llm_calls": [_llm_call("compose_answer", r)]}


# --------------------------------------------------------------------------- routing

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


@lru_cache
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve_schema", retrieve_schema)
    g.add_node("generate_sql", generate_sql)
    g.add_node("check_sql", check)
    g.add_node("execute_sql", execute)
    g.add_node("validate_result", validate)
    g.add_node("finish_failure", finish_failure)
    g.add_node("compose_answer", compose_answer)

    g.add_edge(START, "retrieve_schema")
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
    return g.compile()


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


def run_agent(question: str, db_name: str, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
              compose: bool = True) -> AgentResult:
    final = build_graph().invoke(
        {"question": question, "db_name": db_name, "max_attempts": max_attempts, "compose": compose,
         "attempts": [], "llm_calls": [], "draft": None, "result": None},
        config={"recursion_limit": 10 + 5 * max_attempts},
    )
    attempts = final.get("attempts", [])
    final_sql = final.get("final_sql")
    reasoning = next((a.reasoning for a in attempts if a.sql == final_sql), None)
    return AgentResult(
        question=question,
        outcome=final["outcome"],
        sql=final_sql,
        reasoning=reasoning,
        result=final.get("result") if final["outcome"] in ("answered", "best_effort") else None,
        answer=final.get("answer"),
        warning=final.get("warning"),
        error=final.get("error"),
        attempts=attempts,
        llm_calls=final.get("llm_calls", []),
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
