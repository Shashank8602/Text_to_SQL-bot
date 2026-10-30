"""M2 single-shot pipeline: question -> SQL -> safety check -> execute -> answer.

No retries yet: any failure is returned as-is. M3 replaces this with the LangGraph agent and its
self-correction loop, reusing the same building blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.database.catalog import load_catalog
from app.database.executor import QueryError, QueryResult, execute_readonly
from app.database.safety import check_sql
from app.datasets import registry
from app.llm import client
from app.llm.prompts import Answer, SQLGeneration, answer_messages, generate_sql_messages


@dataclass
class LLMCall:
    step: str
    model: str
    fell_back: bool
    latency_ms: int
    tokens: int


@dataclass
class PipelineResult:
    question: str
    status: Literal["answered", "declined", "unsafe", "db_error", "timeout", "llm_error"]
    sql: str | None = None
    reasoning: str | None = None
    result: QueryResult | None = None
    answer: str | None = None
    error: str | None = None
    llm_calls: list[LLMCall] = field(default_factory=list)


class DatasetNotFound(Exception):
    pass


def answer_question(question: str, dataset: str) -> PipelineResult:
    ds = registry.get(dataset)
    if ds is None or ds.status != "ready":
        raise DatasetNotFound(f"No ready dataset named '{dataset}'.")
    catalog = load_catalog(ds.db_name)
    out = PipelineResult(question=question, status="answered")

    try:
        gen = client.call("generator", generate_sql_messages(question, catalog.to_ddl()), SQLGeneration)
    except client.LLMError as e:
        return _fail(out, "llm_error", str(e))
    out.llm_calls.append(_record("generate_sql", gen))
    out.sql, out.reasoning = gen.data.sql.strip(), gen.data.reasoning
    if not out.sql:
        # The model refused (e.g. a request to modify data). M5 handles this up front.
        out.sql = None
        return _fail(out, "declined", out.reasoning)

    safety = check_sql(out.sql, allowed_tables=catalog.table_names, allowed_schemas=catalog.schemas)
    if not safety.ok:
        return _fail(out, "unsafe", safety.reason)

    try:
        out.result = execute_readonly(out.sql, db_name=ds.db_name)
    except QueryError as e:
        return _fail(out, e.kind, str(e))

    try:
        ans = client.call("generator", answer_messages(question, out.sql, out.result), Answer)
    except client.LLMError as e:
        return _fail(out, "llm_error", str(e))
    out.llm_calls.append(_record("compose_answer", ans))
    out.answer = ans.data.answer
    return out


def _fail(out: PipelineResult, status: str, error: str) -> PipelineResult:
    out.status, out.error = status, error
    return out


def _record(step: str, r: client.LLMResult) -> LLMCall:
    return LLMCall(step, r.model, r.fell_back, r.latency_ms, r.prompt_tokens + r.completion_tokens)
