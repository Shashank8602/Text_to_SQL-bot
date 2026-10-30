from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from app.database.executor import QueryResult

AttemptStatus = Literal["unsafe", "duplicate", "db_error", "timeout", "validation_failed", "passed"]
Outcome = Literal["answered", "best_effort", "gave_up", "declined", "llm_error"]


@dataclass
class Attempt:
    n: int
    sql: str
    reasoning: str
    status: AttemptStatus
    error: str | None = None       # safety reason / database error
    critique: str | None = None    # validator feedback
    row_count: int | None = None
    duration_ms: int = 0           # query execution time
    model: str = ""                # deployment that wrote the SQL
    fell_back: bool = False
    result: QueryResult | None = None  # kept for validation failures (best-effort answer)


@dataclass
class LLMCall:
    step: str
    model: str
    fell_back: bool
    latency_ms: int
    tokens: int


@dataclass
class Draft:
    """SQL produced by generate_sql, not yet checked/executed/validated."""
    sql: str
    reasoning: str
    model: str
    fell_back: bool


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    db_name: str
    max_attempts: int
    compose: bool              # False skips the final answer call (used by the eval)
    # working state
    schema_ddl: str
    draft: Draft | None
    result: QueryResult | None
    attempts: Annotated[list[Attempt], operator.add]
    llm_calls: Annotated[list[LLMCall], operator.add]
    # outputs
    outcome: Outcome
    final_sql: str | None
    answer: str | None
    warning: str | None
    error: str | None
