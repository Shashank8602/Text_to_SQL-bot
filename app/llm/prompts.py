"""Prompt builders and structured-output schemas.

User text is always placed inside a delimited block and never interpolated into the
instructions (docs/DESIGN.md §7, prompt hygiene).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from app.database.executor import QueryResult
from app.llm.result_format import format_result


class SQLGeneration(BaseModel):
    reasoning: str = Field(description=(
        "Brief plan before writing SQL: the tables needed, how they join, filters, "
        "grouping/aggregation and ordering."
    ))
    sql: str = Field(description=(
        "One PostgreSQL SELECT statement that answers the question, or an empty string if the "
        "request cannot be answered with a read-only query (explain why in reasoning)."
    ))


class Answer(BaseModel):
    answer: str = Field(description="The answer for the user, in Markdown.")


class JudgeVerdict(BaseModel):
    verdict: Literal["pass", "fail"]
    critique: str = Field(description=(
        "If fail: what is wrong and how to fix the query, in one or two sentences. If pass: empty."
    ))


class FailedAttempt(Protocol):
    sql: str
    status: str
    error: str | None
    critique: str | None


GENERATE_SQL_SYSTEM = """\
You are an expert PostgreSQL analyst. Write ONE read-only SQL query that answers the user's question
using ONLY the tables and columns in the schema below.

Rules:
- PostgreSQL dialect. A single SELECT statement (CTEs allowed). Never modify data.
- Use only tables/columns that appear in the schema. Do not guess names.
- Join through foreign keys (or matching *_id columns when no key is declared). Use table aliases
  and qualify every column when more than one table is involved.
- For text filters on names/titles, use case-insensitive matching (ILIKE) unless the value is
  listed in the schema's "values:" comments, in which case use the exact value.
- Return human-readable columns (names, titles) rather than only ids. Give computed columns
  clear aliases.
- Add ORDER BY when the question implies ranking or ordering. Use LIMIT only when the question asks
  for a specific number of results ("top 5"); otherwise do not limit.
- For superlative questions ("which X has the most Y", "the highest", "the cheapest"), return only the
  top row(s), keeping ties: compare against MAX/MIN or filter on RANK() = 1. Do not return the whole
  ranking, and do not use LIMIT 1 (it silently drops ties).
- Relative dates ("last month", "this year") are relative to today's date: {today}.
- Do not end the query with a semicolon.

Schema:
{schema}
"""

ANSWER_SYSTEM = """\
You answer a user's question about their data, using ONLY the query result provided.

Rules:
- Lead with the direct answer in one or two sentences, then add supporting detail if useful
  (a short list or small Markdown table for multiple rows).
- Every number you state must appear in the result. Do not invent, extrapolate or round
  away precision that matters. Format money and large numbers readably.
- If the result is empty, say that no matching data was found; do not speculate about why.
- If the result was truncated, say the answer is based on the first rows shown.
- Do not mention SQL, tables or columns unless the user asked about them.
"""


JUDGE_SYSTEM = """\
You review whether a SQL query's result correctly answers a user's question.

Fail ONLY for clear problems, for example:
- the result answers a different question (wrong metric, wrong grouping, wrong entity)
- a filter, condition or time range the question asks for is missing or wrong
- the question asks for a number/ranking/name and the result cannot provide it
- values are clearly implausible (e.g. duplicated rows from a bad join inflating totals)
Do NOT fail for style, column naming, extra helpful columns, or ordering the question did not ask for.
If the result is empty, pass only if "no matching data" is a plausible answer to the question.
When in doubt, pass.
"""


def generate_sql_messages(
    question: str,
    schema_ddl: str,
    *,
    previous_attempts: Sequence[FailedAttempt] = (),
    today: date | None = None,
) -> list[dict]:
    system = GENERATE_SQL_SYSTEM.format(today=(today or date.today()).isoformat(), schema=schema_ddl)
    user = f"Question:\n<question>\n{question}\n</question>"
    if previous_attempts:
        user += "\n\n" + _feedback(previous_attempts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _feedback(attempts: Sequence[FailedAttempt]) -> str:
    parts = ["Your previous queries for this question failed. Learn from them and write a corrected query. "
             "Do not repeat a query that already failed."]
    for i, a in enumerate(attempts, 1):
        problem = a.critique or a.error or a.status
        parts.append(f"Attempt {i} ({a.status}):\n<sql>\n{a.sql}\n</sql>\nProblem: {problem}")
    return "\n\n".join(parts)


def judge_messages(question: str, sql: str, result: QueryResult, schema_ddl: str,
                   filter_notes: Sequence[str] = ()) -> list[dict]:
    user = (
        f"Question:\n<question>\n{question}\n</question>\n\n"
        f"Relevant schema:\n<schema>\n{schema_ddl}\n</schema>\n\n"
        f"SQL:\n<sql>\n{sql}\n</sql>\n\n"
        f"Result:\n<result>\n{format_result(result, preview_rows=10)}\n</result>"
    )
    if filter_notes:
        user += ("\n\nValues the query's pattern filters actually matched in the database "
                 "(check they are what the question means):\n<filter_matches>\n"
                 + "\n".join(filter_notes) + "\n</filter_matches>")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def answer_messages(question: str, sql: str, result: QueryResult, *, caveat: str | None = None) -> list[dict]:
    system = ANSWER_SYSTEM
    user = (
        f"Question:\n<question>\n{question}\n</question>\n\n"
        f"SQL that was run:\n<sql>\n{sql}\n</sql>\n\n"
        f"Result:\n<result>\n{format_result(result)}\n</result>"
    )
    if caveat:
        system += (
            "- A reviewer doubted that this result fully answers the question (see <reviewer_note>). "
            "After the answer, add one short sentence warning the user about that.\n"
        )
        user += f"\n\n<reviewer_note>\n{caveat}\n</reviewer_note>"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
