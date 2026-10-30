"""Prompt builders and structured-output schemas.

User text is always placed inside a delimited block and never interpolated into the
instructions (docs/DESIGN.md §7, prompt hygiene).
"""

from __future__ import annotations

from datetime import date

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


def generate_sql_messages(question: str, schema_ddl: str, *, today: date | None = None) -> list[dict]:
    system = GENERATE_SQL_SYSTEM.format(today=(today or date.today()).isoformat(), schema=schema_ddl)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question:\n<question>\n{question}\n</question>"},
    ]


def answer_messages(question: str, sql: str, result: QueryResult) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": (
            f"Question:\n<question>\n{question}\n</question>\n\n"
            f"SQL that was run:\n<sql>\n{sql}\n</sql>\n\n"
            f"Result:\n<result>\n{format_result(result)}\n</result>"
        )},
    ]
