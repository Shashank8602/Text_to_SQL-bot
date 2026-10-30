"""Ask a question from the command line (the Streamlit chat comes in M6).

    python -m app.ask --dataset pagila "Which film category earned the most revenue?"
"""

from __future__ import annotations

import argparse
import sys

from app.llm.result_format import format_result
from app.pipeline import DatasetNotFound, answer_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ask")
    parser.add_argument("question")
    parser.add_argument("--dataset", required=True, help="dataset slug (see: python -m app.datasets.cli list)")
    parser.add_argument("--rows", type=int, default=10, help="result rows to print")
    args = parser.parse_args(argv)
    # Windows consoles default to cp1252; LLM output often contains other characters.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        r = answer_question(args.question, args.dataset)
    except DatasetNotFound as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if r.reasoning and r.status != "declined":
        print(f"Plan:\n  {r.reasoning}\n")
    if r.sql:
        print("SQL:\n" + "\n".join("  " + line for line in r.sql.splitlines()) + "\n")
    if r.result is not None:
        print("Result:\n" + "\n".join("  " + line for line in format_result(r.result, args.rows).splitlines()) + "\n")
    if r.answer:
        print(f"Answer:\n{r.answer}\n")
    if r.status == "declined":
        print(f"Declined: {r.error}\n")
    elif r.error:
        print(f"Failed ({r.status}): {r.error}\n")
    for c in r.llm_calls:
        note = "  (fallback)" if c.fell_back else ""
        print(f"[{c.step}: {c.model}, {c.latency_ms} ms, {c.tokens} tokens{note}]")
    return 0 if r.status == "answered" else 2


if __name__ == "__main__":
    sys.exit(main())
