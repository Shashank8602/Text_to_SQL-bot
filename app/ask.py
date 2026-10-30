"""Ask a question from the command line (the Streamlit chat comes in M6).

    python -m app.ask --dataset pagila "Which film category earned the most revenue?"
"""

from __future__ import annotations

import argparse
import sys

from app.agent.graph import DEFAULT_MAX_ATTEMPTS, run_agent
from app.datasets import registry
from app.llm.result_format import format_result

_ICONS = {"passed": "✓", "validation_failed": "✗", "db_error": "✗", "timeout": "✗", "unsafe": "✗", "duplicate": "✗"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ask")
    parser.add_argument("question")
    parser.add_argument("--dataset", required=True, help="dataset slug (see: python -m app.datasets.cli list)")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--rows", type=int, default=10, help="result rows to print")
    args = parser.parse_args(argv)
    # Windows consoles default to cp1252; LLM output often contains other characters.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ds = registry.get(args.dataset)
    if ds is None or ds.status != "ready":
        print(f"Error: no ready dataset named '{args.dataset}'.", file=sys.stderr)
        return 1

    r = run_agent(args.question, ds.db_name, max_attempts=args.max_attempts)

    if r.attempts:
        print(f"Agent trace ({len(r.attempts)} attempt{'s' if len(r.attempts) != 1 else ''}):")
        for a in r.attempts:
            via = f"{a.model}{', fallback' if a.fell_back else ''}"
            rows = f", {a.row_count} rows" if a.row_count is not None else ""
            print(f"  {_ICONS[a.status]} #{a.n} {a.status} ({via}{rows})")
            if a.status != "passed":
                print("      " + " ".join(a.sql.split())[:160])
                print(f"      → {a.critique or a.error}")
        print()
    if r.sql:
        print("SQL:\n" + "\n".join("  " + line for line in r.sql.splitlines()) + "\n")
    if r.result is not None:
        print("Result:\n" + "\n".join("  " + line for line in format_result(r.result, args.rows).splitlines()) + "\n")
    if r.outcome == "declined":
        print(f"Declined: {r.error}\n")
    elif r.answer:
        print(f"Answer{' (best effort)' if r.outcome == 'best_effort' else ''}:\n{r.answer}\n")
    if r.warning and r.outcome == "answered":
        print(f"Note: {r.warning}\n")
    if r.error and r.outcome != "declined":
        print(f"Error ({r.outcome}): {r.error}\n")
    for c in r.llm_calls:
        print(f"[{c.step}: {c.model}, {c.latency_ms} ms, {c.tokens} tokens{', fallback' if c.fell_back else ''}]")
    return 0 if r.outcome in ("answered", "best_effort") else 2


if __name__ == "__main__":
    sys.exit(main())
