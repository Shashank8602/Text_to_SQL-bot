"""Compare schema-retrieval setups without any LLM calls (docs/DESIGN.md §6.3).

    python -m eval.retrieval_eval --embedders local,gemini --k 3,5,8

For each question, the tables the retriever would send (top-k + join-path tables) are compared with
the question's expected tables:
- recall: share of expected tables that were sent (averaged over questions)
- full coverage: share of questions where ALL expected tables were sent (what the SQL needs)
- tables / tokens sent: prompt size, versus sending the full schema
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from app.database.catalog import load_catalog
from app.datasets import descriptions as descriptions_store
from app.datasets import registry
from app.retrieval.retriever import glossary_tables, rank_tables, select, value_matches
from eval.run_eval import REPORTS, load_questions

ROOT = Path(__file__).parent
SETS = {"main": ROOT / "questions.yaml", "heldout": ROOT / "questions_heldout.yaml"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.retrieval_eval")
    p.add_argument("--dataset", default="pagila")
    p.add_argument("--embedders", default="local,gemini")
    p.add_argument("--k", default="3,5,8")
    p.add_argument("--sets", default="main,heldout")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ds = registry.get(args.dataset)
    if ds is None:
        print(f"Dataset '{args.dataset}' is not loaded.", file=sys.stderr)
        return 1
    catalog = load_catalog(ds.db_name)
    desc = descriptions_store.load(args.dataset)
    full_tokens = _tokens(catalog.to_ddl(descriptions=desc))
    ks = [int(k) for k in args.k.split(",")]

    rows = [("full schema (no retrieval)", "-", "100%", "100%", len(catalog.tables), full_tokens, "")]
    misses: list[str] = []
    for set_name in args.sets.split(","):
        questions = load_questions(SETS[set_name])
        for embedder in args.embedders.split(","):
            start = time.perf_counter()
            ranked = {q["id"]: rank_tables(args.dataset, q["question"], catalog, embedder_name=embedder)
                      for q in questions}
            matches = {q["id"]: value_matches(q["question"], catalog) for q in questions}
            per_q_ms = (time.perf_counter() - start) * 1000 / len(questions)
            for k, use_values in [(k, v) for k in ks for v in (False, True)]:
                recalls, covered, n_tables, n_tokens = [], 0, [], []
                for q in questions:
                    forced = glossary_tables(q["question"], desc.glossary, catalog) if use_values else []
                    sent = select(ranked[q["id"]], catalog, k, matches[q["id"]] if use_values else [], forced).tables
                    expected = set(q["tables"])
                    hit = expected & set(sent)
                    recalls.append(len(hit) / len(expected))
                    covered += hit == expected
                    n_tables.append(len(sent))
                    n_tokens.append(_tokens(catalog.to_ddl(sent, descriptions=desc)))
                    if hit != expected and k == max(ks) and use_values:
                        misses.append(f"{set_name}/{embedder} k={k} {q['id']}: missing "
                                      f"{sorted(expected - hit)} for {q['question']!r}")
                n = len(questions)
                rows.append((f"{embedder}{' + values/glossary' if use_values else ''} ({set_name})", k, f"{100 * sum(recalls) / n:.0f}%",
                             f"{covered}/{n} ({100 * covered / n:.0f}%)", round(sum(n_tables) / n, 1),
                             round(sum(n_tokens) / n), f"{per_q_ms:.0f} ms/question"))

    header = ("setup", "k", "recall", "full coverage", "avg tables", "avg tokens", "ranking time")
    table = _markdown([header] + rows)
    print(table)
    if misses:
        print("\nMisses at the largest k:\n" + "\n".join(f"- {m}" for m in misses))

    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{datetime.now():%Y%m%d-%H%M}_retrieval.md"
    path.write_text(f"# Retrieval comparison ({args.dataset})\n\n{table}\n\n## Misses at k={max(ks)}\n\n"
                    + "\n".join(f"- {m}" for m in misses) + "\n", encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


def _tokens(text: str) -> int:
    return len(text) // 4  # rough, model-independent


def _markdown(rows) -> str:
    out = ["| " + " | ".join(str(c) for c in rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
