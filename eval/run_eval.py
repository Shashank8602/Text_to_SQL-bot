"""Run the evaluation suite against the agent (docs/DESIGN.md §10).

    python -m eval.run_eval                       # all questions, 3 attempts
    python -m eval.run_eval --ids q01,q23 --delay 0
    python -m eval.run_eval --max-attempts 1 --label single-shot

One run with max_attempts=3 also yields the single-shot baseline: the agent keeps the result of
every rejected attempt, so attempt 1 can be scored on its own.

Writes eval/reports/<timestamp>_<label>.md and .json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from app.agent.graph import DEFAULT_MAX_ATTEMPTS, AgentResult, run_agent
from app.database.executor import QueryResult, execute_readonly
from app.datasets import registry
from eval.scoring import results_match

ROOT = Path(__file__).parent
QUESTIONS = ROOT / "questions.yaml"
REPORTS = ROOT / "reports"


@dataclass
class QuestionScore:
    id: str
    question: str
    tags: list[str]
    outcome: str
    correct: bool
    attempt1_correct: bool
    statuses: list[str]
    false_rejections: int            # attempts rejected by the validator whose result was actually right
    llm_calls: int
    fallbacks: int
    seconds: float
    sql: str | None
    error: str | None = None
    attempt_sql: list[str] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.run_eval")
    p.add_argument("--dataset", default="pagila")
    p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    p.add_argument("--ids", help="comma-separated question ids")
    p.add_argument("--tags", help="comma-separated tags; questions with any of them")
    p.add_argument("--delay", type=float, default=2.0, help="seconds between questions (rate limits)")
    p.add_argument("--label", default=None)
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ds = registry.get(args.dataset)
    if ds is None or ds.status != "ready":
        print(f"Dataset '{args.dataset}' is not loaded. Load the sample first:\n"
              "  python -m app.datasets.cli load sample_data/pagila/pagila.sql.gz --name Pagila", file=sys.stderr)
        return 1

    questions = load_questions()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.tags:
        wanted = set(args.tags.split(","))
        questions = [q for q in questions if wanted & set(q["tags"])]

    label = args.label or f"attempts{args.max_attempts}"
    REPORTS.mkdir(exist_ok=True)
    base = REPORTS / f"{datetime.now():%Y%m%d-%H%M}_{label}"
    progress = base.with_suffix(".jsonl")  # one line per question, written as we go
    print(f"Evaluating {len(questions)} questions on '{args.dataset}' (max {args.max_attempts} attempts)")
    print(f"Progress file: {progress}\n")
    scores = []
    for i, q in enumerate(questions, 1):
        if i > 1 and args.delay:
            time.sleep(args.delay)
        s = score_question(q, ds.db_name, args.max_attempts)
        scores.append(s)
        with progress.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(s), default=str) + "\n")
        mark = "✓" if s.correct else "✗"
        first = "" if s.attempt1_correct == s.correct else (" (fixed by retry)" if s.correct else " (broken by retry)")
        print(f"{mark} {s.id} {s.outcome:<11} attempts={len(s.statuses)} {s.seconds:5.1f}s  {s.question}{first}")

    summary = summarize(scores, args.max_attempts)
    path = write_report(scores, summary, label, args, base)
    print("\n" + format_summary(summary))
    print(f"\nReport: {path}")
    return 0


def score_question(q: dict, db_name: str, max_attempts: int) -> QuestionScore:
    gold = execute_readonly(q["gold_sql"], db_name=db_name)
    order = bool(q.get("order_matters"))
    start = time.perf_counter()
    try:
        r = run_agent(q["question"], db_name, max_attempts=max_attempts, compose=False)
        if r.outcome == "llm_error":  # provider limits: wait for the minute window, retry once
            time.sleep(60)
            r = run_agent(q["question"], db_name, max_attempts=max_attempts, compose=False)
    except Exception as e:  # a crash is a bug to report, not a reason to stop the run
        return QuestionScore(q["id"], q["question"], q["tags"], "crash", False, False, [], 0, 0, 0,
                             time.perf_counter() - start, None, error=f"{type(e).__name__}: {e}")
    seconds = time.perf_counter() - start

    final_result = r.result if r.outcome in ("answered", "best_effort") else None
    attempt_results = [_attempt_result(a, r) for a in r.attempts]
    return QuestionScore(
        id=q["id"], question=q["question"], tags=q["tags"], outcome=r.outcome,
        correct=results_match(gold, final_result, order_matters=order),
        attempt1_correct=bool(attempt_results) and results_match(gold, attempt_results[0], order_matters=order),
        statuses=[a.status for a in r.attempts],
        false_rejections=sum(1 for a, res in zip(r.attempts, attempt_results)
                             if a.status == "validation_failed" and results_match(gold, res, order_matters=order)),
        llm_calls=len(r.llm_calls),
        fallbacks=sum(c.fell_back for c in r.llm_calls),
        seconds=round(seconds, 2),
        sql=r.sql,
        error=r.error,
        attempt_sql=[a.sql for a in r.attempts],
    )


def _attempt_result(attempt, r: AgentResult) -> QueryResult | None:
    if attempt.status == "passed":
        return r.result
    return attempt.result  # set for validation failures; None for errors


def summarize(scores: list[QuestionScore], max_attempts: int) -> dict:
    n = len(scores)
    first_wrong = [s for s in scores if not s.attempt1_correct]
    rejected = sum(s.statuses.count("validation_failed") for s in scores)
    by_tag: dict[str, list[bool]] = {}
    for s in scores:
        for t in s.tags:
            by_tag.setdefault(t, []).append(s.correct)
    secs = sorted(s.seconds for s in scores)
    return {
        "questions": n,
        "max_attempts": max_attempts,
        "accuracy": _pct(sum(s.correct for s in scores), n),
        "single_shot_accuracy": _pct(sum(s.attempt1_correct for s in scores), n),
        "recovery_rate": _pct(sum(s.correct for s in first_wrong), len(first_wrong)),
        "broken_by_retry": sum(1 for s in scores if s.attempt1_correct and not s.correct),
        "validator_rejections": rejected,
        "validator_false_rejections": sum(s.false_rejections for s in scores),
        "avg_attempts": round(statistics.mean(len(s.statuses) for s in scores), 2) if n else 0,
        "avg_llm_calls": round(statistics.mean(s.llm_calls for s in scores), 2) if n else 0,
        "fallback_calls": sum(s.fallbacks for s in scores),
        "p50_seconds": secs[n // 2] if n else 0,
        "p95_seconds": secs[min(n - 1, int(n * 0.95))] if n else 0,
        "outcomes": dict(sorted({o: sum(s.outcome == o for s in scores) for o in {s.outcome for s in scores}}.items())),
        "accuracy_by_tag": {t: _pct(sum(v), len(v)) for t, v in sorted(by_tag.items())},
    }


def format_summary(m: dict) -> str:
    lines = [
        f"Accuracy (with self-correction): {m['accuracy']}",
        f"Single-shot accuracy (attempt 1): {m['single_shot_accuracy']}",
        f"Recovery rate (wrong first, right in the end): {m['recovery_rate']}",
        f"Broken by retry: {m['broken_by_retry']}",
        f"Validator rejections: {m['validator_rejections']} ({m['validator_false_rejections']} of them were actually correct)",
        f"Avg attempts: {m['avg_attempts']} · Avg LLM calls: {m['avg_llm_calls']} · Fallback calls: {m['fallback_calls']}",
        f"Latency p50/p95: {m['p50_seconds']}s / {m['p95_seconds']}s",
        f"Outcomes: {m['outcomes']}",
    ]
    return "\n".join(lines)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    questions = yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids in questions.yaml")
    for q in questions:
        missing = {"id", "question", "gold_sql", "tags", "tables"} - q.keys()
        if missing:
            raise ValueError(f"{q.get('id')}: missing {sorted(missing)}")
        # YAML turns unquoted null/yes/no/1 into non-strings; tags and tables must stay names.
        for key in ("tags", "tables"):
            if not all(isinstance(v, str) for v in q[key]):
                raise ValueError(f"{q['id']}: every entry in '{key}' must be a string (quote it): {q[key]}")
    return questions


def write_report(scores: list[QuestionScore], summary: dict, label: str, args, base: Path) -> Path:
    base.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "questions": [asdict(s) for s in scores]}, indent=2, default=str),
        encoding="utf-8")

    md = [f"# Eval report: {label}", "",
          f"{datetime.now():%Y-%m-%d %H:%M} · dataset `{args.dataset}` · {summary['questions']} questions · "
          f"max {summary['max_attempts']} attempts", "", "## Summary", "", "| Metric | Value |", "|---|---|"]
    for k, v in summary.items():
        if k not in ("accuracy_by_tag", "outcomes"):
            md.append(f"| {k.replace('_', ' ')} | {v} |")
    md.append(f"| outcomes | {summary['outcomes']} |")
    md += ["", "## Accuracy by tag", "", "| Tag | Accuracy |", "|---|---|"]
    md += [f"| {t} | {a} |" for t, a in summary["accuracy_by_tag"].items()]
    md += ["", "## Questions", "", "| | Id | Question | Outcome | Attempts | 1st try | Seconds |", "|---|---|---|---|---|---|---|"]
    for s in scores:
        md.append(f"| {'✓' if s.correct else '✗'} | {s.id} | {s.question} | {s.outcome} | "
                  f"{' → '.join(s.statuses) or '-'} | {'✓' if s.attempt1_correct else '✗'} | {s.seconds} |")
    wrong = [s for s in scores if not s.correct]
    if wrong:
        md += ["", "## Incorrect answers", ""]
        for s in wrong:
            md += [f"### {s.id}: {s.question}", "", f"Outcome: {s.outcome}" + (f" · {s.error}" if s.error else ""), ""]
            for i, sql in enumerate(s.attempt_sql, 1):
                md += [f"Attempt {i} ({s.statuses[i - 1]}):", "```sql", sql, "```"]
            md.append("")
    base.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return base.with_suffix(".md")


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "n/a"


if __name__ == "__main__":
    sys.exit(main())
