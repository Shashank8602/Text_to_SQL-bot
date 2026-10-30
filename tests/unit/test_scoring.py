from datetime import date
from decimal import Decimal

import pytest

from app.database.executor import QueryResult
from eval.run_eval import load_questions
from eval.scoring import results_match


def test_question_file_is_valid():
    questions = load_questions()
    assert len(questions) == 40


def test_question_validation_catches_yaml_nulls(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text("questions:\n  - {id: x, question: q, gold_sql: SELECT 1, tags: [null], tables: [t]}\n")
    with pytest.raises(ValueError, match="must be a string"):
        load_questions(bad)


def r(*rows, cols=None):
    width = len(rows[0]) if rows else 1
    return QueryResult(columns=cols or [f"c{i}" for i in range(width)], column_types=["x"] * width,
                       rows=list(rows), truncated=False, duration_ms=0)


def test_single_value_any_column_and_numeric_rounding():
    assert results_match(r((Decimal("2.98"),)), r((2.98012,)))
    assert results_match(r((1000,)), r(("films", 1000)))       # extra column fine
    assert not results_match(r((1000,)), r((999,)))


def test_order_insensitive_by_default():
    gold = r(("Music",), ("Drama",))
    assert results_match(gold, r(("drama",), ("MUSIC",)))
    assert not results_match(gold, r(("Drama",),))             # missing a row
    assert not results_match(gold, r(("Drama",), ("Music",), ("Travel",)))


def test_order_matters_for_rankings():
    gold = r(("KARL", Decimal("358.12")), ("TIM", Decimal("344.06")))
    assert results_match(gold, r(("KARL", 358.12), ("TIM", 344.06)), order_matters=True)
    assert not results_match(gold, r(("TIM", 344.06), ("KARL", 358.12)), order_matters=True)


def test_rows_must_align_across_columns():
    gold = r(("a", 1), ("b", 2))
    assert results_match(gold, r(("b", 2), ("a", 1)))
    # same column values, but paired differently
    assert not results_match(gold, r(("a", 2), ("b", 1)))


def test_split_name_columns_match_full_name():
    gold = r(("PENELOPE GUINESS",),)
    assert results_match(gold, r((1, "Penelope", "Guiness"),))
    gold2 = r(("KARL", "SEAL", 358.12),)
    assert results_match(gold2, r(("Karl Seal", 358.12),))       # gold split, bot joined
    assert results_match(gold2, r(("KARL", "SEAL", 358.12, 7),))
    assert not results_match(gold2, r(("Karl Smith", 358.12),))


def test_dates_and_nulls():
    assert results_match(r((date(2024, 1, 1), None)), r((date(2024, 1, 1), None)))
    assert not results_match(r((None,)), r((0,)))


def test_empty_results():
    assert results_match(r(), r())
    assert not results_match(r((1,)), None)
