from datetime import date
from decimal import Decimal

from app.charts import ChartSpec, build_figure, suggest_chart
from app.database.executor import QueryResult


def qr(columns, types, rows):
    return QueryResult(columns=columns, column_types=types, rows=rows, truncated=False, duration_ms=1)


def test_category_and_measure_gives_bar():
    r = qr(["category", "revenue"], ["text", "numeric"], [("Sports", Decimal("10")), ("Drama", Decimal("7"))])
    assert suggest_chart(r) == ChartSpec("bar", "category", "revenue", "revenue by category")


def test_date_axis_gives_line():
    r = qr(["month", "rentals"], ["date", "bigint"], [(date(2022, m, 1), m * 10) for m in range(1, 5)])
    assert suggest_chart(r).type == "line"


def test_single_row_or_single_column_is_a_table():
    assert suggest_chart(qr(["n"], ["bigint"], [(1,), (2,)])) is None
    assert suggest_chart(qr(["a", "b"], ["text", "int4"], [("x", 1)])) is None


def test_id_columns_are_not_measures():
    r = qr(["name", "customer_id"], ["text", "int4"], [("a", 1), ("b", 2)])
    assert suggest_chart(r) is None


def test_repeated_categories_are_not_barred():
    r = qr(["country", "n"], ["text", "int8"], [("UK", 1), ("UK", 2), ("FR", 3)])
    assert suggest_chart(r) is None


def test_too_many_rows_is_a_table():
    r = qr(["name", "n"], ["text", "int8"], [(str(i), i) for i in range(100)])
    assert suggest_chart(r) is None


def test_two_measures_give_scatter():
    r = qr(["length", "rate"], ["int4", "numeric"], [(i, Decimal(i) / 2) for i in range(10)])
    assert suggest_chart(r).type == "scatter"


def test_figure_builds_from_decimals():
    r = qr(["category", "revenue"], ["text", "numeric"], [("Sports", Decimal("10.5")), ("Drama", Decimal("7"))])
    fig = build_figure(suggest_chart(r), r)
    assert list(fig.data[0].y) == [10.5, 7.0]
