from pathlib import Path

import pytest

from app.datasets.relationships import detect
from app.datasets.tabular import TabularError, clean_identifier, coerces, infer_type, parse_file, unique_table_names

FIX = Path(__file__).parents[1] / "fixtures" / "sample_csv"


def parse_all():
    tables = [t for f in ("customers", "products", "orders") for t in parse_file(FIX / f"{f}.csv")]
    unique_table_names(tables)
    return {t.name: t for t in tables}


def test_clean_identifier():
    assert clean_identifier("Order Date") == "order_date"
    assert clean_identifier("2024 Sales") == "col_2024_sales"
    assert clean_identifier("  ") == "col"
    assert clean_identifier("User") == "user_"           # reserved word
    assert clean_identifier("Résumé (€)") == "r_sum"
    assert clean_identifier("x" * 90) == "x" * 50


@pytest.mark.parametrize("values,expected", [
    (["1", "2", None], "integer"), (["1", "3000000000"], "bigint"), (["1.5", "2"], "numeric"),
    (["2024-01-31", "2023-12-01"], "date"), (["2024-01-31 10:00:00", "2024-02-01T09:30"], "timestamp"),
    (["yes", "no"], "boolean"), (["true", "false", "true"], "boolean"),
    (["007", "010"], "text"),                       # codes with leading zeros stay text
    (["2024-13-45"], "text"), ([None, None], "text"), (["a", "1"], "text"), (["y", "n"], "text"),
])
def test_infer_type(values, expected):
    assert infer_type(values) == expected


def test_coerces():
    assert coerces(["1", "2"], "integer") and not coerces(["1", "x"], "integer")
    assert coerces(["x"], "text") and not coerces(["2024-02-30"], "date")


def test_csv_parsing_cleans_names_and_infers_types():
    t = parse_all()["customers"]
    assert [c.name for c in t.columns] == ["customer_id", "full_name", "country", "signup_date", "password_hint"]
    assert [c.type for c in t.columns] == ["integer", "text", "text", "date", "text"]
    assert len(t.rows) == 4


def test_semicolon_delimiter_and_bom(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes("\ufeffa;b\n1;2\n3;4\n".encode("utf-8"))
    t = parse_file(f)[0]
    assert [c.name for c in t.columns] == ["a", "b"] and t.rows == [["1", "2"], ["3", "4"]]


def test_duplicate_headers_are_made_unique(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("name,Name,\n1,2,3\n", encoding="utf-8")
    assert [c.name for c in parse_file(f)[0].columns] == ["name", "name_2", "column_3"]


def test_header_only_file_is_rejected(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(TabularError):
        parse_file(f)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "x.pdf"
    f.write_bytes(b"x")
    with pytest.raises(TabularError):
        parse_file(f)


def test_xlsx_sheets_become_tables(tmp_path):
    from datetime import datetime
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales Data"
    ws.append(["Day", "Amount", "Paid"])
    ws.append([datetime(2024, 5, 1), 10.5, True])
    ws.append([datetime(2024, 5, 2), 20.0, False])
    ws2 = wb.create_sheet("Notes")
    ws2.append(["id", "text"])
    ws2.append([1, "hello"])
    path = tmp_path / "book.xlsx"
    wb.save(path)
    tables = {t.name: t for t in parse_file(path)}
    assert set(tables) == {"sales_data", "notes"}
    assert [c.type for c in tables["sales_data"].columns] == ["date", "numeric", "boolean"] or \
        [c.type for c in tables["sales_data"].columns] == ["timestamp", "numeric", "boolean"]
    assert tables["notes"].rows == [["1", "hello"]]


def test_relationships_detected_in_sample_set():
    rels = {(r.from_table, r.from_column, r.to_table, r.to_column) for r in detect(list(parse_all().values()))}
    assert rels == {("orders", "customer_id", "customers", "customer_id"),
                    ("orders", "sku", "products", "sku")}


def test_no_relationship_when_values_do_not_overlap():
    t = parse_all()
    t["orders"].rows = [[r[0], "999", *r[2:]] for r in t["orders"].rows]
    rels = detect(list(t.values()))
    assert not [r for r in rels if r.from_column == "customer_id"]


def test_descriptions_yaml_round_trip():
    from app.datasets.descriptions import Descriptions, parse_yaml_text, to_yaml
    d = Descriptions({"orders": "One row per order."}, {"orders": {"status": "0=open, 1=paid"}},
                     {"revenue": "SUM(orders.total)"})
    assert parse_yaml_text(to_yaml(d)).to_dict() == d.to_dict()
