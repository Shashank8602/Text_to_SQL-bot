"""CSV/Excel upload path against the real database."""

from pathlib import Path

import psycopg
import pytest

from app.database.catalog import load_catalog
from app.database.connection import conninfo
from app.database.executor import execute_readonly
from app.datasets import registry
from app.datasets.loader import DatasetLoadError, delete_dataset
from app.datasets.relationships import detect
from app.datasets.tabular_loader import load_tabular, parse_files

from .conftest import drop_if_exists

FIX = Path(__file__).parents[1] / "fixtures" / "sample_csv"
SLUG = "pytest_csv"
FILES = [FIX / "customers.csv", FIX / "products.csv", FIX / "orders.csv"]


@pytest.fixture
def csv_dataset():
    drop_if_exists(SLUG)
    tables = parse_files(FILES)
    report = load_tabular(tables, "CSV test", slug=SLUG, relationships=detect(tables))
    yield report
    drop_if_exists(SLUG)


def test_loaded_and_queryable_as_bot_reader(csv_dataset):
    r = execute_readonly("SELECT count(*) FROM orders", db_name=csv_dataset.db_name)
    assert r.rows == [(6,)]
    r = execute_readonly("SELECT sum(price) FROM products", db_name=csv_dataset.db_name)
    assert float(r.rows[0][0]) == pytest.approx(258.4)
    assert registry.get(SLUG).source_type == "csv"


def test_sensitive_column_hidden_from_the_bot(csv_dataset):
    assert "customers.password_hint" in csv_dataset.hidden_columns
    with pytest.raises(Exception):
        execute_readonly("SELECT password_hint FROM customers", db_name=csv_dataset.db_name)
    cols = {c.name for t in load_catalog(csv_dataset.db_name).tables if t.name == "customers" for c in t.columns}
    assert "password_hint" not in cols and "customer_id" in cols


def test_confirmed_relationships_become_foreign_keys(csv_dataset):
    orders = next(t for t in load_catalog(csv_dataset.db_name).tables if t.name == "orders")
    declared = {(fk.columns[0], fk.ref_table, fk.inferred) for fk in orders.foreign_keys}
    assert ("customer_id", "customers", False) in declared
    assert ("sku", "products", False) in declared


def test_type_override_and_bad_override():
    tables = parse_files(FILES)
    drop_if_exists("pytest_csv2")
    try:
        load_tabular(tables, "x", slug="pytest_csv2", type_overrides={"orders.quantity": "bigint"})
        with psycopg.connect(conninfo("data_loader", "ds_pytest_csv2")) as c:
            assert c.execute("SELECT data_type FROM information_schema.columns "
                             "WHERE table_name='orders' AND column_name='quantity'").fetchone()[0] == "bigint"
    finally:
        drop_if_exists("pytest_csv2")
    with pytest.raises(DatasetLoadError):
        load_tabular(parse_files(FILES), "x", slug="pytest_csv3", type_overrides={"customers.country": "integer"})
    assert registry.get("pytest_csv3") is None


def test_failure_leaves_nothing_behind():
    tables = parse_files(FILES)
    with pytest.raises(DatasetLoadError):
        load_tabular(tables, "x", slug="pytest_csv4", hide=["nosuchtable.col"])
    assert registry.get("pytest_csv4") is None
    with psycopg.connect(conninfo("data_loader", "postgres")) as c:
        assert c.execute("SELECT 1 FROM pg_database WHERE datname='ds_pytest_csv4'").fetchone() is None


def test_data_is_never_executed_as_sql(tmp_path):
    f = tmp_path / "evil.csv"
    f.write_text("name\n\"x'); DROP TABLE evil; --\"\n=cmd|' /C calc'!A0\n", encoding="utf-8")
    drop_if_exists("pytest_evil")
    try:
        rep = load_tabular(parse_files([f]), "evil", slug="pytest_evil")
        assert execute_readonly("SELECT count(*) FROM evil", db_name=rep.db_name).rows == [(2,)]
    finally:
        drop_if_exists("pytest_evil")


def test_xlsx_dataset(tmp_path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Cities"
    ws.append(["City", "Population"])
    ws.append(["Paris", 2100000])
    ws.append(["Rome", 2800000])
    wb.create_sheet("Empty")
    path = tmp_path / "geo.xlsx"
    wb.save(path)
    drop_if_exists("pytest_xlsx")
    try:
        rep = load_tabular(parse_files([path]), "geo", slug="pytest_xlsx", source_type="excel")
        assert execute_readonly("SELECT max(population) FROM cities", db_name=rep.db_name).rows == [(2800000,)]
    finally:
        drop_if_exists("pytest_xlsx")
