import pytest

from app.database.catalog import Catalog, Column, ForeignKey, Table, _infer_foreign_keys
from app.datasets.descriptions import Descriptions, parse_yaml, validate
from app.retrieval.retriever import connect_tables, fuse, glossary_tables, select
from app.retrieval.values import ValueIndex


def table(name, cols, pk=None, fks=(), kind="table"):
    return Table("public", name, kind, 10, columns=[Column(c, "integer", False) for c in cols],
                 primary_key=pk or [], foreign_keys=list(fks))


@pytest.fixture
def catalog():
    # customer -> address -> city -> country ; payment has no declared FKs (like Pagila)
    tables = [
        table("country", ["country_id", "country"], ["country_id"]),
        table("city", ["city_id", "city", "country_id"], ["city_id"], [ForeignKey(["country_id"], "country", ["country_id"])]),
        table("address", ["address_id", "city_id"], ["address_id"], [ForeignKey(["city_id"], "city", ["city_id"])]),
        table("customer", ["customer_id", "address_id"], ["customer_id"], [ForeignKey(["address_id"], "address", ["address_id"])]),
        table("payment", ["payment_id", "customer_id", "amount"], ["payment_id"]),
        table("customer_list", ["id", "country"], kind="view"),
    ]
    _infer_foreign_keys(tables)
    return Catalog("ds_x", tables)


def test_foreign_keys_inferred_from_naming(catalog):
    payment = next(t for t in catalog.tables if t.name == "payment")
    assert [(fk.columns, fk.ref_table, fk.inferred) for fk in payment.foreign_keys] == [
        (["customer_id"], "customer", True)]
    customer = next(t for t in catalog.tables if t.name == "customer")
    assert all(not fk.inferred for fk in customer.foreign_keys)  # declared FK not duplicated


def test_fuse_rewards_agreement():
    ranked = fuse([["a", "b", "c"], ["b", "a"]])
    assert [t for t, _ in ranked][:2] in (["a", "b"], ["b", "a"])
    assert dict(ranked)["c"] < dict(ranked)["a"]


def test_join_path_adds_bridge_tables(catalog):
    assert connect_tables(["payment", "country"], catalog) == ["customer", "address", "city"]
    assert connect_tables(["payment"], catalog) == []
    assert connect_tables(["customer_list", "country"], catalog) == []  # views have no FK edges


def test_select_forces_value_and_glossary_tables(catalog):
    ranked = [("customer", 0.9), ("customer_list", 0.8), ("payment", 0.1)]
    r = select(ranked, catalog, k=1, forced=["country"])
    assert r.selected == ["customer", "country"]
    assert r.tables == ["customer", "country", "address", "city"]


def test_glossary_tables(catalog):
    glossary = {"revenue": "SUM(payment.amount)", "customer's country": "customer -> address -> city -> country"}
    assert glossary_tables("What was total revenue last year?", glossary, catalog) == ["payment"]
    assert glossary_tables("Which films?", glossary, catalog) == []


class TestValueIndex:
    idx = ValueIndex([
        ("country", "country", "Canada"), ("country", "country", "United States"),
        ("city", "city", "Varanasi (Benares)"), ("category", "name", "New"),
        ("film", "title", "ACADEMY DINOSAUR"), ("film", "title", "ACADEMY"),
        ("address", "postal_code", "2024"),
    ])

    def hints(self, q):
        return [m.hint() for m in self.idx.find(q)]

    def test_single_and_multi_word_values(self):
        assert self.hints("customers in canada") == ["country.country = 'Canada'"]
        assert self.hints("revenue in the United States") == ["country.country = 'United States'"]

    def test_parenthesized_alias_ignored_for_matching(self):
        assert self.hints("Who lives in Varanasi?") == ["city.city = 'Varanasi (Benares)'"]

    def test_longer_values_first(self):
        assert self.hints("copies of academy dinosaur")[0] == "film.title = 'ACADEMY DINOSAUR'"

    def test_stopwords_and_numbers_ignored(self):
        assert self.hints("How many new customers in 2024?") == []


def test_descriptions_parse_and_validate(tmp_path, catalog):
    f = tmp_path / "d.yaml"
    f.write_text("tables:\n  payment:\n    description: Money.\n    columns:\n      amount: Dollars.\n"
                 "      nope: x\n  ghost:\n    description: y\nglossary:\n  revenue: SUM(payment.amount)\n")
    d = parse_yaml(f)
    assert d.table("payment") == "Money." and d.column("payment", "amount") == "Dollars."
    assert validate(d, catalog) == ["unknown column 'payment.nope'", "unknown table 'ghost'"]


def test_descriptions_rendered_in_ddl(catalog):
    d = Descriptions(tables={"payment": "Money paid."}, columns={"payment": {"amount": "Dollars."}},
                     glossary={"revenue": "SUM(payment.amount)"})
    ddl = catalog.to_ddl(["payment"], descriptions=d)
    assert "-- table, ~10 rows: Money paid." in ddl
    assert "  amount integer NOT NULL  -- Dollars." in ddl
    assert "customer_id integer NOT NULL REFERENCES customer(customer_id)," in ddl
    assert "-- revenue: SUM(payment.amount)" in ddl
