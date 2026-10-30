import gzip

import psycopg
import pytest

from app.database.connection import conninfo
from app.database.executor import QueryError, execute_readonly
from app.datasets import registry
from app.datasets.loader import DatasetLoadError, load_pg_dump

from .conftest import drop_if_exists

SLUG = "pytest_small"

SMALL_DUMP = """\
SET statement_timeout = 0;
SELECT pg_catalog.set_config('search_path', '', false);
CREATE SCHEMA sales;
ALTER SCHEMA sales OWNER TO someone_else;
CREATE TABLE sales.customers (customer_id int PRIMARY KEY, name text, api_key text);
CREATE TABLE sales.orders (order_id int PRIMARY KEY, customer_id int REFERENCES sales.customers, total numeric);
GRANT ALL ON sales.orders TO PUBLIC;
COPY sales.customers (customer_id, name, api_key) FROM stdin;
1\tAda\tk-123
2\tGrace\t\\N
\\.
COPY sales.orders (order_id, customer_id, total) FROM stdin;
10\t1\t99.50
11\t2\t5.00
\\.
"""


@pytest.fixture
def write_dump(tmp_path):
    def _write(text: str, gz: bool = False):
        path = tmp_path / ("dump.sql.gz" if gz else "dump.sql")
        if gz:
            with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
                f.write(text)
        else:
            path.write_text(text, encoding="utf-8", newline="")
        return path
    return _write


@pytest.fixture(autouse=True)
def cleanup():
    drop_if_exists(SLUG)
    yield
    drop_if_exists(SLUG)


def _database_exists(db_name: str) -> bool:
    try:
        psycopg.connect(conninfo("data_loader", db_name)).close()
        return True
    except psycopg.OperationalError:
        return False


class TestSuccessfulLoad:
    def test_custom_schema_loaded_and_queryable(self, write_dump):
        report = load_pg_dump(write_dump(SMALL_DUMP, gz=True), "Small", slug=SLUG)
        rows = execute_readonly(
            "SELECT c.name, o.total FROM sales.customers c JOIN sales.orders o USING (customer_id) ORDER BY 1",
            db_name=report.db_name,
        ).rows
        assert [r[0] for r in rows] == ["Ada", "Grace"]
        assert report.statements_skipped == 3  # SET timeout, OWNER TO, GRANT
        assert registry.get(SLUG).status == "ready"

    def test_sensitive_column_hidden_by_default(self, write_dump):
        report = load_pg_dump(write_dump(SMALL_DUMP), "Small", slug=SLUG)
        assert report.hidden_columns == ["sales.customers.api_key"]
        with pytest.raises(QueryError):
            execute_readonly("SELECT api_key FROM sales.customers", db_name=report.db_name)

    def test_expose_and_hide_overrides(self, write_dump):
        report = load_pg_dump(
            write_dump(SMALL_DUMP), "Small", slug=SLUG,
            expose=["customers.api_key"], hide=["sales.orders"],
        )
        assert execute_readonly("SELECT api_key FROM sales.customers", db_name=report.db_name).rows
        with pytest.raises(QueryError):
            execute_readonly("SELECT * FROM sales.orders", db_name=report.db_name)

    def test_unknown_override_rejected(self, write_dump):
        with pytest.raises(DatasetLoadError, match="no_such"):
            load_pg_dump(write_dump(SMALL_DUMP), "Small", slug=SLUG, hide=["no_such.col"])
        assert registry.get(SLUG) is None


class TestRejectedDumps:
    @pytest.mark.parametrize(
        ("payload", "error"),
        [
            ("\\! echo pwned > /tmp/pwned", "meta-command"),
            ("CREATE ROLE evil SUPERUSER;", "server-level"),
            ("ALTER ROLE data_loader SET statement_timeout = 0;", "server-level"),
            ("COPY sales.customers FROM PROGRAM 'id';", "only allowed FROM stdin"),
            ("DO $$ BEGIN PERFORM 1; END $$;", "DO"),
            ("CREATE EXTENSION dblink;", "Dump failed"),          # untrusted: needs superuser
            ("SELECT pg_read_file('/etc/passwd');", "Dump failed"),  # needs pg_read_server_files
        ],
    )
    def test_malicious_dump_fails_and_leaves_nothing_behind(self, write_dump, payload, error):
        with pytest.raises(DatasetLoadError, match=error):
            load_pg_dump(write_dump(SMALL_DUMP + payload + "\n"), "Small", slug=SLUG)
        assert registry.get(SLUG) is None
        assert not _database_exists(f"ds_{SLUG}")

    def test_restore_time_limit(self, write_dump):
        with pytest.raises(DatasetLoadError, match="time limit"):
            load_pg_dump(write_dump("SELECT pg_sleep(30);\n"), "Slow", slug=SLUG, restore_timeout_s=1)
        assert not _database_exists(f"ds_{SLUG}")

    def test_custom_format_explains_alternative(self, tmp_path):
        path = tmp_path / "db.dump"
        path.write_bytes(b"PGDMP\x01\x0e\x00")
        with pytest.raises(DatasetLoadError, match="--format=plain"):
            load_pg_dump(path, "Custom", slug=SLUG)

    def test_duplicate_slug(self, write_dump):
        load_pg_dump(write_dump(SMALL_DUMP), "Small", slug=SLUG)
        with pytest.raises(Exception, match="already exists"):
            load_pg_dump(write_dump(SMALL_DUMP), "Small", slug=SLUG)
        assert registry.get(SLUG).status == "ready"  # the original survives
