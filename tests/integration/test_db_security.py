import psycopg
import pytest

from app.config import get_settings
from app.database.connection import conninfo
from app.database.executor import QueryError, execute_readonly


class TestExecutor:
    def test_returns_rows_columns_and_types(self, pagila_db):
        result = execute_readonly(
            "SELECT film_id, title, rental_rate FROM film ORDER BY film_id LIMIT 3", db_name=pagila_db
        )
        assert result.columns == ["film_id", "title", "rental_rate"]
        assert result.column_types == ["int4", "text", "numeric"]
        assert len(result.rows) == 3
        assert not result.truncated

    def test_dataset_fully_loaded(self, pagila_db):
        counts = execute_readonly(
            "SELECT (SELECT count(*) FROM film), (SELECT count(*) FROM rental), (SELECT count(*) FROM payment)",
            db_name=pagila_db,
        ).rows[0]
        assert counts[0] == 1000
        assert counts[1] > 10_000
        assert counts[2] > 10_000

    def test_truncates_at_max_rows(self, pagila_db):
        result = execute_readonly("SELECT rental_id FROM rental", db_name=pagila_db, max_rows=100)
        assert len(result.rows) == 100
        assert result.truncated

    def test_timeout(self, pagila_db):
        with pytest.raises(QueryError) as exc:
            execute_readonly("SELECT pg_sleep(3)", db_name=pagila_db, timeout_ms=300)
        assert exc.value.kind == "timeout"

    def test_db_error_carries_postgres_message(self, pagila_db):
        with pytest.raises(QueryError) as exc:
            execute_readonly("SELECT no_such_column FROM film", db_name=pagila_db)
        assert exc.value.kind == "db_error"
        assert "no_such_column" in exc.value.message

    def test_trailing_semicolon_ok(self, pagila_db):
        assert execute_readonly("SELECT 1;", db_name=pagila_db).rows == [(1,)]


class TestBotReaderRole:
    """Layer 1: even if the safety checker were bypassed, the role must hold."""

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM actor",
            "UPDATE film SET rental_rate = 0",
            "INSERT INTO category (name) VALUES ('x')",
            "CREATE TABLE evil (id int)",
            "DROP TABLE film",
        ],
    )
    def test_cannot_modify_via_executor(self, pagila_db, sql):
        with pytest.raises(QueryError):
            execute_readonly(sql, db_name=pagila_db)

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM actor",
            "CREATE TEMP TABLE t (id int)",
            "CREATE TABLE public.evil (id int)",
        ],
    )
    def test_cannot_modify_even_outside_read_only_transaction(self, pagila_db, sql):
        # Raw connection, explicitly read-write: privileges alone must still block it.
        with psycopg.connect(conninfo("bot_reader", pagila_db), autocommit=True) as conn:
            conn.execute("SET default_transaction_read_only = off")
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(sql)

    def test_sensitive_columns_hidden(self, pagila_db):
        for column in ("password", "picture"):
            with pytest.raises(QueryError) as exc:
                execute_readonly(f"SELECT {column} FROM staff", db_name=pagila_db)
            assert exc.value.sqlstate == "42501"  # insufficient_privilege
        assert execute_readonly("SELECT username FROM staff", db_name=pagila_db).rows

    def test_partitions_not_directly_queryable(self, pagila_db):
        with pytest.raises(QueryError) as exc:
            execute_readonly("SELECT * FROM payment_p2022_01 LIMIT 1", db_name=pagila_db)
        assert exc.value.sqlstate == "42501"

    def test_dataset_functions_not_callable(self, pagila_db):
        with pytest.raises(QueryError) as exc:
            execute_readonly("SELECT get_customer_balance(1, now())", db_name=pagila_db)
        assert exc.value.sqlstate == "42501"

    @pytest.mark.parametrize("dbname", ["app", "postgres", "template1"])
    def test_cannot_connect_to_other_databases(self, dbname):
        with pytest.raises(psycopg.OperationalError, match="permission denied"):
            psycopg.connect(conninfo("bot_reader", dbname)).close()


class TestRagWriterRole:
    def test_cannot_connect_to_datasets(self, pagila_db):
        with pytest.raises(psycopg.OperationalError, match="permission denied"):
            psycopg.connect(conninfo("rag_writer", pagila_db)).close()

    def test_can_use_vectors_in_rag_schema(self):
        with psycopg.connect(conninfo("rag_writer", get_settings().app_db)) as conn:
            conn.execute("CREATE TABLE rag._probe (v vector(3))")
            conn.execute("INSERT INTO rag._probe VALUES ('[1,2,3]')")
            row = conn.execute("SELECT v <=> '[1,2,3]' FROM rag._probe").fetchone()
            assert row[0] == pytest.approx(0.0)
            conn.rollback()


class TestDataLoaderRole:
    def test_cannot_connect_to_app_database(self):
        with pytest.raises(psycopg.OperationalError, match="permission denied"):
            psycopg.connect(conninfo("data_loader", get_settings().app_db)).close()

    def test_is_not_superuser(self, pagila_db):
        with psycopg.connect(conninfo("data_loader", pagila_db)) as conn:
            assert conn.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user").fetchone() == (False,)
