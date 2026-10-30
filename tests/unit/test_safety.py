import pytest

from app.database.safety import check_sql

ALLOWED = [
    "SELECT count(*) FROM film",
    "SELECT title FROM film WHERE rating = 'PG-13' ORDER BY title LIMIT 5;",
    """SELECT c.name, sum(p.amount) AS revenue
       FROM payment p
       JOIN rental r ON r.rental_id = p.rental_id
       JOIN inventory i ON i.inventory_id = r.inventory_id
       JOIN film_category fc ON fc.film_id = i.film_id
       JOIN category c ON c.category_id = fc.category_id
       GROUP BY c.name ORDER BY revenue DESC""",
    """WITH monthly AS (
         SELECT date_trunc('month', payment_date) AS m, sum(amount) AS total
         FROM payment GROUP BY 1)
       SELECT m, total, lag(total) OVER (ORDER BY m) FROM monthly""",
    "SELECT first_name FROM actor UNION SELECT first_name FROM customer",
    "SELECT * FROM film WHERE title ILIKE '%drop table%'",  # keywords inside literals are data
    "SELECT 'a; DELETE FROM film' AS s",
    "SELECT g FROM generate_series(1, 12) AS g",
    "SELECT * FROM customer WHERE customer_id IN (SELECT customer_id FROM rental)",
    "SELECT * FROM public.film",
]

BLOCKED = [
    ("", "Empty"),
    ("   ", "Empty"),
    ("DROP TABLE film", "Only SELECT"),
    ("DELETE FROM payment", "Only SELECT"),
    ("INSERT INTO actor (first_name, last_name) VALUES ('a', 'b')", "Only SELECT"),
    ("UPDATE film SET rental_rate = 0", "Only SELECT"),
    ("TRUNCATE rental", "Only SELECT"),
    ("SELECT 1; DROP TABLE film", "Exactly one statement"),
    ("SELECT 1; SELECT 2", "Exactly one statement"),
    ("WITH d AS (DELETE FROM actor RETURNING *) SELECT * FROM d", "found DELETE"),
    ("SELECT * INTO film_copy FROM film", "SELECT ... INTO"),
    ("SELECT * FROM film FOR UPDATE", "row locking"),
    ("COPY film TO '/tmp/film.csv'", "Only SELECT"),
    ("EXPLAIN ANALYZE SELECT 1", "Only SELECT"),
    ("DO $$ BEGIN DELETE FROM film; END $$", "Only SELECT"),
    ("CALL some_proc()", "Only SELECT"),
    ("VACUUM", "Only SELECT"),
    ("SET ROLE postgres", "Only SELECT"),
    ("GRANT ALL ON film TO bot_reader", "Only SELECT"),
    ("BEGIN", "Only SELECT"),
    ("SELECT pg_sleep(60)", "pg_sleep"),
    ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
    ("SELECT lo_import('/etc/passwd')", "lo_import"),
    ("SELECT dblink('host=evil', 'select 1')", "dblink"),
    ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
    ("SELECT query_to_xml('DELETE FROM film RETURNING *', true, true, '')", "query_to_xml"),
    ("SELECT nextval('actor_actor_id_seq')", "nextval"),
    ("SELECT * FROM pg_catalog.pg_roles", "System table"),
    ("SELECT usename, passwd FROM pg_shadow", "System table"),
    ("SELECT * FROM pg_stat_activity", "System table"),
    ("SELECT * FROM information_schema.tables", "System table"),
    ("SELECT * FROM rag.schema_docs_local", "System table"),
    ("SELECT * FROM other_schema.secrets", "Schema 'other_schema'"),
    ("SELEC title FRM film", "could not be parsed"),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_read_only_queries(sql):
    result = check_sql(sql)
    assert result.ok, result.reason


@pytest.mark.parametrize(("sql", "reason_fragment"), BLOCKED)
def test_blocks_unsafe_queries(sql, reason_fragment):
    result = check_sql(sql)
    assert not result.ok, f"should have been blocked: {sql!r}"
    assert reason_fragment in result.reason


class TestTableAllowlist:
    allowed = {"film", "actor", "film_actor"}

    def test_known_tables_pass(self):
        sql = "SELECT a.first_name FROM actor a JOIN film_actor fa USING (actor_id)"
        assert check_sql(sql, self.allowed).ok

    def test_unknown_table_rejected_with_actionable_reason(self):
        result = check_sql("SELECT * FROM films", self.allowed)
        assert not result.ok
        assert "Unknown table 'films'" in result.reason

    def test_cte_names_are_not_treated_as_tables(self):
        sql = "WITH top_films AS (SELECT * FROM film LIMIT 5) SELECT * FROM top_films"
        assert check_sql(sql, self.allowed).ok

    def test_allowlist_is_case_insensitive(self):
        assert check_sql("SELECT * FROM FILM", self.allowed).ok


class TestAllowedSchemas:
    def test_dataset_schema_allowed_when_listed(self):
        assert check_sql("SELECT * FROM sales.orders", allowed_schemas={"public", "sales"}).ok

    def test_system_schemas_stay_blocked_even_if_listed(self):
        result = check_sql("SELECT * FROM pg_catalog.pg_roles", allowed_schemas={"pg_catalog"})
        assert not result.ok
