# Text-to-SQL Agentic Bot

Ask questions about your own data in plain English. An LLM writes the SQL, and when the query fails or
returns a wrong-looking result, the agent feeds the failure back and retries.

Design: [docs/DESIGN.md](docs/DESIGN.md) · Status: **M1 (database foundation + dump loading) complete**

## Quickstart

Requires Docker and Python 3.11+.

```bash
cp .env.example .env              # then replace the change-me passwords
docker compose up -d --wait       # empty Postgres 18 + pgvector server
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt    # Windows (Linux/macOS: .venv/bin/pip)
```

## Load data

The bot has no built-in data. Load a plain-SQL PostgreSQL dump (`.sql` or `.sql.gz`), for example the
included sample:

```bash
.venv/Scripts/python -m app.datasets.cli load sample_data/pagila/pagila.sql.gz --name Pagila
.venv/Scripts/python -m app.datasets.cli list
.venv/Scripts/python -m app.datasets.cli delete pagila
```

To load your own database, export it with
`pg_dump --format=plain --no-owner --no-privileges -f mydb.sql mydb`. Sensitive-looking columns
(`password`, `token`, `api_key`, …) are hidden from the bot automatically. Use `--hide table.column` or
`--expose table.column` to adjust. CSV and Excel uploads come in M7.

## Tests

```bash
.venv/Scripts/python -m pytest      # integration tests need the database running
```

See [db/README.md](db/README.md) for the database roles and what each can access.
