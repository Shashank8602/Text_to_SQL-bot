# Text-to-SQL Agentic Bot

Ask questions about your own data in plain English. An LLM writes the SQL, and when the query fails or
returns a wrong-looking result, the agent feeds the failure back and retries.

Design: [docs/DESIGN.md](docs/DESIGN.md) · Status: **M2 (single-shot question → SQL → answer) complete**; the self-correcting agent loop is next (M3)

## Quickstart

Requires Docker and Python 3.11+.

```bash
cp .env.example .env              # then set the passwords and GROQ_API_KEY / GEMINI_API_KEY
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

## Ask a question

```bash
.venv/Scripts/python -m app.ask --dataset pagila "Which film category earned the most revenue?"
```

It prints the LLM's plan, the SQL, a preview of the result, the answer, and which model answered each
step (Groq, or Gemini when Groq is rate-limited). Free API keys:
[Groq](https://console.groq.com/keys) and [Gemini](https://aistudio.google.com/apikey).

## Tests

```bash
.venv/Scripts/python -m pytest          # integration tests need the database running
.venv/Scripts/python -m pytest -m llm   # live LLM test (uses API quota)
```

See [db/README.md](db/README.md) for the database roles and what each can access.
