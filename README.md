# Text-to-SQL Agentic Bot

Ask questions about your own data in plain English. An LLM writes the SQL, and when the query fails or
returns a wrong-looking result, the agent feeds the failure back and retries.

Design: [docs/DESIGN.md](docs/DESIGN.md) · Status: **M3 (self-correcting agent + evaluation) complete**; schema retrieval is next (M4)

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

It prints the agent trace (each attempt, and why a failed one was rejected), the final SQL, a preview of
the result, the answer, and which model answered each step (Groq, or Gemini when Groq is rate-limited).
A failed attempt is fed back to the LLM with the error or the reviewer's critique, up to 3 attempts. Free API keys:
[Groq](https://console.groq.com/keys) and [Gemini](https://aistudio.google.com/apikey).

## Evaluation

40 questions on the Pagila sample, with hand-verified gold SQL ([eval/questions.yaml](eval/questions.yaml)):

| | Accuracy |
|---|---|
| Single-shot (first attempt only) | 36/40 (90%) |
| **With self-correction (up to 3 attempts)** | **40/40 (100%)** |

All 4 first-attempt failures were recovered (e.g. "UK" → `United Kingdom` after the first query matched
Ukraine), and no correct first attempt was broken. See
[DESIGN.md §10](docs/DESIGN.md#10-evaluation-suite) for the caveats.

```bash
.venv/Scripts/python -m eval.run_eval                 # needs the Pagila sample loaded as 'pagila'
.venv/Scripts/python -m eval.run_eval --ids q08,q13 --delay 0
```

## Tests

```bash
.venv/Scripts/python -m pytest          # integration tests need the database running
.venv/Scripts/python -m pytest -m llm   # live LLM test (uses API quota)
```

See [db/README.md](db/README.md) for the database roles and what each can access.
