# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An agentic **Text-to-SQL bot**. A user asks a question in plain English about their own data. An LLM
writes PostgreSQL. The agent checks the SQL, runs it read-only, validates the result, and on failure
feeds the error or critique back to the LLM and retries, up to 3 attempts. A successful result is turned
into a written answer. Portfolio/learning project: **free-tier resources only** (Groq + Gemini LLMs,
local Docker Postgres).

- Full design, decisions and milestone plan: [docs/DESIGN.md](docs/DESIGN.md). Read it before changing
  architecture, and keep it updated when behavior changes.
- The bot ships with **no data**. Users load datasets; `sample_data/pagila/` is a sample dataset and the
  eval reference, loaded like any upload.

## Architecture at a glance

| Path | Purpose |
|---|---|
| `db/init/01_setup.sh` | Server setup: roles, `app` database, `rag` schema, template extensions. No business data. |
| `app/datasets/` | Load a plain-SQL Postgres dump as an isolated dataset database `ds_<slug>` (own parser, never `psql`), registry, CLI |
| `app/database/` | `safety.py` (sqlglot SELECT-only checker), `executor.py` (read-only execution as `bot_reader`), `catalog.py` (schema DDL as the bot sees it), `connection.py` |
| `app/llm/` | `client.py`: the single `call(group, messages, schema)` entry point (LiteLLM Router: Groq primary, Gemini fallback, rpm/tpm limits). `prompts.py`, `result_format.py` |
| `app/agent/` | LangGraph agent (`graph.py`), state, deterministic validation (`validation.py`) |
| `app/ask.py` | CLI: ask a question, prints the attempt trace |
| `eval/` | 40 Pagila questions with gold SQL, scoring, runner, reports |
| `tests/` | `unit/` (no DB), `integration/` (needs Docker DB; loads Pagila as `pytest_pagila`) |

### Security model (do not weaken)
- Database roles:
  - `bot_reader` is the only role that runs LLM-written SQL. It is read-only, has a 5 s timeout, and
    gets `SELECT` only on the columns exposed per dataset.
  - `rag_writer` owns metadata in the `app` database.
  - `data_loader` creates the dataset databases. It is not a superuser.
  - The app never uses `postgres`.
- Defense in depth for generated SQL: sqlglot safety check → read-only role → read-only transaction →
  server-side cursor with a row cap.
- User text and LLM-derived text always go inside delimited blocks (`<question>`, `<reviewer_note>`),
  never into instructions.
- Sensitive columns (password, token, api_key, bytea/vector/tsvector, …) are hidden at load time.

## Commands

Windows, Git Bash or PowerShell. Python 3.11, venv in `.venv/`.

```bash
docker compose up -d --wait                                   # Postgres 18 + pgvector on 127.0.0.1:5433
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m app.datasets.cli load sample_data/pagila/pagila.sql.gz --name Pagila
.venv/Scripts/python -m app.datasets.cli list
.venv/Scripts/python -m app.ask --dataset pagila "Which film category earned the most revenue?"
.venv/Scripts/python -m pytest                                # offline + integration (no LLM quota)
.venv/Scripts/python -m pytest -m llm                         # live LLM test (uses quota)
.venv/Scripts/python -m eval.run_eval --delay 3               # full eval, ~15 min on free tiers
```

Resetting the database (`docker compose down -v`) deletes every loaded dataset.

## Conventions

- **Secrets**: `.env` holds passwords and API keys and is git-ignored. Never commit it, print keys, or
  paste them anywhere. `.env.example` lists every setting with placeholders.
- **Models change**: free-tier model names and limits drift. Verify against the providers' `/models`
  endpoints before changing defaults in `app/config.py` / `.env.example`.
- **Dependencies**: exact pins in `requirements.txt`. Prefer releases that have been public for at least
  a week, and check PyPI for yanked versions.
- **Tests**: run the full suite before every commit. Agent and pipeline tests use the scripted fake LLM
  (`tests/integration/test_agent.py`) so they cost no quota. Live LLM tests must carry `@pytest.mark.llm`.
- **Eval**: run it after changes to prompts, validation or retrieval. Report results honestly in
  DESIGN.md §10, including caveats; don't tune on the eval set without saying so.
- **Windows**: CLIs call `sys.stdout.reconfigure(encoding="utf-8")`. Use `127.0.0.1`, not `localhost`
  (IPv6 hangs). `*.sh`/`*.sql` stay LF (`.gitattributes`).
- **Docs**: update DESIGN.md (status, decisions, results) and README.md with each milestone.

## Git & GitHub workflow (rules from the project owner)

Repository: https://github.com/Shashank8602/Text_to_SQL-bot

1. **Commit and push after each milestone.**
2. **Use branching when needed**: one branch per milestone (e.g. `m4-retrieval`) with a pull request
   into `main`.
3. **Merge the branches by accepting the PRs.** Claude merges its own milestone PRs once tests pass;
   no separate review is required.
4. **Claude owns all Git and GitHub operations**: branches, commits, pushes, PRs, merges, keeping local
   `main` in sync, and cleaning up merged branches. The owner should not need to run git.
5. **Wait for the owner's command before starting the next milestone or phase.** After finishing one,
   report the results and stop.

Commit messages: `M<n>: <summary>`, a body explaining what and why, and the attribution trailer. Never
commit `.env`, `.venv/`, or generated eval artifacts other than the milestone's baseline report.

## Milestone status

See the table in DESIGN.md §13. Done: M1 (DB foundation + dump loading), M2 (single-shot pipeline),
M3 (self-correcting agent + eval: 36/40 single-shot → 40/40). Next: M4 (schema retrieval, embedding
comparison, held-out questions), but only on the owner's command.
