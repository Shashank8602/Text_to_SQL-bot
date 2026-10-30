# Database server

`init/01_setup.sh` runs once, the first time the Postgres volume is created. The server starts with
**no business data**. Datasets are loaded by the app (see the main README).

## Databases

| Database | Contents |
|---|---|
| `app` | `rag` schema: dataset registry (`rag.datasets`), later the schema-retrieval vectors |
| `ds_<slug>` | One per loaded dataset, e.g. `ds_pagila` |

`template1` has `vector` and `pg_trgm` pre-installed, so every dataset database gets them. `vector` is
an untrusted extension, so `data_loader` couldn't create it itself.

## Roles

| Role | Can | Cannot |
|---|---|---|
| `bot_reader` | Per dataset: `CONNECT` + `SELECT` on exposed tables/columns | Write; connect to `app`/`postgres`; see hidden columns; call dataset functions |
| `rag_writer` | Own and write the `rag` schema in `app` | Connect to dataset databases |
| `data_loader` | `CREATEDB`; create, fill and drop `ds_*` databases | Connect to `app`; superuser actions |
| `postgres` | Setup only | Never used by the app |

`bot_reader` defaults to read-only transactions and a 5 s statement timeout. `data_loader` has a 120 s
statement timeout, and the loader adds a wall-clock limit for the whole restore.

## Reset

Init scripts only run on an empty volume. This deletes all loaded datasets:

```bash
docker compose down -v && docker compose up -d --wait
```
