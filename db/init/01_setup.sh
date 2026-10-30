#!/usr/bin/env bash
# Server setup. Runs once, on first container start (docker-entrypoint-initdb.d).
#
# The server starts with NO business data. Datasets are loaded by the app, each into its
# own database (ds_<slug>), via app/datasets/loader.py.
#
# Databases:
#   app        - app metadata: rag schema (dataset registry, schema-retrieval vectors)
#   ds_<slug>  - one per uploaded dataset, created and owned by data_loader
#
# Roles:
#   bot_reader   - the ONLY role that runs LLM-generated SQL. Read-only, short timeout.
#                  Gets CONNECT + SELECT per dataset, only on columns the user exposed.
#   rag_writer   - owns the rag schema in the app database. No access to dataset data.
#   data_loader  - creates and fills ds_* databases. Not a superuser; no access to app.
set -euo pipefail

: "${BOT_READER_PASSWORD:?BOT_READER_PASSWORD must be set}"
: "${RAG_WRITER_PASSWORD:?RAG_WRITER_PASSWORD must be set}"
: "${DATA_LOADER_PASSWORD:?DATA_LOADER_PASSWORD must be set}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
  -v bot_pw="$BOT_READER_PASSWORD" -v rag_pw="$RAG_WRITER_PASSWORD" \
  -v loader_pw="$DATA_LOADER_PASSWORD" -v app_db="$POSTGRES_DB" <<'SQL'

------------------------------------------------------------------------------
-- Roles
------------------------------------------------------------------------------
CREATE ROLE bot_reader LOGIN PASSWORD :'bot_pw' CONNECTION LIMIT 20;
ALTER ROLE bot_reader SET default_transaction_read_only = on;
ALTER ROLE bot_reader SET statement_timeout = '5s';
ALTER ROLE bot_reader SET idle_in_transaction_session_timeout = '30s';

CREATE ROLE rag_writer LOGIN PASSWORD :'rag_pw' CONNECTION LIMIT 10;
ALTER ROLE rag_writer SET search_path = rag, public;

CREATE ROLE data_loader LOGIN CREATEDB PASSWORD :'loader_pw' CONNECTION LIMIT 5;
ALTER ROLE data_loader SET statement_timeout = '120s';

------------------------------------------------------------------------------
-- Templates: every new database (incl. datasets) gets these extensions.
-- vector is not a "trusted" extension, so data_loader could not create it itself.
------------------------------------------------------------------------------
\connect template1
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
REVOKE ALL ON DATABASE template1 FROM PUBLIC;

\connect postgres
-- Maintenance DB: only data_loader needs it (to run CREATE/DROP DATABASE).
REVOKE ALL ON DATABASE postgres FROM PUBLIC;
GRANT CONNECT ON DATABASE postgres TO data_loader;

------------------------------------------------------------------------------
-- app database (created by the entrypoint before this script, so before the
-- template1 extensions existed)
------------------------------------------------------------------------------
\connect :app_db
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
REVOKE ALL ON DATABASE :"app_db" FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE :"app_db" TO rag_writer;

CREATE SCHEMA rag AUTHORIZATION rag_writer;

SET ROLE rag_writer;
CREATE TABLE rag.datasets (
    slug           text PRIMARY KEY CHECK (slug ~ '^[a-z][a-z0-9_]{0,39}$'),
    name           text NOT NULL,
    db_name        text NOT NULL UNIQUE,
    source_type    text NOT NULL CHECK (source_type IN ('pg_dump', 'csv', 'excel')),
    source_file    text,
    status         text NOT NULL CHECK (status IN ('loading', 'ready')),
    hidden_columns jsonb NOT NULL DEFAULT '[]',
    created_at     timestamptz NOT NULL DEFAULT now()
);
RESET ROLE;
SQL
