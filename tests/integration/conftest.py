"""Integration tests need the docker compose database: `docker compose up -d`."""

from pathlib import Path

import psycopg
import pytest

from app.database.connection import app_conninfo
from app.datasets import descriptions, registry
from app.datasets.loader import delete_dataset, load_pg_dump

PAGILA_DUMP = Path(__file__).parents[2] / "sample_data" / "pagila" / "pagila.sql.gz"
PAGILA_DESCRIPTIONS = PAGILA_DUMP.parent / "descriptions.yaml"
TEST_SLUG = "pytest_pagila"


def _db_available() -> bool:
    try:
        with psycopg.connect(app_conninfo()):
            return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _db_available():
        return
    skip = pytest.mark.skip(reason="database not running (docker compose up -d)")
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(skip)


def drop_if_exists(slug: str) -> None:
    if registry.get(slug):
        delete_dataset(slug)


@pytest.fixture(scope="session")
def pagila_db() -> str:
    """Pagila loaded through the real upload path, with descriptions; returns its database name."""
    drop_if_exists(TEST_SLUG)
    report = load_pg_dump(PAGILA_DUMP, "Pagila (pytest)", slug=TEST_SLUG)
    descriptions.save(TEST_SLUG, descriptions.parse_yaml(PAGILA_DESCRIPTIONS))
    yield report.db_name
    delete_dataset(TEST_SLUG)


@pytest.fixture(scope="session")
def pagila_index(pagila_db) -> str:
    """Local-embedder retrieval index for the test dataset (downloads the model on first use)."""
    from app.retrieval.indexer import build_index
    build_index(TEST_SLUG, "local")
    return TEST_SLUG
