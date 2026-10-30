"""Integration tests need the docker compose database: `docker compose up -d`."""

from pathlib import Path

import psycopg
import pytest

from app.database.connection import app_conninfo
from app.datasets import registry
from app.datasets.loader import delete_dataset, load_pg_dump

PAGILA_DUMP = Path(__file__).parents[2] / "sample_data" / "pagila" / "pagila.sql.gz"
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
    """Pagila loaded through the real upload path; returns its database name."""
    drop_if_exists(TEST_SLUG)
    report = load_pg_dump(PAGILA_DUMP, "Pagila (pytest)", slug=TEST_SLUG)
    yield report.db_name
    delete_dataset(TEST_SLUG)
