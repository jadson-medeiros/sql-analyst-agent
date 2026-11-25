from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sql_analyst_agent.seed import generate


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    logging.getLogger("mcp").setLevel(logging.WARNING)
    return generate(tmp_path_factory.mktemp("db") / "store.db")
