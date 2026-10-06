from __future__ import annotations

import os

# Before anything reads the settings: the app's own services stay in memory
# during tests, so the suite never touches the developer's ./data/state.
# (Environment variables win over .env.) Tests of the SQLite backend build it
# explicitly in a temporary directory.
os.environ["STATE_BACKEND"] = "memory"

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mailpilot.config import Settings, get_settings  # noqa: E402
from mailpilot.main import create_app  # noqa: E402

RUN_INTEGRATION_TESTS = os.environ.get("MAILPILOT_RUN_INTEGRATION_TESTS") == "1"

LOOPBACK = ("127.0.0.1", 50000)


def local_client(app: FastAPI, *, api_key: str | None = None, client: tuple[str, int] = LOOPBACK) -> TestClient:
    """A test client calling `app` from this machine.

    The app's settings are pinned (no `.env`, `MAILPILOT_API_KEY` = `api_key`),
    so whether a request is let in never depends on the developer's own
    configuration.
    """
    pinned = Settings(_env_file=None, mailpilot_api_key=api_key)
    app.dependency_overrides[get_settings] = lambda: pinned
    return TestClient(app, client=client)


@pytest.fixture
def client() -> TestClient:
    return local_client(create_app())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Integration tests (real Gmail, real model) only run when explicitly enabled."""
    if RUN_INTEGRATION_TESTS:
        return
    skip = pytest.mark.skip(reason="integration test: set MAILPILOT_RUN_INTEGRATION_TESTS=1 to run it")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
