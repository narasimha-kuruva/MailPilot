from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from mailpilot.main import create_app

RUN_INTEGRATION_TESTS = os.environ.get("MAILPILOT_RUN_INTEGRATION_TESTS") == "1"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Integration tests (real Gmail, real model) only run when explicitly enabled."""
    if RUN_INTEGRATION_TESTS:
        return
    skip = pytest.mark.skip(reason="integration test: set MAILPILOT_RUN_INTEGRATION_TESTS=1 to run it")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
