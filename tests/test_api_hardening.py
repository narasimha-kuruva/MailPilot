"""Phase 5.10: request ids, request logging, sanitized 500s, input limits."""

from __future__ import annotations

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from mailpilot.api.deps import get_agent
from mailpilot.logging_config import StructuredFormatter
from mailpilot.main import create_app
from tests.conftest import local_client

FAKE_API_KEY = "AIza" + "Sy" + "q" * 33


@pytest.fixture
def log_lines():
    """Captures what the app logs, formatted exactly as in production.

    Request it after any fixture that builds an app: `create_app()` resets
    the root logger's handlers.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    root.removeHandler(handler)


@pytest.fixture
def failing_client() -> TestClient:
    app = create_app()

    @app.get("/api/v1/boom")
    async def boom() -> None:
        raise RuntimeError(f"Gemini rejected key {FAKE_API_KEY} for alice@example.com's thread")

    return local_client(app)


def test_every_response_carries_a_generated_request_id(client: TestClient) -> None:
    first = client.get("/api/v1/health").headers["X-Request-ID"]
    second = client.get("/api/v1/health").headers["X-Request-ID"]

    assert len(first) == 32 and first != second


def test_a_safe_client_request_id_is_kept_and_an_unsafe_one_replaced(client: TestClient) -> None:
    kept = client.get("/api/v1/health", headers={"X-Request-ID": "trace-42.a_b"})
    replaced = client.get("/api/v1/health", headers={"X-Request-ID": "bad id\nInjected: yes"})

    assert kept.headers["X-Request-ID"] == "trace-42.a_b"
    assert replaced.headers["X-Request-ID"] != "bad id\nInjected: yes"
    assert len(replaced.headers["X-Request-ID"]) == 32


def test_unhandled_errors_become_a_generic_500(failing_client: TestClient, log_lines) -> None:
    response = failing_client.get("/api/v1/boom")

    assert response.status_code == 500
    request_id = response.headers["X-Request-ID"]
    assert response.json() == {
        "detail": "Internal server error. Quote the request_id when reporting it.",
        "request_id": request_id,
    }
    # The detail is in the server log -- with the same request id, and redacted.
    [error] = [line for line in log_lines() if line["message"] == "Unhandled error while serving a request"]
    assert error["request_id"] == request_id
    assert "RuntimeError" in error["exception"]
    assert FAKE_API_KEY not in json.dumps(error)


def test_each_request_is_logged_once_with_its_id_status_and_duration(client: TestClient, log_lines) -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "req-1"})

    [line] = [line for line in log_lines() if line["message"] == "http_request"]
    assert response.status_code == 200
    assert line["request_id"] == "req-1"
    assert (line["method"], line["path"], line["status_code"]) == ("GET", "/api/v1/health", 200)
    assert line["duration_ms"] >= 0


def test_log_lines_outside_a_request_have_no_request_id(log_lines) -> None:
    logging.getLogger("mailpilot.test").warning("background work")

    [line] = log_lines()
    assert "request_id" not in line


@pytest.mark.parametrize(
    "body",
    [
        {"instruction": ""},
        {"instruction": "x" * 10_001},
        {"instruction": "check my inbox", "conversation_id": ""},
        {"conversation_id": "c1"},
    ],
)
def test_invalid_agent_requests_are_rejected_with_422(body: dict) -> None:
    app = create_app()
    # A stand-in: FastAPI resolves dependencies even for a request it then rejects.
    app.dependency_overrides[get_agent] = object
    client = local_client(app)

    response = client.post("/api/v1/agent/run", json=body)

    assert response.status_code == 422
