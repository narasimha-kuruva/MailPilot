"""Tests for `GoogleGmailClient` with a stubbed `googleapiclient` service.

No network, no OAuth: `_get_service` is replaced with a stub that records
how each request was executed, so we can assert the one property that
matters for correctness under concurrency -- each request gets its own
HTTP connection -- without talking to Gmail.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from mailpilot.config import Settings
from mailpilot.gmail.google_client import GoogleGmailClient


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8")


def _raw_message(message_id: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "threadId": "thread-1",
        "snippet": "",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": f"Subject {message_id}"},
                {"name": "From", "value": "alice@example.com"},
            ],
            "body": {"data": _b64("hello")},
        },
    }


class _Request:
    def __init__(self, result: Any, log: list[Any]) -> None:
        self._result = result
        self._log = log

    def execute(self, http: Any = None) -> Any:
        self._log.append(http)
        return self._result


class _Messages:
    def __init__(self, ids: list[str], log: list[Any]) -> None:
        self._ids = ids
        self._log = log

    def list(self, **_: Any) -> _Request:
        return _Request({"messages": [{"id": i} for i in self._ids]}, self._log)

    def get(self, *, id: str, **_: Any) -> _Request:  # noqa: A002 - mirrors the googleapiclient signature
        return _Request(_raw_message(id), self._log)


class _Users:
    def __init__(self, ids: list[str], log: list[Any]) -> None:
        self._messages = _Messages(ids, log)

    def messages(self) -> _Messages:
        return self._messages


class _Service:
    def __init__(self, ids: list[str]) -> None:
        self.http_log: list[Any] = []
        self._users = _Users(ids, self.http_log)

    def users(self) -> _Users:
        return self._users


def _client(service: _Service) -> GoogleGmailClient:
    client = GoogleGmailClient(Settings(gmail_max_retries=0))
    client._get_service = lambda: service  # type: ignore[method-assign]
    client._new_http = lambda: object()  # type: ignore[method-assign]  # fresh sentinel per call
    return client


@pytest.mark.asyncio
async def test_search_messages_uses_a_distinct_http_connection_per_request() -> None:
    """The shared httplib2 connection is not thread-safe; concurrent fetches
    sharing it corrupt TLS records and drop messages. Each request must get
    its own connection."""
    service = _Service(["m1", "m2", "m3"])

    messages = await _client(service).search_messages("in:inbox", max_results=3)

    assert [m.message_id for m in messages] == ["m1", "m2", "m3"]
    assert len(service.http_log) == 4  # 1 list + 3 gets
    assert all(http is not None for http in service.http_log)
    assert len({id(http) for http in service.http_log}) == 4


@pytest.mark.asyncio
async def test_search_messages_returns_empty_when_nothing_matches() -> None:
    service = _Service([])

    assert await _client(service).search_messages("is:unread") == []
    assert len(service.http_log) == 1
