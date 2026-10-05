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


class _GmailHttpError(Exception):
    """Shaped like `googleapiclient.errors.HttpError`: `.resp.status` plus Gmail's wording."""

    class _Resp:
        def __init__(self, status: int) -> None:
            self.status = status

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.resp = self._Resp(status)


class _FailingMessages(_Messages):
    """`get` fails for the ids in `errors` with the given exception."""

    def __init__(self, ids: list[str], log: list[Any], errors: dict[str, Exception]) -> None:
        super().__init__(ids, log)
        self._errors = errors
        self.get_calls = 0

    def get(self, *, id: str, **_: Any) -> _Request:  # noqa: A002
        self.get_calls += 1
        if id in self._errors:
            raise self._errors[id]
        return super().get(id=id)


def _failing_service(ids: list[str], errors: dict[str, Exception]) -> _Service:
    service = _Service(ids)
    service._users._messages = _FailingMessages(ids, service.http_log, errors)
    return service


@pytest.mark.asyncio
async def test_search_drops_only_messages_deleted_between_list_and_fetch() -> None:
    service = _failing_service(["m1", "m2", "m3"], {"m2": _GmailHttpError(404, "<HttpError 404 ... Not Found>")})

    messages = await _client(service).search_messages("in:inbox", max_results=3)

    assert [m.message_id for m in messages] == ["m1", "m3"]


@pytest.mark.asyncio
async def test_search_refuses_to_return_a_partial_result_when_gmail_rate_limits() -> None:
    """Seen live: a 50-message search silently came back with 6. The agent
    would have presented that as the complete inbox."""
    from mailpilot.gmail.client import GmailSearchIncompleteError

    quota = _GmailHttpError(403, "<HttpError 403 ... \"Quota exceeded for quota metric 'Total Query Cost'\" rateLimitExceeded>")
    service = _failing_service(["m1", "m2", "m3"], {"m2": quota, "m3": quota})

    with pytest.raises(GmailSearchIncompleteError, match="matched 3 message\\(s\\) but 2 could not be fetched") as info:
        await _client(service).search_messages("in:inbox", max_results=3)

    # Each fetch already retried; re-running the whole search would only burn
    # more of a per-minute quota, so the tool layer must not retry it either.
    from mailpilot.resilience import ErrorClass, classify_error

    assert classify_error(info.value) is ErrorClass.PERMANENT


@pytest.mark.asyncio
async def test_search_stops_fetching_once_a_rate_limit_is_confirmed() -> None:
    """Seen live: 13 failing fetches each spent their own 7s of backoff -> a
    43s search. After the first confirmed rate limit the rest fail at once."""
    from mailpilot.config import Settings
    from mailpilot.gmail.client import GmailSearchIncompleteError

    quota = _GmailHttpError(403, "<HttpError 403 ... rateLimitExceeded>")
    ids = [f"m{i}" for i in range(12)]
    service = _failing_service(ids, {mid: quota for mid in ids[2:]})
    client = GoogleGmailClient(Settings(_env_file=None, gmail_max_retries=0, gmail_max_concurrent_fetches=1))
    client._get_service = lambda: service  # type: ignore[method-assign]
    client._new_http = lambda: object()  # type: ignore[method-assign]

    with pytest.raises(GmailSearchIncompleteError, match="10 could not be fetched"):
        await client.search_messages("in:inbox", max_results=12)

    messages = service._users._messages
    assert messages.get_calls == 3  # m0, m1 fetched; m2 hit the limit; m3..m11 never asked Gmail


@pytest.mark.asyncio
async def test_search_never_fetches_more_bodies_at_once_than_configured() -> None:
    import asyncio

    from mailpilot.config import Settings

    in_flight = {"now": 0, "peak": 0}

    class _SlowMessages(_Messages):
        def get(self, *, id: str, **_: Any) -> _Request:  # noqa: A002
            return super().get(id=id)

    service = _Service([f"m{i}" for i in range(12)])
    client = GoogleGmailClient(Settings(_env_file=None, gmail_max_retries=0, gmail_max_concurrent_fetches=3))
    client._get_service = lambda: service  # type: ignore[method-assign]
    client._new_http = lambda: object()  # type: ignore[method-assign]

    original_run = client._run

    async def counting_run(fn):  # type: ignore[no-untyped-def]
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        try:
            await asyncio.sleep(0.01)
            return await original_run(fn)
        finally:
            in_flight["now"] -= 1

    client._run = counting_run  # type: ignore[method-assign]

    messages = await client.search_messages("in:inbox", max_results=12)

    assert len(messages) == 12
    assert in_flight["peak"] <= 3 + 1  # +1: the initial list call is not throttled
