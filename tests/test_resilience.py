from __future__ import annotations

import pytest

from mailpilot.resilience import (
    ErrorClass,
    TimeoutClassifiedError,
    classify_error,
    with_retries,
    with_timeout,
)


class _FakeHttpError(Exception):
    class _Resp:
        def __init__(self, status: int) -> None:
            self.status = status

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.resp = self._Resp(status)


def test_classify_error_uses_http_status_when_present() -> None:
    assert classify_error(_FakeHttpError(503)) is ErrorClass.TRANSIENT
    assert classify_error(_FakeHttpError(429)) is ErrorClass.TRANSIENT
    assert classify_error(_FakeHttpError(404)) is ErrorClass.PERMANENT
    assert classify_error(_FakeHttpError(400)) is ErrorClass.PERMANENT


def test_classify_error_falls_back_to_message_heuristics() -> None:
    assert classify_error(TimeoutError("request timed out")) is ErrorClass.TRANSIENT
    assert classify_error(ValueError("invalid argument: message_id required")) is ErrorClass.PERMANENT


def test_classify_error_treats_timeout_wrapper_as_transient() -> None:
    assert classify_error(TimeoutClassifiedError("slow")) is ErrorClass.TRANSIENT


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_with_retries_retries_transient_and_eventually_succeeds() -> None:
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _FakeHttpError(503)
        return "ok"

    result = await with_retries(flaky, max_retries=5, sleep=_no_sleep)

    assert result == "ok"
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_with_retries_does_not_retry_permanent_errors() -> None:
    attempts = {"n": 0}

    async def always_bad() -> str:
        attempts["n"] += 1
        raise _FakeHttpError(404)

    with pytest.raises(_FakeHttpError):
        await with_retries(always_bad, max_retries=5, sleep=_no_sleep)

    assert attempts["n"] == 1  # no retries on a permanent error


@pytest.mark.asyncio
async def test_with_retries_gives_up_after_max_retries() -> None:
    attempts = {"n": 0}

    async def always_transient() -> str:
        attempts["n"] += 1
        raise _FakeHttpError(503)

    with pytest.raises(_FakeHttpError):
        await with_retries(always_transient, max_retries=2, sleep=_no_sleep)

    assert attempts["n"] == 3  # initial attempt + 2 retries


@pytest.mark.asyncio
async def test_with_timeout_raises_timeout_classified_error() -> None:
    import asyncio

    async def slow() -> None:
        await asyncio.sleep(10)

    with pytest.raises(TimeoutClassifiedError):
        await with_timeout(slow, timeout_seconds=0.01)
