from __future__ import annotations

import pytest

from mailpilot.resilience import (
    ErrorClass,
    TimeoutClassifiedError,
    classify_error,
    describe_error,
    is_rate_limited,
    retry_after_seconds,
    status_code,
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


# --- real-world error shapes seen in live testing -------------------------------


class _ServerError(Exception):
    """Like `langchain_google_genai.GoogleAPIError`: the HTTP code sits on the exception itself."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class _WrappedClientError(Exception):
    """Like `GoogleRateLimitError`: no code of its own, raised *from* a google.genai error that has one."""


def _wrapped(code: int, message: str) -> _WrappedClientError:
    outer = _WrappedClientError(f"Error calling model 'x' (RESOURCE_EXHAUSTED): {message}")
    outer.__cause__ = _ServerError(code, message)
    return outer


class _OllamaStyleError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("server error")
        self.status_code = status_code


class _RetryableModelError(Exception):
    is_retryable = True  # langchain_core.exceptions.ModelError convention


class _FinalModelError(Exception):
    is_retryable = False


_DAILY_QUOTA = (
    "429 RESOURCE_EXHAUSTED. {'error': {'message': 'You exceeded your current quota. "
    "Please retry in 11h27m32.845279834s.', 'details': [{'retryDelay': '41252s'}]}}"
)
_PER_MINUTE = "429 RESOURCE_EXHAUSTED. {'error': {'message': 'Rate limited. Please retry in 7s.'}}"


def test_status_code_is_found_on_the_exception_or_its_cause() -> None:
    assert status_code(_ServerError(503, "503 UNAVAILABLE")) == 503
    assert status_code(_wrapped(429, _PER_MINUTE)) == 429
    assert status_code(_OllamaStyleError(500)) == 500
    assert status_code(_OllamaStyleError(-1)) is None  # ollama's "no HTTP status" sentinel
    assert status_code(ValueError("nope")) is None


def test_gemini_high_demand_503_is_transient() -> None:
    assert classify_error(_ServerError(503, "503 UNAVAILABLE. high demand")) is ErrorClass.TRANSIENT
    # even without a code, the wording alone is enough
    assert classify_error(Exception("This model is currently experiencing high demand")) is ErrorClass.TRANSIENT


def test_daily_quota_429_is_permanent_but_short_rate_limit_is_transient() -> None:
    assert retry_after_seconds(_wrapped(429, _DAILY_QUOTA)) == 11 * 3600 + 27 * 60 + 32.845279834
    assert classify_error(_wrapped(429, _DAILY_QUOTA)) is ErrorClass.PERMANENT

    assert retry_after_seconds(_wrapped(429, _PER_MINUTE)) == 7.0
    assert classify_error(_wrapped(429, _PER_MINUTE)) is ErrorClass.TRANSIENT


def test_retry_delay_detail_is_parsed_when_message_has_no_retry_in() -> None:
    assert retry_after_seconds(Exception("details: [{'retryDelay': '12s'}]")) == 12.0
    assert retry_after_seconds(Exception("nothing useful")) is None


def test_langchain_is_retryable_flag_is_honoured() -> None:
    assert classify_error(_RetryableModelError("x")) is ErrorClass.TRANSIENT
    assert classify_error(_FinalModelError("x")) is ErrorClass.PERMANENT


def test_model_not_found_404_on_cause_is_permanent() -> None:
    assert classify_error(_wrapped(404, "models/gemini-1.5-pro is not found")) is ErrorClass.PERMANENT


@pytest.mark.asyncio
async def test_with_retries_waits_at_least_the_server_suggested_delay() -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    attempts = {"n": 0}

    async def rate_limited_once() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _wrapped(429, _PER_MINUTE)
        return "ok"

    assert await with_retries(rate_limited_once, max_retries=2, base_delay_seconds=0.5, sleep=record_sleep) == "ok"
    assert sleeps == [7.0]  # the 7s "retry in" beats the 0.5s backoff


@pytest.mark.asyncio
async def test_with_retries_never_sleeps_past_the_deadline() -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def always_503() -> str:
        attempts["n"] += 1
        raise _ServerError(503, "503 UNAVAILABLE")

    with pytest.raises(_ServerError):
        await with_retries(
            always_503, max_retries=5, base_delay_seconds=10.0, sleep=record_sleep, deadline=1005.0, clock=lambda: 1000.0
        )

    assert attempts["n"] == 1  # a 10s backoff would overshoot the 5s left in the budget
    assert sleeps == []


def test_describe_error_is_short_and_single_line() -> None:
    text = describe_error(_ServerError(503, "503 UNAVAILABLE.\n" + "x" * 1000))

    assert text.startswith("_ServerError: 503 UNAVAILABLE. x")
    assert "\n" not in text
    assert len(text) <= 303


# --- Gmail's rate limiting arrives as a 403, not a 429 -----------------------------

_GMAIL_QUOTA_403 = (
    "<HttpError 403 when requesting .../messages/abc?format=full returned \"Quota exceeded for quota metric "
    "'Total Query Cost' and limit 'Units per minute per user' of service 'gmail.googleapis.com'\". "
    "Details: \"[{'reason': 'rateLimitExceeded'}]\">"
)
_GMAIL_FORBIDDEN_403 = "<HttpError 403 ... returned \"Insufficient Permission\". Details: \"[{'reason': 'insufficientPermissions'}]\">"


def test_gmail_quota_403_is_transient_but_a_permissions_403_is_permanent() -> None:
    assert is_rate_limited(_FakeHttpError(403)) is False  # bare 403: no rate-limit wording
    quota = _FakeHttpError(403)
    quota.args = (_GMAIL_QUOTA_403,)
    forbidden = _FakeHttpError(403)
    forbidden.args = (_GMAIL_FORBIDDEN_403,)

    assert is_rate_limited(quota) is True
    assert classify_error(quota) is ErrorClass.TRANSIENT
    assert is_rate_limited(forbidden) is False
    assert classify_error(forbidden) is ErrorClass.PERMANENT


def test_retry_after_iso_timestamp_is_converted_to_seconds_from_now() -> None:
    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    past = "2020-01-01T00:00:00.000Z"

    assert 28.0 <= retry_after_seconds(Exception(f"User-rate limit exceeded. Retry after {future}")) <= 30.0
    assert retry_after_seconds(Exception(f"User-rate limit exceeded. Retry after {past}")) == 0.0
