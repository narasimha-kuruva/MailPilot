"""Generic bounded-retry helper with exponential backoff, and error classification.

Shared by the agent graph's tool execution (Phase 4), the Gmail client
(Phase 5), and -- since the first live runs -- the LLM calls themselves
(`agent_node`, `LangGraphAgent.resume()` / `plan()`), so error
classification and backoff behavior live in one place instead of being
re-implemented per caller. Never used for `send_email` (see
`mailpilot.safety.idempotency`) -- a network error on a send doesn't tell
you whether the email actually went out, so retrying it automatically
risks a duplicate send.

Classification, in order of reliability:

1. An HTTP status code found on the exception or its `__cause__` chain:
   `googleapiclient` `HttpError.resp.status`; `google.genai` `APIError.code`
   (LangChain's Gemini wrapper raises its own error types *from* that
   error, so the code lives on the cause); `ollama.ResponseError.status_code`;
   httpx responses. Rate limiting is special: Gemini says so with a 429,
   Gmail with a 403 whose reason is `rateLimitExceeded` / a per-minute
   "Quota exceeded" -- both are retried. Google says how long to wait, and
   a daily-quota 429 asks for hours; that run is not going to succeed, so
   it is treated as permanent rather than burning the time budget.
2. LangChain's `ModelError.is_retryable` flag.
3. The exception's type name (connection / timeout families).
4. Message heuristics ("timed out", "high demand", ...).
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from enum import StrEnum
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_TRANSIENT_EXCEPTION_NAMES = {
    "TimeoutError",
    "ConnectionError",
    "ConnectionResetError",
    "ServerNotFoundError",
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
    "InternalServerError",
    # httpx, used by both the Gemini and the Ollama clients
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
}
_TRANSIENT_MESSAGE_TOKENS = (
    "timeout",
    "timed out",
    "rate limit",
    "temporarily unavailable",
    "connection reset",
    "high demand",
    "try again later",
    "overloaded",
    "503 unavailable",
)

# A server-suggested wait longer than this (e.g. a daily quota that resets
# in eleven hours) means no retry within a single run can succeed.
MAX_RETRY_AFTER_SECONDS = 60.0

# Google phrases it three ways: "Please retry in 11h27m32.8s." in a Gemini
# message, a RetryInfo detail {'retryDelay': '41252s'}, and Gmail's
# "User-rate limit exceeded. Retry after 2026-10-05T16:45:00.000Z".
_RETRY_IN_PATTERN = re.compile(r"retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", re.IGNORECASE)
_RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)
_RETRY_AFTER_ISO_PATTERN = re.compile(r"retry after (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z", re.IGNORECASE)

# A 403 is normally final (permissions) -- unless it is Gmail's way of saying
# "slow down", which it does with these words.
_RATE_LIMIT_TOKENS = ("ratelimitexceeded", "rate limit", "quota exceeded", "resource_exhausted")


class ErrorClass(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class TimeoutClassifiedError(Exception):
    """Raised when a call exceeds its timeout; always classified TRANSIENT."""


def _cause_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__
    return chain


def status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status from an exception or anything it was raised from."""
    for err in _cause_chain(exc):
        resp = getattr(err, "resp", None)
        candidates = (
            getattr(resp, "status", None) if resp is not None else None,
            getattr(err, "status_code", None),
            getattr(err, "code", None),
            getattr(getattr(err, "response", None), "status_code", None),
        )
        for value in candidates:
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return value
    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """The server-suggested wait before retrying, if the error states one."""
    text = " ".join(str(err) for err in _cause_chain(exc))
    match = _RETRY_IN_PATTERN.search(text)
    if match and any(match.groups()):
        hours, minutes, seconds = match.groups()
        return float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)
    match = _RETRY_DELAY_PATTERN.search(text)
    if match:
        return float(match.group(1))
    match = _RETRY_AFTER_ISO_PATTERN.search(text)
    if match:
        resume_at = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
        return max(0.0, (resume_at - datetime.now(timezone.utc)).total_seconds())
    return None


def is_rate_limited(exc: BaseException) -> bool:
    """A 429, or a 403 that Gmail uses for rate limiting rather than permissions."""
    status = status_code(exc)
    if status == 429:
        return True
    if status == 403:
        text = " ".join(str(err) for err in _cause_chain(exc)).lower()
        return any(token in text for token in _RATE_LIMIT_TOKENS)
    return False


def describe_error(exc: BaseException, max_chars: int = 300) -> str:
    """Short, user-presentable summary: type plus the start of the message."""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def classify_error(exc: BaseException) -> ErrorClass:
    """Transient (worth retrying) vs permanent (fail now) -- see the module docstring."""
    if isinstance(exc, TimeoutClassifiedError):
        return ErrorClass.TRANSIENT

    status = status_code(exc)
    if is_rate_limited(exc):
        wait = retry_after_seconds(exc)
        if wait is not None and wait > MAX_RETRY_AFTER_SECONDS:
            return ErrorClass.PERMANENT  # quota exhausted for hours, not a blip
        return ErrorClass.TRANSIENT
    if status is not None:
        return ErrorClass.TRANSIENT if status in _TRANSIENT_STATUS_CODES else ErrorClass.PERMANENT

    chain = _cause_chain(exc)
    for err in chain:
        flag = getattr(err, "is_retryable", None)  # langchain_core.exceptions.ModelError
        if isinstance(flag, bool):
            return ErrorClass.TRANSIENT if flag else ErrorClass.PERMANENT

    if any(type(err).__name__ in _TRANSIENT_EXCEPTION_NAMES for err in chain):
        return ErrorClass.TRANSIENT

    message = " ".join(str(err) for err in chain).lower()
    if any(token in message for token in _TRANSIENT_MESSAGE_TOKENS):
        return ErrorClass.TRANSIENT

    return ErrorClass.PERMANENT


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    base_delay_seconds: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Call `fn`, retrying on transient errors with exponential backoff.

    Permanent errors re-raise immediately. Transient errors re-raise once
    `max_retries` attempts are exhausted. A server-suggested wait (a 429's
    "retry in Ns") is honoured when it is longer than the backoff. If a
    `deadline` (a `time.monotonic()` value) is given, no sleep may run past
    it: the caller's wall-clock budget wins over another attempt. Tests
    should pass a no-op `sleep` to avoid slowing down the suite.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - re-raised once non-retryable; never swallowed
            if classify_error(exc) is ErrorClass.PERMANENT or attempt >= max_retries:
                raise
            delay = base_delay_seconds * (2**attempt)
            suggested = retry_after_seconds(exc)
            if suggested is not None:
                delay = max(delay, min(suggested, MAX_RETRY_AFTER_SECONDS))
            if deadline is not None and clock() + delay > deadline:
                raise
            await sleep(delay)
            attempt += 1


async def with_timeout(fn: Callable[[], Awaitable[T]], timeout_seconds: float) -> T:
    """Run `fn`, converting a timeout into a `TimeoutClassifiedError` so
    `classify_error` can retry it like any other transient failure."""
    try:
        return await asyncio.wait_for(fn(), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise TimeoutClassifiedError(f"Timed out after {timeout_seconds}s") from exc
