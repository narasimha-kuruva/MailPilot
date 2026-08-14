"""Generic bounded-retry helper with exponential backoff.

Shared by the agent graph's tool execution (Phase 4) and the Gmail client
(Phase 5), so error classification and backoff behavior live in one place
instead of being re-implemented per caller. Never used for `send_email`
(see `mailpilot.safety.idempotency`) -- a network error on a send doesn't
tell you whether the email actually went out, so retrying it automatically
risks a duplicate send.
"""

from __future__ import annotations

import asyncio
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
}
_TRANSIENT_MESSAGE_TOKENS = ("timeout", "timed out", "rate limit", "temporarily unavailable", "connection reset")


class ErrorClass(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class TimeoutClassifiedError(Exception):
    """Raised when a call exceeds its timeout; always classified TRANSIENT."""


def classify_error(exc: BaseException) -> ErrorClass:
    """Best-effort heuristic: network/rate-limit/server errors are retried,
    validation/programming/client errors are not.

    Checked in order: an HTTP-style status code on the exception (e.g.
    `googleapiclient.errors.HttpError.resp.status`) is the most reliable
    signal when present; otherwise fall back to the exception's type name
    and message.
    """
    if isinstance(exc, TimeoutClassifiedError):
        return ErrorClass.TRANSIENT

    resp = getattr(exc, "resp", None)
    status = getattr(resp, "status", None) if resp is not None else None
    if status is not None:
        return ErrorClass.TRANSIENT if status in _TRANSIENT_STATUS_CODES else ErrorClass.PERMANENT

    if type(exc).__name__ in _TRANSIENT_EXCEPTION_NAMES:
        return ErrorClass.TRANSIENT

    message = str(exc).lower()
    if any(token in message for token in _TRANSIENT_MESSAGE_TOKENS):
        return ErrorClass.TRANSIENT

    return ErrorClass.PERMANENT


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    base_delay_seconds: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call `fn`, retrying on transient errors with exponential backoff.

    Permanent errors re-raise immediately. Transient errors re-raise once
    `max_retries` attempts are exhausted. Tests should pass a no-op `sleep`
    to avoid slowing down the suite.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - re-raised once non-retryable; never swallowed
            if classify_error(exc) is ErrorClass.PERMANENT or attempt >= max_retries:
                raise
            await sleep(base_delay_seconds * (2**attempt))
            attempt += 1


async def with_timeout(fn: Callable[[], Awaitable[T]], timeout_seconds: float) -> T:
    """Run `fn`, converting a timeout into a `TimeoutClassifiedError` so
    `classify_error` can retry it like any other transient failure."""
    try:
        return await asyncio.wait_for(fn(), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise TimeoutClassifiedError(f"Timed out after {timeout_seconds}s") from exc
