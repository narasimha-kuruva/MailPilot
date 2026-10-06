"""Request IDs, request logging, and the last-resort error handler (Phase 5.10).

Every request gets an id: the caller's `X-Request-ID` if it is a safe
token, otherwise a fresh one. It is returned in the `X-Request-ID` response
header and stamped on every log line written while the request is served
(`mailpilot.logging_config.request_id_var`), audit log lines included, so
one id ties a client's report to the server's logs.

An exception no route handled becomes a 500 with a generic message and the
request id -- never a traceback, an exception message, or anything else
that might carry a secret or an email's contents. The full detail goes to
the server log, which is redacted (`mailpilot.redaction`). Expected errors
keep their own status codes: routes raise `HTTPException` (404 unknown
approval, 410 expired approval, 503 provider unavailable) and FastAPI
answers invalid input with 422.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from mailpilot.logging_config import get_logger, request_id_var

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

# The web UI shows text that came from emails. It never inserts it as HTML,
# and this policy is the second line: only the UI's own script and styles may
# run, it may only talk to this server, and no other site may frame it.
UI_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else uuid4().hex
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - logged in full here, answered generically below
            logger.exception(
                "Unhandled error while serving a request",
                extra={"extra_fields": {"method": request.method, "path": request.url.path}},
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error. Quote the request_id when reporting it.",
                    "request_id": request_id,
                },
            )
        logger.info(
            "http_request",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            },
        )
    finally:
        request_id_var.reset(token)
    response.headers[REQUEST_ID_HEADER] = request_id
    if request.url.path.startswith("/ui"):
        response.headers.update(UI_SECURITY_HEADERS)
    return response
