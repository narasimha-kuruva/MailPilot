"""Structured logging setup.

Uses the standard library `logging` module with a JSON-ish structured
formatter so log lines are easy to parse later (e.g. by the audit layer or
an external log aggregator), without pulling in an extra dependency for
Phase 1.

Every line is scrubbed by `mailpilot.redaction` before it is written
(Phase 5.1), so an API key or OAuth token that ends up in a message, an
exception, or `extra_fields` never reaches the log output.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from mailpilot.redaction import redact_text, redact_value


class StructuredFormatter(logging.Formatter):
    """Renders log records as single-line JSON objects, with secrets redacted."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)

        # Objects json can't encode are rendered with str() and scrubbed too.
        return json.dumps(redact_value(payload), default=lambda value: redact_text(str(value)))


def configure_logging(log_level: str = "INFO") -> None:
    """Configure the root logger once, at application startup."""
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    # Avoid duplicate handlers if called more than once (e.g. in tests).
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor mirroring `logging.getLogger`."""
    return logging.getLogger(name)
