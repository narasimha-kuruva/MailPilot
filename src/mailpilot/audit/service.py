"""Audit service interface.

Persists one `AuditRecord` per tool invocation (request, tool, args,
result, approval status, timestamp) so a run's full execution history can
be reconstructed and reviewed.

`record()` is the same for every backend and does the parts that must not
be forgotten: the record is redacted (`mailpilot.redaction`) before it is
stored -- an audit trail outlives the request and is read by people other
than the one who made it -- then written to the structured log and counted
in the optional `MetricsRegistry` (Phase 5.7). Backends only implement
`_append()` and `get_history()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mailpilot.logging_config import get_logger
from mailpilot.observability.metrics import MetricsRegistry
from mailpilot.redaction import redact_text, redact_value
from mailpilot.schemas.audit import AuditRecord

logger = get_logger(__name__)


def redact_record(record: AuditRecord) -> AuditRecord:
    # Only the content fields: the ids are what records are looked up by,
    # and redacting one ("ticket-api_key:7781") would hide the record.
    return record.model_copy(
        update={
            "agent_request": redact_text(record.agent_request),
            "tool_args": redact_value(record.tool_args),
            "result_summary": None if record.result_summary is None else redact_text(record.result_summary),
        }
    )


class AuditService(ABC):
    """Interface for recording and retrieving agent execution history."""

    def __init__(self, metrics: MetricsRegistry | None = None) -> None:
        self._metrics = metrics

    async def record(self, record: AuditRecord) -> None:
        record = redact_record(record)
        await self._append(record)
        logger.info("agent_tool_call", extra={"extra_fields": record.model_dump(mode="json")})
        if self._metrics is not None:
            self._metrics.observe_audit(record)

    @abstractmethod
    async def _append(self, record: AuditRecord) -> None:
        """Store an already-redacted record."""
        raise NotImplementedError

    @abstractmethod
    async def get_history(self, conversation_id: str) -> list[AuditRecord]:
        raise NotImplementedError
