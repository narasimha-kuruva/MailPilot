"""In-memory `AuditService` implementation.

Every record is also emitted through the structured logger, so the audit
trail is visible in process logs even before a persistent store (e.g. a
database) backs this in a later phase.

Records are redacted (`mailpilot.redaction`) before they are stored, so a
credential that shows up in an instruction, a tool argument, or a tool
result is never kept or served back by `GET /agent/{id}/audit`.

Every record also feeds the optional `MetricsRegistry` (Phase 5.7): the
audit trail already sees every tool call and approval decision, so it is
the one place metrics need to be counted.
"""

from __future__ import annotations

from collections import defaultdict

from mailpilot.audit.service import AuditService
from mailpilot.logging_config import get_logger
from mailpilot.observability.metrics import MetricsRegistry
from mailpilot.redaction import redact_text, redact_value
from mailpilot.schemas.audit import AuditRecord

logger = get_logger(__name__)


class InMemoryAuditService(AuditService):
    def __init__(self, metrics: MetricsRegistry | None = None) -> None:
        self._records: dict[str, list[AuditRecord]] = defaultdict(list)
        self._metrics = metrics

    async def record(self, record: AuditRecord) -> None:
        # Only the content fields: the ids are what records are looked up by,
        # and redacting one ("ticket-api_key:7781") would hide the record.
        record = record.model_copy(
            update={
                "agent_request": redact_text(record.agent_request),
                "tool_args": redact_value(record.tool_args),
                "result_summary": None if record.result_summary is None else redact_text(record.result_summary),
            }
        )
        self._records[record.conversation_id].append(record)
        logger.info("agent_tool_call", extra={"extra_fields": record.model_dump(mode="json")})
        if self._metrics is not None:
            self._metrics.observe_audit(record)

    async def get_history(self, conversation_id: str) -> list[AuditRecord]:
        return list(self._records.get(conversation_id, []))
