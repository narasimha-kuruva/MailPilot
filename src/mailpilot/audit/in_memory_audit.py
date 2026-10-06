"""In-memory `AuditService` implementation.

Every record is also emitted through the structured logger, so the audit
trail is visible in process logs even before a persistent store (e.g. a
database) backs this in a later phase.

Records are redacted (`mailpilot.redaction`) before they are stored, so a
credential that shows up in an instruction, a tool argument, or a tool
result is never kept or served back by `GET /agent/{id}/audit`.
"""

from __future__ import annotations

from collections import defaultdict

from mailpilot.audit.service import AuditService
from mailpilot.logging_config import get_logger
from mailpilot.redaction import redact_value
from mailpilot.schemas.audit import AuditRecord

logger = get_logger(__name__)


class InMemoryAuditService(AuditService):
    def __init__(self) -> None:
        self._records: dict[str, list[AuditRecord]] = defaultdict(list)

    async def record(self, record: AuditRecord) -> None:
        record = AuditRecord.model_validate(redact_value(record.model_dump()))
        self._records[record.conversation_id].append(record)
        logger.info("agent_tool_call", extra={"extra_fields": record.model_dump(mode="json")})

    async def get_history(self, conversation_id: str) -> list[AuditRecord]:
        return list(self._records.get(conversation_id, []))
