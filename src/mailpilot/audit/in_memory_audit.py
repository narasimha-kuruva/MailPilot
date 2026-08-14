"""In-memory `AuditService` implementation.

Every record is also emitted through the structured logger, so the audit
trail is visible in process logs even before a persistent store (e.g. a
database) backs this in a later phase.
"""

from __future__ import annotations

from collections import defaultdict

from mailpilot.audit.service import AuditService
from mailpilot.logging_config import get_logger
from mailpilot.schemas.audit import AuditRecord

logger = get_logger(__name__)


class InMemoryAuditService(AuditService):
    def __init__(self) -> None:
        self._records: dict[str, list[AuditRecord]] = defaultdict(list)

    async def record(self, record: AuditRecord) -> None:
        self._records[record.conversation_id].append(record)
        logger.info("agent_tool_call", extra={"extra_fields": record.model_dump(mode="json")})

    async def get_history(self, conversation_id: str) -> list[AuditRecord]:
        return list(self._records.get(conversation_id, []))
