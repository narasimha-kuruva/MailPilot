"""In-memory `AuditService` implementation.

Records are lost on restart; `mailpilot.persistence.sqlite.SqliteAuditService`
keeps them. Redaction, logging, and metrics happen in `AuditService.record()`.
"""

from __future__ import annotations

from collections import defaultdict

from mailpilot.audit.service import AuditService
from mailpilot.observability.metrics import MetricsRegistry
from mailpilot.schemas.audit import AuditRecord


class InMemoryAuditService(AuditService):
    def __init__(self, metrics: MetricsRegistry | None = None) -> None:
        super().__init__(metrics)
        self._records: dict[str, list[AuditRecord]] = defaultdict(list)

    async def _append(self, record: AuditRecord) -> None:
        self._records[record.conversation_id].append(record)

    async def get_history(self, conversation_id: str) -> list[AuditRecord]:
        return list(self._records.get(conversation_id, []))
