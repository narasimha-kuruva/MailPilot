"""Audit service interface.

The concrete implementation (Phase 2+) will persist one `AuditRecord` per
tool invocation (request, tool, args, result, approval status, timestamp)
so a run's full execution history can be reconstructed and reviewed.

Implementations must store records redacted (`mailpilot.redaction`), as
`InMemoryAuditService` does: an audit trail outlives the request and is
read by people other than the one who made it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mailpilot.schemas.audit import AuditRecord


class AuditService(ABC):
    """Interface for recording and retrieving agent execution history."""

    @abstractmethod
    async def record(self, record: AuditRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_history(self, conversation_id: str) -> list[AuditRecord]:
        raise NotImplementedError
