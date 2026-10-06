"""In-memory `ApprovalService` implementation.

Suitable for a single-process deployment (Phase 2). State does not survive
a process restart. A persistent implementation (e.g. database-backed) can
replace this later without touching callers, since they depend only on the
`ApprovalService` interface.
"""

from __future__ import annotations

from mailpilot.safety.approval import ApprovalService
from mailpilot.schemas.agent import ApprovalStatus


class InMemoryApprovalService(ApprovalService):
    def __init__(self) -> None:
        self._decisions: dict[tuple[str, str], ApprovalStatus] = {}

    async def request_approval(
        self, conversation_id: str, step_id: str, description: str
    ) -> ApprovalStatus:
        self._decisions[(conversation_id, step_id)] = ApprovalStatus.PENDING
        return ApprovalStatus.PENDING

    async def record_decision(
        self, conversation_id: str, step_id: str, approved: bool
    ) -> ApprovalStatus:
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        self._decisions[(conversation_id, step_id)] = status
        return status

    async def get_status(self, conversation_id: str, step_id: str) -> ApprovalStatus:
        return self._decisions.get((conversation_id, step_id), ApprovalStatus.NOT_REQUIRED)

    async def mark_expired(self, conversation_id: str, step_id: str) -> ApprovalStatus:
        self._decisions[(conversation_id, step_id)] = ApprovalStatus.EXPIRED
        return ApprovalStatus.EXPIRED

    async def mark_cancelled(self, conversation_id: str, step_id: str) -> ApprovalStatus:
        self._decisions[(conversation_id, step_id)] = ApprovalStatus.CANCELLED
        return ApprovalStatus.CANCELLED
