"""Approval service interface.

The concrete implementation (Phase 2+) tracks pending approval requests
per (conversation, step) and is the single gate the agent layer must pass
before invoking any sensitive tool such as `send_email`. `send_email`
itself performs no approval logic (see `mailpilot.gmail.client`) — this
service is where that check belongs, so it cannot be bypassed by a new
caller forgetting to add it elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mailpilot.schemas.agent import ApprovalStatus


class ApprovalService(ABC):
    """Interface for requesting and recording human approval decisions."""

    @abstractmethod
    async def request_approval(
        self, conversation_id: str, step_id: str, description: str
    ) -> ApprovalStatus:
        raise NotImplementedError

    @abstractmethod
    async def record_decision(
        self, conversation_id: str, step_id: str, approved: bool
    ) -> ApprovalStatus:
        raise NotImplementedError

    @abstractmethod
    async def get_status(self, conversation_id: str, step_id: str) -> ApprovalStatus:
        raise NotImplementedError

    @abstractmethod
    async def mark_expired(self, conversation_id: str, step_id: str) -> ApprovalStatus:
        """Record that a pending approval's TTL elapsed before a decision was made."""
        raise NotImplementedError
