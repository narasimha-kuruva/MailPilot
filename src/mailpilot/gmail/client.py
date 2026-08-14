"""Gmail client interface.

The concrete implementation (Phase 2+) will wrap an OAuth-authenticated
`googleapiclient` Gmail service behind this provider-agnostic contract, so
MCP tools and the agent layer never touch the raw Gmail API surface
directly and the provider could be swapped later if needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mailpilot.schemas.email import (
    CreatedDraft,
    DraftEmail,
    EmailMessage,
    EmailThread,
    Label,
)


class GmailClient(ABC):
    """Interface for Gmail read/write operations."""

    @abstractmethod
    async def search_messages(
        self, query: str, max_results: int = 25
    ) -> list[EmailMessage]:
        raise NotImplementedError

    @abstractmethod
    async def get_message(self, message_id: str) -> EmailMessage:
        raise NotImplementedError

    @abstractmethod
    async def get_thread(self, thread_id: str) -> EmailThread:
        raise NotImplementedError

    @abstractmethod
    async def list_labels(self) -> list[Label]:
        raise NotImplementedError

    @abstractmethod
    async def apply_label(self, message_id: str, label_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def create_draft(self, draft: DraftEmail) -> CreatedDraft:
        raise NotImplementedError

    @abstractmethod
    async def get_draft(self, draft_id: str) -> CreatedDraft:
        """Fetch a previously created draft's actual content.

        Used to show a human approver the real To/Subject/Body before they
        approve `send_email` -- the approval request otherwise only has a
        `draft_id`, which isn't reviewable on its own (see
        `mailpilot.agent.langgraph_agent`).
        """
        raise NotImplementedError

    @abstractmethod
    async def send_email(self, draft_id: str) -> EmailMessage:
        """Send a previously created draft.

        Callers must have obtained explicit human approval (via the safety
        layer) before invoking this. This method itself is a thin API
        wrapper and performs no approval checks of its own — that
        responsibility belongs to the agent/safety layers so it cannot be
        bypassed by adding a new caller here.
        """
        raise NotImplementedError
