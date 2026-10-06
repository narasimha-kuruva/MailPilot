"""Where a run that stopped for human approval keeps the action it's waiting on.

`LangGraphAgent.run()` puts one entry per conversation when it reaches the
approval gate; `resume()` takes it, exactly once. `take()` must be atomic:
two decisions arriving for one conversation get the entry once between
them, which is what stops one approval from running an action twice.

`requested_at` is wall-clock time (seconds since the epoch), so an
approval's expiry still means something after a restart -- see
`mailpilot.persistence.sqlite.SqlitePendingCallStore`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class PendingToolCall(BaseModel):
    """A tool call the agent wants to make but that needs human sign-off."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]


class PendingCallStore(ABC):
    @abstractmethod
    async def put(self, conversation_id: str, call: PendingToolCall, requested_at: float) -> None:
        """Record the call a conversation now waits on (replacing any earlier one)."""

    @abstractmethod
    async def take(self, conversation_id: str) -> tuple[PendingToolCall, float] | None:
        """Remove and return the conversation's pending call, or None. Atomic."""


class InMemoryPendingCallStore(PendingCallStore):
    def __init__(self) -> None:
        self._calls: dict[str, tuple[PendingToolCall, float]] = {}

    async def put(self, conversation_id: str, call: PendingToolCall, requested_at: float) -> None:
        self._calls[conversation_id] = (call, requested_at)

    async def take(self, conversation_id: str) -> tuple[PendingToolCall, float] | None:
        return self._calls.pop(conversation_id, None)  # no await in between: atomic on the event loop
