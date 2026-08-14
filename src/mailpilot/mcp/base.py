"""MCP tool interface.

Concrete tools (Phase 3+), such as `search_emails`, `read_email`,
`read_thread`, `list_labels`, `apply_label`, `create_draft`, and
`send_email`, will subclass `MCPTool`, declare a Pydantic `args_schema` for
validation, and delegate to a `GmailClient` instance. Keeping tools thin
wrappers around the Gmail client means the same client implementation can
be reused outside of MCP (e.g. directly from the agent layer in tests).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class MCPTool(ABC):
    """Interface for a single MCP-exposed tool."""

    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Validate `kwargs` against `args_schema` and execute the tool."""
        raise NotImplementedError
