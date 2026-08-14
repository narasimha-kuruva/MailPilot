"""`list_labels` MCP tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.email import Label


class ListLabelsArgs(BaseModel):
    """No arguments required."""


class ListLabelsTool(MCPTool):
    name = "list_labels"
    description = "List all Gmail labels available in the user's mailbox."
    args_schema = ListLabelsArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> list[Label]:
        self.args_schema.model_validate(kwargs)
        return await self._gmail_client.list_labels()
