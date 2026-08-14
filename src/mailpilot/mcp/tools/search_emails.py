"""`search_emails` MCP tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.email import EmailMessage


class SearchEmailsArgs(BaseModel):
    query: str = Field(
        ..., description="Gmail search query, e.g. 'from:alice is:unread newer_than:7d'."
    )
    max_results: int = Field(default=25, ge=1, le=100)


class SearchEmailsTool(MCPTool):
    name = "search_emails"
    description = "Search the user's Gmail mailbox with Gmail search syntax and return matching messages."
    args_schema = SearchEmailsArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> list[EmailMessage]:
        args = self.args_schema.model_validate(kwargs)
        return await self._gmail_client.search_messages(args.query, args.max_results)
