"""`search_emails` MCP tool.

Returns compact `EmailSummary` rows (ids, sender, subject, snippet, labels)
rather than full `EmailMessage`s. Against a real mailbox, five full
messages with HTML bodies serialize to ~30k characters -- far beyond the
per-result cap -- which left the model with a truncated, unparseable
result and sent it into a re-search loop. Bodies are fetched on demand
with `read_email` / `read_thread`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.email import EmailSummary


class SearchEmailsArgs(BaseModel):
    query: str = Field(
        ..., min_length=1, description="Gmail search query, e.g. 'from:alice is:unread newer_than:7d'."
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Use the smallest number that answers the question; every result costs Gmail quota.",
    )


class SearchEmailsTool(MCPTool):
    name = "search_emails"
    description = (
        "Search the user's Gmail mailbox with Gmail search syntax. Returns a compact summary "
        "per match (message_id, thread_id, sender, subject, snippet, labels, is_unread) -- "
        "NOT the body. Use read_email or read_thread with the returned ids to get full content."
    )
    args_schema = SearchEmailsArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> list[EmailSummary]:
        args = self.args_schema.model_validate(kwargs)
        messages = await self._gmail_client.search_messages(args.query, args.max_results)
        return [EmailSummary.from_message(message) for message in messages]
