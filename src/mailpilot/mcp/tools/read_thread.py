"""`read_thread` MCP tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.email import EmailThread


class ReadThreadArgs(BaseModel):
    thread_id: str = Field(..., min_length=1, description="Gmail thread ID, as returned by search_emails.")


class ReadThreadTool(MCPTool):
    name = "read_thread"
    description = "Fetch every message in a Gmail conversation thread by its thread ID."
    args_schema = ReadThreadArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> EmailThread:
        args = self.args_schema.model_validate(kwargs)
        thread = await self._gmail_client.get_thread(args.thread_id)
        # HTML bodies are dropped for the same reason as in `read_email`.
        return thread.model_copy(
            update={"messages": [m.model_copy(update={"body_html": None}) for m in thread.messages]}
        )
