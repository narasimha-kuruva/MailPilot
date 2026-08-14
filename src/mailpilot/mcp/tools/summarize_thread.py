"""`summarize_thread` MCP tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.intelligence.service import IntelligenceService
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.intelligence import ThreadSummary


class SummarizeThreadArgs(BaseModel):
    thread_id: str = Field(..., description="Gmail thread ID, as returned by search_emails.")


class SummarizeThreadTool(MCPTool):
    name = "summarize_thread"
    description = "Summarize an email thread: what it's about, where it stands, and its key points."
    args_schema = SummarizeThreadArgs

    def __init__(self, gmail_client: GmailClient, intelligence_service: IntelligenceService) -> None:
        self._gmail_client = gmail_client
        self._intelligence_service = intelligence_service

    async def run(self, **kwargs: Any) -> ThreadSummary:
        args = self.args_schema.model_validate(kwargs)
        thread = await self._gmail_client.get_thread(args.thread_id)
        return await self._intelligence_service.summarize_thread(thread)
