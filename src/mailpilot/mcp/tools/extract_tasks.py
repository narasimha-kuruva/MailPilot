"""`extract_tasks` MCP tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.intelligence.service import IntelligenceService
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.intelligence import TaskExtractionResult


class ExtractTasksArgs(BaseModel):
    thread_id: str = Field(..., min_length=1, description="Gmail thread ID, as returned by search_emails.")


class ExtractTasksTool(MCPTool):
    name = "extract_tasks"
    description = (
        "Extract concrete, actionable tasks from an email thread. Deadlines and owners "
        "are only filled in when the thread states them explicitly."
    )
    args_schema = ExtractTasksArgs

    def __init__(self, gmail_client: GmailClient, intelligence_service: IntelligenceService) -> None:
        self._gmail_client = gmail_client
        self._intelligence_service = intelligence_service

    async def run(self, **kwargs: Any) -> TaskExtractionResult:
        args = self.args_schema.model_validate(kwargs)
        thread = await self._gmail_client.get_thread(args.thread_id)
        return await self._intelligence_service.extract_tasks(thread)
