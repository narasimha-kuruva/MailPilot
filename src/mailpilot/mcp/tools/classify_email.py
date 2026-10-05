"""`classify_email` MCP tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.intelligence.service import IntelligenceService
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.intelligence import EmailClassification


class ClassifyEmailArgs(BaseModel):
    message_id: str = Field(..., min_length=1, description="Gmail message ID, as returned by search_emails.")


class ClassifyEmailTool(MCPTool):
    name = "classify_email"
    description = (
        "Classify a single email: category (client/internal/newsletter/notification/"
        "spam/personal/other), priority, whether it's urgent, and whether it requires action."
    )
    args_schema = ClassifyEmailArgs

    def __init__(self, gmail_client: GmailClient, intelligence_service: IntelligenceService) -> None:
        self._gmail_client = gmail_client
        self._intelligence_service = intelligence_service

    async def run(self, **kwargs: Any) -> EmailClassification:
        args = self.args_schema.model_validate(kwargs)
        message = await self._gmail_client.get_message(args.message_id)
        return await self._intelligence_service.classify_email(message)
