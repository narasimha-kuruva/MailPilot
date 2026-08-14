"""`apply_label` MCP tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool


class ApplyLabelArgs(BaseModel):
    message_id: str = Field(..., description="Gmail message ID to label.")
    label_id: str = Field(..., description="Gmail label ID, as returned by list_labels.")


class ApplyLabelTool(MCPTool):
    name = "apply_label"
    description = "Apply an existing Gmail label to a message."
    args_schema = ApplyLabelArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> dict[str, str]:
        args = self.args_schema.model_validate(kwargs)
        await self._gmail_client.apply_label(args.message_id, args.label_id)
        return {"status": "applied", "message_id": args.message_id, "label_id": args.label_id}
