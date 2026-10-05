"""`read_email` MCP tool.

The HTML body is dropped from the result: it's several times the size of
the text body and the model gains nothing from markup. `body_text` is
always populated (derived from HTML when the message has no text part --
see `mailpilot.gmail.mime_utils.parse_message`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.email import EmailMessage


class ReadEmailArgs(BaseModel):
    message_id: str = Field(..., description="Gmail message ID, as returned by search_emails.")


class ReadEmailTool(MCPTool):
    name = "read_email"
    description = "Fetch the full content of a single email message by its Gmail message ID."
    args_schema = ReadEmailArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> EmailMessage:
        args = self.args_schema.model_validate(kwargs)
        message = await self._gmail_client.get_message(args.message_id)
        return message.model_copy(update={"body_html": None})
