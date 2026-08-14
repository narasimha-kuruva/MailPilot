"""`send_email` MCP tool.

Sends a previously created draft. This tool performs no approval check of
its own: the agent graph (`mailpilot.agent.graph`) never routes a
`send_email` call to the automatic tool-execution node — it always stops
at the approval gate first (`mailpilot.safety.policy.requires_approval`)
and only calls this tool's `run()` from `LangGraphAgent.resume()`, after an
explicit human approval decision. Keeping the check out of the tool itself
means there is exactly one code path that can authorize a send.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.email import EmailMessage


class SendEmailArgs(BaseModel):
    draft_id: str = Field(..., description="ID of a draft previously created with create_draft.")


class SendEmailTool(MCPTool):
    name = "send_email"
    description = "Send a previously created Gmail draft. Requires human approval."
    args_schema = SendEmailArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> EmailMessage:
        args = self.args_schema.model_validate(kwargs)
        return await self._gmail_client.send_email(args.draft_id)
