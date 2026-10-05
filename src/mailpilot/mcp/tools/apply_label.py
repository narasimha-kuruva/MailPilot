"""`apply_label` MCP tool.

Applying a label needs no human approval (it only organises mail), with
one hard exception enforced in code: the `TRASH` and `SPAM` system labels
are refused outright via `mailpilot.safety.guardrails.assert_label_is_safe`,
because they remove a message from the user's view -- an agent must never
be able to make mail disappear, whether by mistake or by prompt injection.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.safety.guardrails import assert_label_is_safe


class ApplyLabelArgs(BaseModel):
    message_id: str = Field(..., min_length=1, description="Gmail message ID to label.")
    label_id: str = Field(..., min_length=1, description="Gmail label ID, as returned by list_labels.")


class ApplyLabelTool(MCPTool):
    name = "apply_label"
    description = (
        "Apply an existing Gmail label to a message. Cannot be used to trash a "
        "message or mark it as spam."
    )
    args_schema = ApplyLabelArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> dict[str, str]:
        args = self.args_schema.model_validate(kwargs)
        assert_label_is_safe(args.label_id)
        await self._gmail_client.apply_label(args.message_id, args.label_id)
        return {"status": "applied", "message_id": args.message_id, "label_id": args.label_id}
