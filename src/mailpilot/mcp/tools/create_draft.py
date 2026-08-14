"""`create_draft` MCP tool.

When `thread_id` is provided (this is a reply, not a new message), the
recipients are validated against that thread the same way
`draft_grounded_reply` validates its own drafts (see
`mailpilot.safety.guardrails.validate_reply_recipients`) -- this tool is
callable directly by the agent, so it needs the same guardrail, not just
the composite grounded-reply workflow. A brand-new message (no
`thread_id`) has no known-safe recipient set to validate against; that
remains an accepted, lower-risk path since it only ever creates a draft --
see `mailpilot.mcp.tools.send_email` for why nothing gets sent without a
separate human approval step that now shows the real recipient (Phase 5).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.gmail.mime_utils import parse_address
from mailpilot.mcp.base import MCPTool
from mailpilot.safety.guardrails import validate_reply_recipients
from mailpilot.schemas.email import CreatedDraft, DraftEmail


class CreateDraftArgs(BaseModel):
    to: list[str] = Field(..., min_length=1, description="Recipient email addresses.")
    cc: list[str] = Field(default_factory=list, description="CC email addresses.")
    subject: str = Field(..., min_length=1)
    body_text: str = Field(..., min_length=1)
    thread_id: str | None = Field(default=None, description="Existing thread to reply within, if any.")


class CreateDraftTool(MCPTool):
    name = "create_draft"
    description = "Create a Gmail draft reply or new message. This never sends anything."
    args_schema = CreateDraftArgs

    def __init__(self, gmail_client: GmailClient) -> None:
        self._gmail_client = gmail_client

    async def run(self, **kwargs: Any) -> CreatedDraft:
        args = self.args_schema.model_validate(kwargs)
        draft = DraftEmail(
            thread_id=args.thread_id,
            to=[parse_address(address) for address in args.to],
            cc=[parse_address(address) for address in args.cc],
            subject=args.subject,
            body_text=args.body_text,
        )

        if args.thread_id:
            thread = await self._gmail_client.get_thread(args.thread_id)
            validate_reply_recipients(draft, thread)

        return await self._gmail_client.create_draft(draft)
