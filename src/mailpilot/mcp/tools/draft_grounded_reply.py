"""`draft_grounded_reply` MCP tool.

Composes the Phase 3.4 workflow (`mailpilot.agent.reply_drafting`) with the
recipient guardrail and `create_draft`. Like `create_draft`, this never
sends anything -- only `send_email`, gated behind human approval, does
that (see `mailpilot.safety.policy`).

Recipients are "reply all" to the thread's last message (its sender plus
its To/Cc, minus the user's own address -- see
`mailpilot.gmail.mime_utils.reply_all_recipients`). The model's structured
output (`GroundedDraft`) has no recipient field at all, so there is no code
path by which email content could redirect the reply; every address comes
from the thread itself.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.agent.reply_drafting import draft_grounded_reply
from mailpilot.gmail.client import GmailClient
from mailpilot.gmail.mime_utils import reply_all_recipients
from mailpilot.rag.service import RAGService
from mailpilot.safety.guardrails import validate_reply_recipients
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.email import DraftEmail
from mailpilot.schemas.intelligence import DraftGroundedReplyResult


class DraftGroundedReplyArgs(BaseModel):
    thread_id: str = Field(..., min_length=1, description="Gmail thread ID to reply within.")
    intent: str = Field(
        ..., min_length=1, description="What the reply should accomplish, e.g. 'confirm the meeting time and next steps'."
    )


class DraftGroundedReplyTool(MCPTool):
    name = "draft_grounded_reply"
    description = (
        "Draft a reply to a thread, grounded in MailPilot's knowledge store: the indexed "
        "threads and documents (pricing notes, policies, past conversations) that you can't "
        "see otherwise. Use it whenever a reply needs facts beyond the thread itself -- e.g. "
        "'using our pricing notes' -- instead of asking the user for them. Never invents "
        "recipients, facts, dates, or prices. Creates a Gmail draft; never sends it."
    )
    args_schema = DraftGroundedReplyArgs

    def __init__(
        self,
        gmail_client: GmailClient,
        rag_service: RAGService,
        chat_model: Any,
        max_context_chars: int = 6000,
        own_email: str | None = None,
        top_k: int = 5,
    ) -> None:
        self._gmail_client = gmail_client
        self._rag_service = rag_service
        self._chat_model = chat_model
        self._max_context_chars = max_context_chars
        self._top_k = top_k  # chunks retrieved before packing (Settings.rag_top_k)
        # The user's own address, if configured (Settings.gmail_user_email),
        # so a reply-all doesn't include themselves.
        self._own_email = own_email

    async def run(self, **kwargs: Any) -> DraftGroundedReplyResult:
        args = self.args_schema.model_validate(kwargs)
        thread = await self._gmail_client.get_thread(args.thread_id)
        if not thread.messages:
            raise ValueError(f"Thread '{args.thread_id}' has no messages to reply to.")

        grounded_draft = await draft_grounded_reply(
            thread=thread,
            intent=args.intent,
            chat_model=self._chat_model,
            rag_service=self._rag_service,
            top_k=self._top_k,
            max_context_chars=self._max_context_chars,
        )

        to, cc = reply_all_recipients(thread.messages[-1], own_email=self._own_email)
        draft_email = DraftEmail(
            thread_id=thread.thread_id,
            to=to,
            cc=cc,
            subject=grounded_draft.subject or f"Re: {thread.subject}",
            body_text=grounded_draft.body_text,
        )

        # Raises GuardrailViolation (uncaught here -> surfaced to the graph's
        # tool-error handling) if the model added a recipient not already in
        # the thread.
        validate_reply_recipients(draft_email, thread)

        created = await self._gmail_client.create_draft(draft_email)

        return DraftGroundedReplyResult(
            draft_id=created.draft_id,
            subject=draft_email.subject,
            grounded=grounded_draft.grounded,
            validation_notes=grounded_draft.validation_notes,
            source_references=grounded_draft.source_references,
        )
