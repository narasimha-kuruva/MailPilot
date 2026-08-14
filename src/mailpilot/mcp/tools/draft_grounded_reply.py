"""`draft_grounded_reply` MCP tool.

Composes the Phase 3.4 workflow (`mailpilot.agent.reply_drafting`) with the
recipient guardrail and `create_draft`. Like `create_draft`, this never
sends anything -- only `send_email`, gated behind human approval, does
that (see `mailpilot.safety.policy`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.agent.reply_drafting import draft_grounded_reply
from mailpilot.gmail.client import GmailClient
from mailpilot.rag.service import RAGService
from mailpilot.safety.guardrails import validate_reply_recipients
from mailpilot.mcp.base import MCPTool
from mailpilot.schemas.email import DraftEmail
from mailpilot.schemas.intelligence import DraftGroundedReplyResult


class DraftGroundedReplyArgs(BaseModel):
    thread_id: str = Field(..., description="Gmail thread ID to reply within.")
    intent: str = Field(
        ..., description="What the reply should accomplish, e.g. 'confirm the meeting time and next steps'."
    )


class DraftGroundedReplyTool(MCPTool):
    name = "draft_grounded_reply"
    description = (
        "Read a thread, retrieve relevant historical context, and draft a grounded reply "
        "(never inventing recipients, facts, dates, or prices). Creates a Gmail draft; "
        "never sends it."
    )
    args_schema = DraftGroundedReplyArgs

    def __init__(
        self,
        gmail_client: GmailClient,
        rag_service: RAGService,
        chat_model: Any,
        max_context_chars: int = 6000,
    ) -> None:
        self._gmail_client = gmail_client
        self._rag_service = rag_service
        self._chat_model = chat_model
        self._max_context_chars = max_context_chars

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
            max_context_chars=self._max_context_chars,
        )

        last_message = thread.messages[-1]
        draft_email = DraftEmail(
            thread_id=thread.thread_id,
            to=[last_message.sender],
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
