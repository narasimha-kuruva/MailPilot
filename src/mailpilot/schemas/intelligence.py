"""Structured outputs for email understanding (Phase 3).

The LLM is asked to return these shapes directly (via
`BaseChatModel.with_structured_output`) rather than free-form text, so
downstream code (the agent graph, the API) can branch on typed fields
instead of parsing prose.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class EmailCategory(StrEnum):
    CLIENT = "client"
    INTERNAL = "internal"
    NEWSLETTER = "newsletter"
    NOTIFICATION = "notification"
    SPAM = "spam"
    PERSONAL = "personal"
    OTHER = "other"


class EmailClassification(BaseModel):
    """Output of `IntelligenceService.classify_email`."""

    category: EmailCategory
    priority: Priority
    is_urgent: bool = Field(description="True if this needs attention within roughly a day.")
    requires_action: bool = Field(description="True if the recipient is expected to do something.")
    reasoning: str = Field(description="One or two sentences grounded in the email's actual content.")


class ThreadSummary(BaseModel):
    """Output of `IntelligenceService.summarize_thread`."""

    thread_id: str
    summary: str = Field(description="A few sentences covering what the thread is about and where it stands.")
    key_points: list[str] = Field(default_factory=list)


class ExtractedTask(BaseModel):
    """One actionable item found in an email/thread.

    `deadline` and `owner` are `None` when the source text doesn't state
    them -- they must never be guessed (see `mailpilot.prompts`).
    """

    action: str
    deadline: str | None = None
    owner: str | None = None
    source_thread_id: str | None = None
    source_message_id: str | None = None


class TaskExtractionResult(BaseModel):
    """Output of `IntelligenceService.extract_tasks`."""

    tasks: list[ExtractedTask] = Field(default_factory=list)


class GroundedDraft(BaseModel):
    """A reply draft the model produced, plus how it was checked.

    `grounded` and `validation_notes` are filled in by
    `mailpilot.safety.guardrails.find_unsupported_claims` after generation,
    not by the model itself -- see `mailpilot.agent.reply_drafting`.
    """

    subject: str
    body_text: str
    source_references: list[str] = Field(
        default_factory=list, description="IDs of the retrieved chunks/messages used to ground this reply."
    )
    grounded: bool = True
    validation_notes: list[str] = Field(default_factory=list)


class DraftGroundedReplyResult(BaseModel):
    """Output of the `draft_grounded_reply` MCP tool."""

    draft_id: str
    subject: str
    grounded: bool
    validation_notes: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
