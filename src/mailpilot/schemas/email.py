"""Domain models representing Gmail messages, threads, and labels.

These are provider-agnostic shapes used across the agent, MCP tool, and
Gmail integration layers, so the rest of the system does not depend
directly on the raw Gmail API JSON structure.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EmailAddress(BaseModel):
    name: str | None = None
    email: str


class EmailMessage(BaseModel):
    message_id: str
    thread_id: str
    subject: str = ""
    sender: EmailAddress
    to: list[EmailAddress] = Field(default_factory=list)
    cc: list[EmailAddress] = Field(default_factory=list)
    snippet: str = ""
    body_text: str | None = None
    body_html: str | None = None
    labels: list[str] = Field(default_factory=list)
    received_at: datetime | None = None


class EmailSummary(BaseModel):
    """The compact, model-facing view of a message returned by `search_emails`.

    Deliberately excludes bodies: a search over a real mailbox returns tens
    of kilobytes of (mostly HTML) body text, far more than fits in one tool
    result, and the agent only needs enough to decide *which* messages to
    read in full via `read_email` / `read_thread`.
    """

    message_id: str
    thread_id: str
    subject: str = ""
    sender: EmailAddress
    to: list[EmailAddress] = Field(default_factory=list)
    snippet: str = ""
    labels: list[str] = Field(default_factory=list)
    is_unread: bool = False
    received_at: datetime | None = None

    @classmethod
    def from_message(cls, message: "EmailMessage") -> "EmailSummary":
        return cls(
            message_id=message.message_id,
            thread_id=message.thread_id,
            subject=message.subject,
            sender=message.sender,
            to=message.to,
            snippet=message.snippet,
            labels=message.labels,
            is_unread="UNREAD" in message.labels,
            received_at=message.received_at,
        )


class EmailThread(BaseModel):
    thread_id: str
    subject: str = ""
    messages: list[EmailMessage] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class Label(BaseModel):
    label_id: str
    name: str
    type: str = "user"


class DraftEmail(BaseModel):
    thread_id: str | None = None
    to: list[EmailAddress]
    cc: list[EmailAddress] = Field(default_factory=list)
    subject: str
    body_text: str


class CreatedDraft(BaseModel):
    draft_id: str
    message: EmailMessage
