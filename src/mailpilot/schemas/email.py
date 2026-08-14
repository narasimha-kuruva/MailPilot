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
