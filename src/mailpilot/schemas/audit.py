"""Domain model for a single audit log entry.

Every tool invocation the agent makes should produce one `AuditRecord`,
capturing what was requested, what was done, and whether a human approved
it, so the full execution history of a run can be reconstructed later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from mailpilot.schemas.agent import ApprovalStatus


class ToolCallStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class AuditRecord(BaseModel):
    conversation_id: str
    step_id: str | None = None
    agent_request: str
    tool_name: str
    tool_args: dict = Field(default_factory=dict)
    status: ToolCallStatus
    result_summary: str | None = None
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
