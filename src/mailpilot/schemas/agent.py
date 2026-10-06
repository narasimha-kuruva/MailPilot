"""Domain models for agent goals, plans, and execution state.

These shapes define the contract between the API layer and the agent layer.
The actual planning/execution logic is implemented in Phase 2; Phase 1 only
establishes the data model so the interface is stable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class ApprovalStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class AgentRequest(BaseModel):
    """A natural-language instruction submitted by the user."""

    instruction: str = Field(..., min_length=1, max_length=10_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=200)


class PlannedStep(BaseModel):
    step_id: str
    description: str
    tool_name: str | None = None
    tool_args: dict = Field(default_factory=dict)
    requires_approval: bool = False


class AgentPlan(BaseModel):
    conversation_id: str
    goal: str
    steps: list[PlannedStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlanOutline(BaseModel):
    """Raw structured-output shape the LLM fills in for `Agent.plan()`.

    Deliberately just an ordered list of short sub-goals, not tool calls:
    which tool accomplishes a given step (and with what arguments) is only
    knowable once earlier steps have produced real data, so tool selection
    stays with the ReAct loop in `mailpilot.agent.graph` at execution time.
    This is preview/transparency data, not a rigid execution script.
    """

    steps: list[str] = Field(default_factory=list, description="Ordered, concise sub-goals.")


class AgentRunStatus(StrEnum):
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class PendingApproval(BaseModel):
    """Describes the single action blocked on a human decision."""

    tool_name: str
    tool_args: dict = Field(default_factory=dict)
    description: str


class ApprovalDecision(BaseModel):
    """A human's decision on a `PendingApproval`."""

    approved: bool


class AgentRunState(BaseModel):
    """In-memory/persistable state for one agent run."""

    conversation_id: str
    status: AgentRunStatus = AgentRunStatus.PLANNING
    plan: AgentPlan | None = None
    completed_step_ids: list[str] = Field(default_factory=list)
    pending_approval: PendingApproval | None = None
    final_response: str | None = None
