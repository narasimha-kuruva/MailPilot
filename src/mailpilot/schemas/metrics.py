"""Response shape of `GET /api/v1/metrics` (Phase 5.7) -- see `mailpilot.observability.metrics`."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AgentRunMetrics(BaseModel):
    """`run()` calls by outcome. A run that pauses for approval counts once, as awaiting_approval."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    awaiting_approval: int = 0
    average_duration_ms: float | None = None


class ToolMetrics(BaseModel):
    """Tool executions by outcome. `skipped` covers calls held or cut off by a limit,
    and approved actions not run again (duplicates)."""

    calls: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    average_duration_ms: float | None = Field(
        default=None, description="Mean wall-clock time of the calls that ran, retries included."
    )


class ApprovalMetrics(BaseModel):
    requested: int = 0
    approved: int = 0
    rejected: int = 0
    expired: int = 0
    cancelled: int = 0


class LLMMetrics(BaseModel):
    """Every chat-model call, including retries and the calls made inside tools."""

    calls: int = 0
    completed: int = 0
    failed: int = Field(default=0, description="Raised an error.")
    unfinished: int = Field(
        default=0, description="Started but never ended: cancelled (usually by a timeout) or still running."
    )
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = Field(
        default=None, description="Null unless per-token prices are configured (LLM_*_USD_PER_MILLION_TOKENS)."
    )


class MetricsSnapshot(BaseModel):
    started_at: datetime
    uptime_seconds: float
    agent_runs: AgentRunMetrics
    tool_calls: ToolMetrics = Field(description="All tools combined.")
    tools: dict[str, ToolMetrics] = Field(default_factory=dict, description="Per tool name.")
    approvals: ApprovalMetrics
    run_events: dict[str, int] = Field(
        default_factory=dict, description="Runs stopped early or flagged, e.g. execution_limit, llm_error."
    )
    llm: LLMMetrics
