"""In-process metrics (Phase 5.7): what the agent did, how long it took, what the model used.

Dependency-free counters behind `GET /api/v1/metrics`, fed from three
existing choke points rather than calls scattered through the agent:

- **The audit trail.** `InMemoryAuditService.record()` hands every
  `AuditRecord` to `observe_audit()`: tool calls by outcome and duration,
  approval requests and decisions, and run events (`__execution_limit__`,
  `__llm_error__`, `__deferred_followup__`).
- **The agent.** `LangGraphAgent.run()` reports each run's outcome and
  duration through `record_run()`.
- **The chat model.** `LLMUsageCallback` is attached to the model itself,
  so every call reports its token usage -- the agent loop, planning, and
  the model calls made inside tools (classify, summarize, extract, grounded
  drafting) -- whichever wrapper (`bind_tools`, `with_structured_output`)
  made it.

Cost is estimated only when per-token prices are configured
(`LLM_INPUT_USD_PER_MILLION_TOKENS` / `LLM_OUTPUT_USD_PER_MILLION_TOKENS`,
both 0 by default): prices change and differ by plan, so none are built
in, and a local Ollama model costs nothing per token.

Counters live in memory and reset when the process restarts, like the
other services in this phase: a live view, not a time series. To keep
history, poll the endpoint from a collector.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from mailpilot.schemas.agent import AgentRunStatus, ApprovalStatus
from mailpilot.schemas.audit import AuditRecord, ToolCallStatus
from mailpilot.schemas.metrics import (
    AgentRunMetrics,
    ApprovalMetrics,
    LLMMetrics,
    MetricsSnapshot,
    ToolMetrics,
)

# Audit records for an approval request or a decision that ran nothing.
_APPROVAL_ONLY = {
    ApprovalStatus.PENDING: "requested",
    ApprovalStatus.REJECTED: "rejected",
    ApprovalStatus.EXPIRED: "expired",
    ApprovalStatus.CANCELLED: "cancelled",
}


def _average(total: float, count: int) -> float | None:
    return round(total / count, 1) if count else None


@dataclass
class _ToolStats:
    calls: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    timed_calls: int = 0
    total_duration_ms: float = 0.0

    def add(self, status: ToolCallStatus, duration_ms: float | None) -> None:
        self.calls += 1
        if status is ToolCallStatus.SUCCESS:
            self.succeeded += 1
        elif status is ToolCallStatus.FAILURE:
            self.failed += 1
        else:
            self.skipped += 1
        if duration_ms is not None:
            self.timed_calls += 1
            self.total_duration_ms += duration_ms

    def merge(self, other: "_ToolStats") -> None:
        for name in ("calls", "succeeded", "failed", "skipped", "timed_calls", "total_duration_ms"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def snapshot(self) -> ToolMetrics:
        return ToolMetrics(
            calls=self.calls,
            succeeded=self.succeeded,
            failed=self.failed,
            skipped=self.skipped,
            average_duration_ms=_average(self.total_duration_ms, self.timed_calls),
        )


class MetricsRegistry:
    def __init__(
        self,
        input_usd_per_million_tokens: float = 0.0,
        output_usd_per_million_tokens: float = 0.0,
    ) -> None:
        self._input_price = input_usd_per_million_tokens
        self._output_price = output_usd_per_million_tokens
        # Model callbacks may run on another thread; everything else on the event loop.
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)
        self._started_monotonic = time.monotonic()
        self._runs: Counter[AgentRunStatus] = Counter()
        self._run_duration_ms = 0.0
        self._tools: defaultdict[str, _ToolStats] = defaultdict(_ToolStats)
        self._approvals: Counter[str] = Counter()
        self._events: Counter[str] = Counter()
        self._llm_started = 0
        self._llm_completed = 0
        self._llm_failures = 0
        self._input_tokens = 0
        self._output_tokens = 0

    def observe_audit(self, record: AuditRecord) -> None:
        with self._lock:
            # Run events use reserved `__name__` tool names; they aren't tool calls.
            if record.tool_name.startswith("__"):
                self._events[record.tool_name.strip("_")] += 1
                return
            if record.approval_status in _APPROVAL_ONLY:
                self._approvals[_APPROVAL_ONLY[record.approval_status]] += 1
                return
            if record.approval_status is ApprovalStatus.APPROVED:
                self._approvals["approved"] += 1
            self._tools[record.tool_name].add(record.status, record.duration_ms)

    def record_run(self, status: AgentRunStatus, duration_ms: float) -> None:
        with self._lock:
            self._runs[status] += 1
            self._run_duration_ms += duration_ms

    def record_llm_start(self) -> None:
        with self._lock:
            self._llm_started += 1

    def record_llm_call(self, input_tokens: int, output_tokens: int) -> None:
        """A model call that completed, with the tokens it used."""
        with self._lock:
            self._llm_completed += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

    def record_llm_failure(self) -> None:
        with self._lock:
            self._llm_failures += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            totals = _ToolStats()
            for stats in self._tools.values():
                totals.merge(stats)
            run_count = sum(self._runs.values())
            # A call that started but never ended was cancelled -- almost always
            # by a timeout, which LangChain reports neither as an end nor as an
            # error -- or is still in flight.
            llm_calls = max(self._llm_started, self._llm_completed + self._llm_failures)
            if self._input_price or self._output_price:
                cost: float | None = round(
                    (self._input_tokens * self._input_price + self._output_tokens * self._output_price) / 1_000_000,
                    6,
                )
            else:
                cost = None
            return MetricsSnapshot(
                started_at=self._started_at,
                uptime_seconds=round(time.monotonic() - self._started_monotonic, 1),
                agent_runs=AgentRunMetrics(
                    total=run_count,
                    completed=self._runs[AgentRunStatus.COMPLETED],
                    failed=self._runs[AgentRunStatus.FAILED],
                    awaiting_approval=self._runs[AgentRunStatus.AWAITING_APPROVAL],
                    average_duration_ms=_average(self._run_duration_ms, run_count),
                ),
                tool_calls=totals.snapshot(),
                tools={name: stats.snapshot() for name, stats in sorted(self._tools.items())},
                approvals=ApprovalMetrics(
                    requested=self._approvals["requested"],
                    approved=self._approvals["approved"],
                    rejected=self._approvals["rejected"],
                    expired=self._approvals["expired"],
                    cancelled=self._approvals["cancelled"],
                ),
                run_events=dict(self._events),
                llm=LLMMetrics(
                    calls=llm_calls,
                    completed=self._llm_completed,
                    failed=self._llm_failures,
                    unfinished=llm_calls - self._llm_completed - self._llm_failures,
                    input_tokens=self._input_tokens,
                    output_tokens=self._output_tokens,
                    estimated_cost_usd=cost,
                ),
            )


class LLMUsageCallback(BaseCallbackHandler):
    """Reports each chat-model call's token usage to a `MetricsRegistry`.

    Attach it to the model (`callbacks=[...]` at construction): LangChain
    then runs it for every call, including through `bind_tools` and
    `with_structured_output`. Models that don't report usage still count as
    a call, with zero tokens. Every start is counted, because a call cancelled
    by a timeout never reaches `on_llm_end` or `on_llm_error`.
    """

    run_inline = True  # update the counters directly instead of in an executor thread

    def __init__(self, metrics: MetricsRegistry) -> None:
        self._metrics = metrics

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        self._metrics.record_llm_start()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._metrics.record_llm_start()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        input_tokens = output_tokens = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
        self._metrics.record_llm_call(input_tokens, output_tokens)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._metrics.record_llm_failure()
