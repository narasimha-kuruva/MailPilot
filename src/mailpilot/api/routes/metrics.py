"""Metrics endpoint (Phase 5.7) -- see `mailpilot.observability.metrics`."""

from __future__ import annotations

from fastapi import APIRouter

from mailpilot.api.deps import MetricsDep
from mailpilot.schemas.metrics import MetricsSnapshot

router = APIRouter(tags=["observability"])


@router.get("/metrics", response_model=MetricsSnapshot)
async def get_metrics_snapshot(metrics: MetricsDep) -> MetricsSnapshot:
    """Counters since the process started: agent runs, tool calls, approvals, model usage.

    Needs no LLM or Gmail configuration, like `/health`.
    """
    return metrics.snapshot()
