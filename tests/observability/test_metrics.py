"""Phase 5.7: metrics from the audit trail, the agent, and the chat model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from mailpilot.agent.graph import build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.observability.metrics import LLMUsageCallback, MetricsRegistry
from mailpilot.safety.in_memory_approval import InMemoryApprovalService
from mailpilot.schemas.agent import AgentRequest, AgentRunStatus, ApprovalStatus
from mailpilot.schemas.audit import AuditRecord, ToolCallStatus
from tests.fakes import FakeChatModel, FakeGmailClient


def _record(tool_name: str, status: ToolCallStatus, **kwargs) -> AuditRecord:
    return AuditRecord(conversation_id="c1", agent_request="x", tool_name=tool_name, status=status, **kwargs)


def test_audit_records_become_tool_approval_and_event_counts() -> None:
    metrics = MetricsRegistry()

    for record in [
        _record("search_emails", ToolCallStatus.SUCCESS, duration_ms=100.0),
        _record("search_emails", ToolCallStatus.FAILURE, duration_ms=300.0),
        _record("read_email", ToolCallStatus.SKIPPED),  # held or over the limit: didn't run
        _record("send_email", ToolCallStatus.SKIPPED, approval_status=ApprovalStatus.PENDING),
        _record("send_email", ToolCallStatus.SUCCESS, approval_status=ApprovalStatus.APPROVED, duration_ms=50.0),
        _record("send_email", ToolCallStatus.SKIPPED, approval_status=ApprovalStatus.REJECTED),
        _record("send_email", ToolCallStatus.SKIPPED, approval_status=ApprovalStatus.EXPIRED),
        _record("__execution_limit__", ToolCallStatus.FAILURE),
        _record("__llm_error__", ToolCallStatus.FAILURE),
    ]:
        metrics.observe_audit(record)

    snapshot = metrics.snapshot()
    assert snapshot.tool_calls.model_dump() == {
        "calls": 4,
        "succeeded": 2,
        "failed": 1,
        "skipped": 1,
        "average_duration_ms": 150.0,
    }
    assert snapshot.tools["search_emails"].average_duration_ms == 200.0
    assert snapshot.tools["read_email"].average_duration_ms is None
    assert snapshot.approvals.model_dump() == {
        "requested": 1,
        "approved": 1,
        "rejected": 1,
        "expired": 1,
        "cancelled": 0,
    }
    assert snapshot.run_events == {"execution_limit": 1, "llm_error": 1}


def test_run_outcomes_and_average_duration() -> None:
    metrics = MetricsRegistry()

    metrics.record_run(AgentRunStatus.COMPLETED, 1000.0)
    metrics.record_run(AgentRunStatus.AWAITING_APPROVAL, 500.0)
    metrics.record_run(AgentRunStatus.FAILED, 0.0)

    runs = metrics.snapshot().agent_runs
    assert (runs.total, runs.completed, runs.awaiting_approval, runs.failed) == (3, 1, 1, 1)
    assert runs.average_duration_ms == 500.0


def test_cost_is_estimated_only_when_prices_are_configured() -> None:
    unpriced = MetricsRegistry()
    priced = MetricsRegistry(input_usd_per_million_tokens=0.5, output_usd_per_million_tokens=2.0)
    for metrics in (unpriced, priced):
        metrics.record_llm_call(input_tokens=2_000_000, output_tokens=1_000_000)

    assert unpriced.snapshot().llm.estimated_cost_usd is None
    assert priced.snapshot().llm.estimated_cost_usd == 3.0


@pytest.mark.asyncio
async def test_usage_callback_counts_every_call_through_any_wrapper() -> None:
    metrics = MetricsRegistry()
    usage = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="a", usage_metadata=usage), AIMessage(content="b", usage_metadata=usage)]),
        callbacks=[LLMUsageCallback(metrics)],
    )

    await model.ainvoke("direct call")
    await model.bind(stop=["\n"]).ainvoke("call through a wrapper, like bind_tools")

    llm = metrics.snapshot().llm
    assert (llm.calls, llm.input_tokens, llm.output_tokens, llm.failed) == (2, 240, 60, 0)


@pytest.mark.asyncio
async def test_usage_callback_counts_failed_calls() -> None:
    metrics = MetricsRegistry()
    model = GenericFakeChatModel(messages=iter([]), callbacks=[LLMUsageCallback(metrics)])

    with pytest.raises(Exception):
        await model.ainvoke("no response queued")

    assert metrics.snapshot().llm.failed == 1


@pytest.mark.asyncio
async def test_agent_run_feeds_the_registry_end_to_end() -> None:
    metrics = MetricsRegistry()
    chat_model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "search_emails", "args": {"query": "is:unread"}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "c2"}]),
            AIMessage(content="Sent."),
        ]
    )
    gmail_client = FakeGmailClient()
    tools = build_tools(gmail_client)
    audit_service = InMemoryAuditService(metrics=metrics)
    agent = LangGraphAgent(
        agent_graph=build_agent_graph(chat_model, tools, audit_service, checkpointer=MemorySaver()),
        tools=tools,
        approval_service=InMemoryApprovalService(),
        audit_service=audit_service,
        metrics=metrics,
    )

    state = await agent.run(AgentRequest(instruction="reply and send", conversation_id="conv"))
    await agent.resume("conv", approved=True)

    snapshot = metrics.snapshot()
    assert state.status is AgentRunStatus.AWAITING_APPROVAL
    assert (snapshot.agent_runs.total, snapshot.agent_runs.awaiting_approval) == (1, 1)
    assert snapshot.tools["search_emails"].succeeded == 1
    assert snapshot.tools["send_email"].succeeded == 1
    assert (snapshot.approvals.requested, snapshot.approvals.approved) == (1, 1)
    history = await audit_service.get_history("conv")
    assert all(r.duration_ms is not None and r.duration_ms >= 0 for r in history if r.status is ToolCallStatus.SUCCESS)


class _ExplodingGraph:
    """Stands in for the compiled graph: the conversation is new, then the run itself raises."""

    def get_state(self, config: dict) -> SimpleNamespace:
        return SimpleNamespace(values={})

    async def ainvoke(self, state: dict, config: dict) -> dict:
        raise RuntimeError("checkpointer unavailable")


@pytest.mark.asyncio
async def test_a_run_that_raises_counts_as_failed() -> None:
    metrics = MetricsRegistry()
    tools = build_tools(FakeGmailClient())
    audit_service = InMemoryAuditService(metrics=metrics)
    graph = build_agent_graph(FakeChatModel([]), tools, audit_service, checkpointer=MemorySaver())
    graph.graph = _ExplodingGraph()
    agent = LangGraphAgent(
        agent_graph=graph,
        tools=tools,
        approval_service=InMemoryApprovalService(),
        audit_service=audit_service,
        metrics=metrics,
    )

    with pytest.raises(RuntimeError):
        await agent.run(AgentRequest(instruction="anything", conversation_id="conv"))

    runs = metrics.snapshot().agent_runs
    assert (runs.total, runs.failed) == (1, 1)


def test_metrics_endpoint_needs_no_llm_or_gmail_configuration(client: TestClient) -> None:
    response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"agent_runs", "tool_calls", "tools", "approvals", "run_events", "llm", "uptime_seconds"}
    assert body["llm"]["estimated_cost_usd"] is None
