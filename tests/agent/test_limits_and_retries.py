"""Phase 4.5/4.8: bounded retries on transient tool errors, and hard execution limits."""

from __future__ import annotations

from dataclasses import fields

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from mailpilot.agent.graph import AgentLimits, build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.config import Settings
from mailpilot.mcp.base import MCPTool
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.safety.in_memory_approval import InMemoryApprovalService
from mailpilot.schemas.agent import AgentRequest, AgentRunStatus
from mailpilot.schemas.audit import ToolCallStatus
from tests.fakes import FakeChatModel, FakeGmailClient


async def _no_sleep(_seconds: float) -> None:
    return None


class _EmptyArgs(BaseModel):
    pass


class _RetryableError(Exception):
    """Mimics an HttpError with a transient status code."""

    class _Resp:
        status = 503

    def __init__(self) -> None:
        super().__init__("temporary failure")
        self.resp = self._Resp()


class _FlakyTool(MCPTool):
    name = "flaky_tool"
    description = "fails transiently a fixed number of times, then succeeds"
    args_schema = _EmptyArgs

    def __init__(self, fail_times: int) -> None:
        self.attempts = 0
        self._fail_times = fail_times

    async def run(self, **kwargs: object) -> dict[str, bool]:
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise _RetryableError()
        return {"ok": True}


class _AlwaysFailsTool(MCPTool):
    name = "bad_tool"
    description = "always fails with a permanent error"
    args_schema = _EmptyArgs

    def __init__(self) -> None:
        self.attempts = 0

    async def run(self, **kwargs: object) -> None:
        self.attempts += 1
        raise ValueError("invalid configuration")


def _agent(chat_model: FakeChatModel, tools: dict, limits: AgentLimits, audit_service: InMemoryAuditService) -> LangGraphAgent:
    agent_graph = build_agent_graph(
        chat_model, tools, audit_service, checkpointer=MemorySaver(), limits=limits, sleep=_no_sleep
    )
    return LangGraphAgent(
        agent_graph=agent_graph, tools=tools, approval_service=InMemoryApprovalService(), audit_service=audit_service
    )


@pytest.mark.asyncio
async def test_transient_tool_failure_is_retried_and_eventually_succeeds() -> None:
    flaky = _FlakyTool(fail_times=2)
    tools = {"flaky_tool": flaky}
    chat_model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "flaky_tool", "args": {}, "id": "call_1"}]),
            AIMessage(content="recovered"),
        ]
    )
    audit_service = InMemoryAuditService()
    agent = _agent(chat_model, tools, AgentLimits(max_tool_retries=3, tool_timeout_seconds=5), audit_service)

    state = await agent.run(AgentRequest(instruction="use flaky tool", conversation_id="retry-1"))

    assert state.status == AgentRunStatus.COMPLETED
    assert flaky.attempts == 3  # 2 failures + 1 success
    history = await audit_service.get_history("retry-1")
    assert history[0].status == ToolCallStatus.SUCCESS


@pytest.mark.asyncio
async def test_transient_failure_exhausting_retries_is_reported_as_failure() -> None:
    flaky = _FlakyTool(fail_times=99)
    tools = {"flaky_tool": flaky}
    chat_model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "flaky_tool", "args": {}, "id": "call_1"}]),
            AIMessage(content="it kept failing"),
        ]
    )
    audit_service = InMemoryAuditService()
    agent = _agent(chat_model, tools, AgentLimits(max_tool_retries=2, tool_timeout_seconds=5), audit_service)

    state = await agent.run(AgentRequest(instruction="use flaky tool", conversation_id="retry-2"))

    assert state.status == AgentRunStatus.COMPLETED  # the agent still wraps up gracefully
    assert flaky.attempts == 3  # initial attempt + 2 retries, then gave up
    history = await audit_service.get_history("retry-2")
    assert history[0].status == ToolCallStatus.FAILURE


@pytest.mark.asyncio
async def test_permanent_tool_failure_is_not_retried() -> None:
    bad = _AlwaysFailsTool()
    tools = {"bad_tool": bad}
    chat_model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "bad_tool", "args": {}, "id": "call_1"}]),
            AIMessage(content="that didn't work"),
        ]
    )
    audit_service = InMemoryAuditService()
    agent = _agent(chat_model, tools, AgentLimits(max_tool_retries=5, tool_timeout_seconds=5), audit_service)

    await agent.run(AgentRequest(instruction="use bad tool", conversation_id="retry-3"))

    assert bad.attempts == 1  # no retries on a permanent (ValueError) error


@pytest.mark.asyncio
async def test_max_steps_limit_stops_a_looping_agent() -> None:
    # Always issues another tool call -- would loop forever without the limit.
    responses = [
        AIMessage(content="", tool_calls=[{"name": "list_labels", "args": {}, "id": f"call_{i}"}]) for i in range(10)
    ]
    chat_model = FakeChatModel(responses)
    tools = build_tools(FakeGmailClient())
    audit_service = InMemoryAuditService()
    limits = AgentLimits(max_steps=3, max_tool_calls=100, max_tool_retries=0, tool_timeout_seconds=5)
    agent = _agent(chat_model, tools, limits, audit_service)

    state = await agent.run(AgentRequest(instruction="loop forever", conversation_id="loop-1"))

    assert state.status == AgentRunStatus.FAILED
    assert "max_steps" in state.final_response
    history = await audit_service.get_history("loop-1")
    assert any(r.tool_name == "__execution_limit__" for r in history)


@pytest.mark.asyncio
async def test_max_tool_calls_limit_skips_excess_calls_within_one_turn() -> None:
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "list_labels", "args": {}, "id": "call_1"},
                {"name": "list_labels", "args": {}, "id": "call_2"},
            ],
        ),
        AIMessage(content="done"),
    ]
    chat_model = FakeChatModel(responses)
    tools = build_tools(FakeGmailClient())
    audit_service = InMemoryAuditService()
    limits = AgentLimits(max_steps=10, max_tool_calls=1, max_tool_retries=0, tool_timeout_seconds=5)
    agent = _agent(chat_model, tools, limits, audit_service)

    state = await agent.run(AgentRequest(instruction="list labels twice", conversation_id="limit-2"))

    assert state.status == AgentRunStatus.COMPLETED
    assert state.final_response == "done"
    history = await audit_service.get_history("limit-2")
    statuses = {record.step_id: record.status for record in history}
    assert statuses["call_1"] == ToolCallStatus.SUCCESS
    assert statuses["call_2"] == ToolCallStatus.SKIPPED


@pytest.mark.asyncio
async def test_max_execution_seconds_stops_a_slow_run() -> None:
    responses = [
        AIMessage(content="", tool_calls=[{"name": "list_labels", "args": {}, "id": f"call_{i}"}]) for i in range(5)
    ]
    chat_model = FakeChatModel(responses)
    tools = build_tools(FakeGmailClient())
    audit_service = InMemoryAuditService()
    # An impossible time budget: the very first agent turn is already "too slow".
    limits = AgentLimits(max_steps=100, max_execution_seconds=-1, tool_timeout_seconds=5)
    agent = _agent(chat_model, tools, limits, audit_service)

    state = await agent.run(AgentRequest(instruction="anything", conversation_id="timeout-1"))

    assert state.status == AgentRunStatus.FAILED
    assert "max_execution_seconds" in state.final_response


# --- Phase 5: the model call itself is retried, and a final failure ends the run cleanly ---


class _TransientModelError(Exception):
    """Mimics `langchain_google_genai.GoogleAPIError` for a 503: the HTTP code sits on the exception."""

    code = 503


def _list_labels_call(index: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "list_labels", "args": {}, "id": f"call_{index}"}])


@pytest.mark.asyncio
async def test_transient_model_failure_is_retried_and_the_run_completes() -> None:
    chat_model = FakeChatModel([_TransientModelError("503 UNAVAILABLE"), AIMessage(content="all good")])
    tools = build_tools(FakeGmailClient())
    audit_service = InMemoryAuditService()
    agent = _agent(chat_model, tools, AgentLimits(max_llm_retries=2), audit_service)

    state = await agent.run(AgentRequest(instruction="anything", conversation_id="llm-retry-1"))

    assert state.status == AgentRunStatus.COMPLETED
    assert state.final_response == "all good"
    assert len(chat_model.invocations) == 2
    assert await audit_service.get_history("llm-retry-1") == []  # a retried blip leaves no failure record


@pytest.mark.asyncio
async def test_model_failure_after_retries_ends_the_run_cleanly_not_with_an_exception() -> None:
    chat_model = FakeChatModel([_TransientModelError("503 UNAVAILABLE")] * 4)
    tools = build_tools(FakeGmailClient())
    audit_service = InMemoryAuditService()
    agent = _agent(chat_model, tools, AgentLimits(max_llm_retries=1), audit_service)

    state = await agent.run(AgentRequest(instruction="anything", conversation_id="llm-retry-2"))

    assert state.status == AgentRunStatus.FAILED
    assert "language model call failed" in state.final_response
    assert "503 UNAVAILABLE" in state.final_response
    assert len(chat_model.invocations) == 2  # initial attempt + 1 retry
    history = await audit_service.get_history("llm-retry-2")
    assert [record.tool_name for record in history] == ["__llm_error__"]
    assert history[0].status == ToolCallStatus.FAILURE


@pytest.mark.asyncio
async def test_permanent_model_error_is_not_retried() -> None:
    chat_model = FakeChatModel([ValueError("invalid request"), AIMessage(content="never reached")])
    tools = build_tools(FakeGmailClient())
    audit_service = InMemoryAuditService()
    agent = _agent(chat_model, tools, AgentLimits(max_llm_retries=5), audit_service)

    state = await agent.run(AgentRequest(instruction="anything", conversation_id="llm-retry-3"))

    assert state.status == AgentRunStatus.FAILED
    assert len(chat_model.invocations) == 1


@pytest.mark.asyncio
async def test_model_failure_mid_run_keeps_earlier_tool_results_in_the_audit_trail() -> None:
    chat_model = FakeChatModel([_list_labels_call(1), ValueError("boom")])
    tools = build_tools(FakeGmailClient())
    audit_service = InMemoryAuditService()
    agent = _agent(chat_model, tools, AgentLimits(), audit_service)

    state = await agent.run(AgentRequest(instruction="anything", conversation_id="llm-retry-4"))

    assert state.status == AgentRunStatus.FAILED
    history = await audit_service.get_history("llm-retry-4")
    assert [record.tool_name for record in history] == ["list_labels", "__llm_error__"]


def test_limits_from_settings_maps_every_field_by_name() -> None:
    settings = Settings(_env_file=None, agent_max_steps=7, agent_max_llm_retries=5, agent_llm_timeout_seconds=9.5)

    limits = AgentLimits.from_settings(settings)

    assert limits.max_steps == 7
    assert limits.max_llm_retries == 5
    assert limits.llm_timeout_seconds == 9.5
    assert limits.max_tool_calls == settings.agent_max_tool_calls
    # Guards the naming convention itself: every limit must have its setting.
    for item in fields(AgentLimits):
        assert hasattr(settings, f"agent_{item.name}"), item.name
