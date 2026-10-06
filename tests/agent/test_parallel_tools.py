"""One model turn's tool calls run together -- bounded, in a deterministic order for limits and results."""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from mailpilot.agent.graph import AgentLimits, build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.mcp.base import MCPTool
from mailpilot.safety.in_memory_approval import InMemoryApprovalService
from mailpilot.schemas.agent import AgentRequest
from mailpilot.schemas.audit import ToolCallStatus
from tests.fakes import FakeChatModel


class _Args(BaseModel):
    seconds: float


class _SlowTool(MCPTool):
    name = "wait"
    description = "waits"
    args_schema = _Args

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    async def run(self, **kwargs: object) -> dict[str, float]:
        args = _Args.model_validate(kwargs)
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(args.seconds)
        self.running -= 1
        return {"waited": args.seconds}


def _turn(*seconds: float) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "wait", "args": {"seconds": s}, "id": f"call-{i}"} for i, s in enumerate(seconds)],
    )


async def _run(seconds: tuple[float, ...], limits: AgentLimits) -> tuple[_SlowTool, FakeChatModel, InMemoryAuditService, float]:
    tool = _SlowTool()
    chat_model = FakeChatModel([_turn(*seconds), AIMessage(content="done")])
    audit = InMemoryAuditService()
    agent = LangGraphAgent(
        agent_graph=build_agent_graph(chat_model, {"wait": tool}, audit, checkpointer=MemorySaver(), limits=limits),
        tools={"wait": tool},
        approval_service=InMemoryApprovalService(),
        audit_service=audit,
    )
    started = time.perf_counter()
    await agent.run(AgentRequest(instruction="wait", conversation_id="c1"))
    return tool, chat_model, audit, time.perf_counter() - started


@pytest.mark.asyncio
async def test_calls_in_one_turn_run_together() -> None:
    tool, _, _, elapsed = await _run((0.2, 0.2, 0.2), AgentLimits())

    assert tool.peak == 3
    assert elapsed < 0.45  # one wait, not three


@pytest.mark.asyncio
async def test_concurrency_is_capped() -> None:
    tool, _, _, _ = await _run((0.05,) * 5, AgentLimits(max_parallel_tool_calls=2))

    assert tool.peak == 2


@pytest.mark.asyncio
async def test_results_and_audit_keep_call_order_when_later_calls_finish_first() -> None:
    _, chat_model, audit, _ = await _run((0.2, 0.01, 0.1), AgentLimits())

    tool_messages = [m for m in chat_model.invocations[1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["call-0", "call-1", "call-2"]
    assert [r.step_id for r in await audit.get_history("c1")] == ["call-0", "call-1", "call-2"]


@pytest.mark.asyncio
async def test_the_tool_call_limit_cuts_off_the_same_calls_however_they_finish() -> None:
    """The first call is the slowest; the limit still applies in call order, not finishing order."""
    tool, _, audit, _ = await _run((0.2, 0.01, 0.01), AgentLimits(max_tool_calls=2))

    statuses = [r.status for r in await audit.get_history("c1")]
    assert statuses == [ToolCallStatus.SUCCESS, ToolCallStatus.SUCCESS, ToolCallStatus.SKIPPED]
    assert tool.peak == 2
