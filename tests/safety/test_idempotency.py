"""Phase 5.5: an approved action runs at most once, however many approvals reach it."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from mailpilot.agent.graph import build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.safety.idempotency import IdempotencyGuard, idempotency_key
from mailpilot.safety.in_memory_approval import InMemoryApprovalService
from mailpilot.schemas.agent import AgentRequest, AgentRunStatus, ApprovalStatus
from mailpilot.schemas.audit import ToolCallStatus
from mailpilot.schemas.email import EmailMessage
from tests.fakes import FakeChatModel, FakeGmailClient


def test_key_identifies_the_action_regardless_of_argument_order() -> None:
    assert idempotency_key("send_email", {"a": 1, "b": 2}) == idempotency_key("send_email", {"b": 2, "a": 1})
    assert idempotency_key("send_email", {"draft_id": "d1"}) != idempotency_key("send_email", {"draft_id": "d2"})


def test_guard_allows_an_action_once() -> None:
    guard = IdempotencyGuard()

    assert guard.try_begin("k")
    assert not guard.try_begin("k")  # running right now
    guard.complete("k", "sent")
    assert not guard.try_begin("k")  # already done
    assert guard.completed_result("k") == "sent"


def test_abandoned_action_can_be_tried_again() -> None:
    guard = IdempotencyGuard()

    assert guard.try_begin("k")
    guard.abandon("k")

    assert guard.completed_result("k") is None
    assert guard.try_begin("k")


class SlowFailingGmailClient(FakeGmailClient):
    """Sends take a while (so two approvals can overlap); the first `failures` sends raise."""

    def __init__(self, failures: int = 0) -> None:
        super().__init__()
        self._failures = failures

    async def send_email(self, draft_id: str) -> EmailMessage:
        await asyncio.sleep(0.01)
        if self._failures:
            self._failures -= 1
            self.calls.append(("send_email_failed", (draft_id,)))
            raise ConnectionError("connection reset")
        return await super().send_email(draft_id)


def _send(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": call_id}])


async def _two_conversations_awaiting_the_same_send(
    gmail_client: FakeGmailClient,
) -> tuple[LangGraphAgent, InMemoryAuditService]:
    """Conversations c1 and c2 each end AWAITING_APPROVAL for send_email(draft-1)."""
    chat_model = FakeChatModel([_send("call_1"), _send("call_2"), AIMessage(content="Done."), AIMessage(content="Done.")])
    tools = build_tools(gmail_client)
    audit_service = InMemoryAuditService()
    agent = LangGraphAgent(
        agent_graph=build_agent_graph(chat_model, tools, audit_service, checkpointer=MemorySaver()),
        tools=tools,
        approval_service=InMemoryApprovalService(),
        audit_service=audit_service,
    )
    for conversation_id in ("c1", "c2"):
        state = await agent.run(AgentRequest(instruction="send the reply", conversation_id=conversation_id))
        assert state.status is AgentRunStatus.AWAITING_APPROVAL
    return agent, audit_service


async def _send_record(audit_service: InMemoryAuditService, conversation_id: str):
    history = await audit_service.get_history(conversation_id)
    return next(r for r in history if r.tool_name == "send_email" and r.approval_status is ApprovalStatus.APPROVED)


@pytest.mark.asyncio
async def test_same_draft_approved_in_two_conversations_is_sent_once() -> None:
    gmail_client = FakeGmailClient()
    agent, audit_service = await _two_conversations_awaiting_the_same_send(gmail_client)

    first = await agent.resume("c1", approved=True)
    second = await agent.resume("c2", approved=True)

    assert gmail_client.sent_draft_ids == ["draft-1"]
    assert first.status is AgentRunStatus.COMPLETED and second.status is AgentRunStatus.COMPLETED
    assert (await _send_record(audit_service, "c1")).status is ToolCallStatus.SUCCESS
    duplicate = await _send_record(audit_service, "c2")
    assert duplicate.status is ToolCallStatus.SKIPPED
    assert "already ran earlier" in (duplicate.result_summary or "")


@pytest.mark.asyncio
async def test_concurrent_approvals_of_the_same_draft_send_once() -> None:
    gmail_client = SlowFailingGmailClient()
    agent, audit_service = await _two_conversations_awaiting_the_same_send(gmail_client)

    await asyncio.gather(agent.resume("c1", approved=True), agent.resume("c2", approved=True))

    assert gmail_client.sent_draft_ids == ["draft-1"]
    records = {r.status: r for r in [await _send_record(audit_service, c) for c in ("c1", "c2")]}
    assert set(records) == {ToolCallStatus.SUCCESS, ToolCallStatus.SKIPPED}
    assert "already running" in (records[ToolCallStatus.SKIPPED].result_summary or "")


@pytest.mark.asyncio
async def test_a_failed_send_can_be_approved_again() -> None:
    gmail_client = SlowFailingGmailClient(failures=1)
    agent, audit_service = await _two_conversations_awaiting_the_same_send(gmail_client)

    await agent.resume("c1", approved=True)
    await agent.resume("c2", approved=True)

    assert ("send_email_failed", ("draft-1",)) in gmail_client.calls
    assert gmail_client.sent_draft_ids == ["draft-1"]
    assert (await _send_record(audit_service, "c1")).status is ToolCallStatus.FAILURE
    assert (await _send_record(audit_service, "c2")).status is ToolCallStatus.SUCCESS


@pytest.mark.asyncio
async def test_rejecting_does_not_consume_the_action() -> None:
    gmail_client = FakeGmailClient()
    agent, _ = await _two_conversations_awaiting_the_same_send(gmail_client)

    await agent.resume("c1", approved=False)
    await agent.resume("c2", approved=True)

    assert gmail_client.sent_draft_ids == ["draft-1"]
