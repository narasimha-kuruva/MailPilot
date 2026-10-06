"""SQLite persistence: each store on its own, then a full restart in the middle of an approval."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from mailpilot.agent.graph import build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.agent.pending import PendingToolCall
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.persistence.sqlite import (
    PersistentState,
    SqliteApprovalService,
    SqliteAuditService,
    SqliteCompletedActions,
    SqliteDatabase,
    SqlitePendingCallStore,
)
from mailpilot.safety.idempotency import IdempotencyGuard
from mailpilot.schemas.agent import AgentRequest, AgentRunStatus, ApprovalStatus
from mailpilot.schemas.audit import AuditRecord, ToolCallStatus
from tests.fakes import FakeChatModel, FakeGmailClient


@pytest.fixture
def db(tmp_path: Path):
    database = SqliteDatabase(tmp_path / "mailpilot.sqlite")
    yield database
    database.close()


# --- The stores ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_records_round_trip_in_order_and_redacted(db: SqliteDatabase) -> None:
    audit = SqliteAuditService(db)
    for tool in ("search_emails", "read_email"):
        await audit.record(
            AuditRecord(
                conversation_id="c1",
                agent_request="find it, password: hunter2",
                tool_name=tool,
                status=ToolCallStatus.SUCCESS,
                duration_ms=12.5,
            )
        )
    await audit.record(AuditRecord(conversation_id="c2", agent_request="x", tool_name="list_labels", status=ToolCallStatus.SUCCESS))

    history = await SqliteAuditService(db).get_history("c1")  # a fresh service: nothing cached in memory

    assert [r.tool_name for r in history] == ["search_emails", "read_email"]
    assert history[0].duration_ms == 12.5
    assert "hunter2" not in history[0].agent_request


@pytest.mark.asyncio
async def test_approval_decisions_are_stored(db: SqliteDatabase) -> None:
    approvals = SqliteApprovalService(db)

    assert await approvals.get_status("c1", "s1") is ApprovalStatus.NOT_REQUIRED
    await approvals.request_approval("c1", "s1", "Send email to: bob@example.com")
    assert await approvals.get_status("c1", "s1") is ApprovalStatus.PENDING
    await approvals.record_decision("c1", "s1", approved=False)
    assert await SqliteApprovalService(db).get_status("c1", "s1") is ApprovalStatus.REJECTED
    await approvals.mark_expired("c1", "s2")
    assert await approvals.get_status("c1", "s2") is ApprovalStatus.EXPIRED


@pytest.mark.asyncio
async def test_a_pending_call_is_taken_exactly_once(db: SqliteDatabase) -> None:
    store = SqlitePendingCallStore(db)
    call = PendingToolCall(tool_call_id="t1", tool_name="send_email", tool_args={"draft_id": "d1"})
    await store.put("c1", call, requested_at=1000.0)

    results = await asyncio.gather(*(SqlitePendingCallStore(db).take("c1") for _ in range(5)))

    assert [r for r in results if r is not None] == [(call, 1000.0)]
    assert await store.take("c1") is None


def test_completed_actions_outlive_the_guard(db: SqliteDatabase) -> None:
    first = IdempotencyGuard(SqliteCompletedActions(db))
    assert first.try_begin("send_email:d1")
    first.complete("send_email:d1", "sent")

    second = IdempotencyGuard(SqliteCompletedActions(db))  # e.g. after a restart

    assert not second.try_begin("send_email:d1")
    assert second.completed_result("send_email:d1") == "sent"
    assert second.try_begin("send_email:d2")


# --- A restart in the middle of an approval -------------------------------------------------


def _send(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": call_id}])


def _agent(state: PersistentState, chat_model: FakeChatModel, gmail: FakeGmailClient) -> tuple[LangGraphAgent, SqliteAuditService]:
    """What api/deps.py builds, minus the services these tests don't need."""
    tools = build_tools(gmail)
    audit = SqliteAuditService(state.database)
    agent = LangGraphAgent(
        agent_graph=build_agent_graph(chat_model, tools, audit, checkpointer=state.checkpointer()),
        tools=tools,
        approval_service=SqliteApprovalService(state.database),
        audit_service=audit,
        gmail_client=gmail,
        pending_store=SqlitePendingCallStore(state.database),
        idempotency_guard=IdempotencyGuard(SqliteCompletedActions(state.database)),
    )
    return agent, audit


@pytest.mark.asyncio
async def test_an_approval_survives_a_restart_and_the_send_still_happens_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gmail = FakeGmailClient()  # Gmail is outside the process: it doesn't restart

    before = PersistentState.open(tmp_path)
    agent, _ = _agent(before, FakeChatModel([_send("call-1")]), gmail)
    state = await agent.run(AgentRequest(instruction="send the reply", conversation_id="c1"))
    assert state.status is AgentRunStatus.AWAITING_APPROVAL
    await before.close()

    # --- restart ---
    after = PersistentState.open(tmp_path)
    chat_model = FakeChatModel([AIMessage(content="Sent."), _send("call-2"), AIMessage(content="Done.")])
    agent, audit = _agent(after, chat_model, gmail)
    with caplog.at_level(logging.WARNING):
        resumed = await agent.resume("c1", approved=True)

    assert resumed.status is AgentRunStatus.COMPLETED
    assert gmail.sent_draft_ids == ["draft-1"]
    # The model saw the conversation from before the restart.
    assert isinstance(chat_model.invocations[0][0], SystemMessage)
    assert "unregistered type" not in caplog.text  # the checkpoint's own types are allow-listed

    history = await audit.get_history("c1")
    assert [(r.tool_name, r.approval_status) for r in history] == [
        ("send_email", ApprovalStatus.PENDING),
        ("send_email", ApprovalStatus.APPROVED),
    ]

    # The same draft proposed again in a new conversation: it already went out.
    second = await agent.run(AgentRequest(instruction="send it again", conversation_id="c2"))
    assert second.status is AgentRunStatus.AWAITING_APPROVAL
    await agent.resume("c2", approved=True)
    assert gmail.sent_draft_ids == ["draft-1"]
    await after.close()


@pytest.mark.asyncio
async def test_an_approval_requested_before_a_restart_still_expires(tmp_path: Path) -> None:
    gmail = FakeGmailClient()
    before = PersistentState.open(tmp_path)
    agent, _ = _agent(before, FakeChatModel([_send("call-1")]), gmail)
    await agent.run(AgentRequest(instruction="send the reply", conversation_id="c1"))
    await before.close()

    after = PersistentState.open(tmp_path)
    tools = build_tools(gmail)
    audit = SqliteAuditService(after.database)
    late = LangGraphAgent(
        agent_graph=build_agent_graph(FakeChatModel([]), tools, audit, checkpointer=after.checkpointer()),
        tools=tools,
        approval_service=SqliteApprovalService(after.database),
        audit_service=audit,
        pending_store=SqlitePendingCallStore(after.database),
        approval_ttl_seconds=60,
        clock=lambda: 10**12,  # far in the future: wall-clock time, so the age survives the restart
    )

    with pytest.raises(ValueError, match="expired"):
        await late.resume("c1", approved=True)
    assert gmail.sent_draft_ids == []
    await after.close()
