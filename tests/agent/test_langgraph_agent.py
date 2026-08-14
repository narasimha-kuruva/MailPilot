"""End-to-end tests of the LangGraph workflow and `LangGraphAgent`.

Uses `FakeChatModel` in place of a real Gemini model and `FakeGmailClient`
in place of the real Gmail API, so the graph's control flow -- tool
execution, the approval gate, resume-after-decision, and audit logging --
is exercised deterministically with no network access or credentials.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from mailpilot.agent.graph import build_agent_graph
from mailpilot.agent.langgraph_agent import ApprovalExpiredError, LangGraphAgent
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.safety.in_memory_approval import InMemoryApprovalService
from mailpilot.schemas.agent import AgentRequest, AgentRunStatus, ApprovalStatus, PlanOutline
from mailpilot.schemas.audit import ToolCallStatus
from tests.fakes import FakeChatModel, FakeGmailClient


def _build_agent(chat_model: FakeChatModel) -> tuple[LangGraphAgent, FakeGmailClient, InMemoryAuditService]:
    gmail_client = FakeGmailClient()
    tools = build_tools(gmail_client)
    audit_service = InMemoryAuditService()
    approval_service = InMemoryApprovalService()
    agent_graph = build_agent_graph(chat_model, tools, audit_service, checkpointer=MemorySaver())
    agent = LangGraphAgent(
        agent_graph=agent_graph,
        tools=tools,
        approval_service=approval_service,
        audit_service=audit_service,
        gmail_client=gmail_client,
    )
    return agent, gmail_client, audit_service


@pytest.mark.asyncio
async def test_run_executes_non_sensitive_tool_and_completes() -> None:
    chat_model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_emails", "args": {"query": "is:unread"}, "id": "call_1"}],
            ),
            AIMessage(content="You have 1 unread email from Alice."),
        ]
    )
    agent, gmail_client, audit_service = _build_agent(chat_model)

    state = await agent.run(AgentRequest(instruction="check my unread emails", conversation_id="c1"))

    assert state.status == AgentRunStatus.COMPLETED
    assert state.final_response == "You have 1 unread email from Alice."
    assert ("search_messages", ("is:unread", 25)) in gmail_client.calls

    history = await audit_service.get_history("c1")
    assert len(history) == 1
    assert history[0].tool_name == "search_emails"
    assert history[0].status == ToolCallStatus.SUCCESS
    assert history[0].approval_status == ApprovalStatus.NOT_REQUIRED


@pytest.mark.asyncio
async def test_run_stops_before_send_email_and_does_not_send() -> None:
    chat_model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_1"}],
            )
        ]
    )
    agent, gmail_client, audit_service = _build_agent(chat_model)

    state = await agent.run(AgentRequest(instruction="send the reply", conversation_id="c2"))

    assert state.status == AgentRunStatus.AWAITING_APPROVAL
    assert state.pending_approval is not None
    assert state.pending_approval.tool_name == "send_email"
    assert gmail_client.sent_draft_ids == []  # nothing was sent

    history = await audit_service.get_history("c2")
    assert history[-1].approval_status == ApprovalStatus.PENDING
    assert history[-1].status == ToolCallStatus.SKIPPED


@pytest.mark.asyncio
async def test_approval_description_shows_real_draft_content_for_send_email() -> None:
    """A human approving send_email must see the real recipient/subject/body,
    not just {'draft_id': '...'} -- otherwise approval is a rubber stamp."""
    chat_model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_draft",
                        "args": {
                            "to": ["alice@example.com"],
                            "subject": "Re: Project update",
                            "body_text": "Sounds good, thanks!",
                        },
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_2"}],
            ),
        ]
    )
    agent, gmail_client, _ = _build_agent(chat_model)

    state = await agent.run(AgentRequest(instruction="reply to alice", conversation_id="c-enrich"))

    assert state.status == AgentRunStatus.AWAITING_APPROVAL
    assert "alice@example.com" in state.pending_approval.description
    assert "Re: Project update" in state.pending_approval.description
    assert "Sounds good, thanks!" in state.pending_approval.description


@pytest.mark.asyncio
async def test_approval_description_falls_back_gracefully_when_draft_lookup_fails() -> None:
    chat_model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "send_email", "args": {"draft_id": "no-such-draft"}, "id": "call_1"}],
            )
        ]
    )
    agent, _, _ = _build_agent(chat_model)

    state = await agent.run(AgentRequest(instruction="send it", conversation_id="c-fallback"))

    assert state.status == AgentRunStatus.AWAITING_APPROVAL
    assert "send_email" in state.pending_approval.description  # generic fallback, not a crash


@pytest.mark.asyncio
async def test_resume_with_approval_sends_and_completes() -> None:
    chat_model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_1"}],
            ),
            AIMessage(content="The email has been sent."),
        ]
    )
    agent, gmail_client, audit_service = _build_agent(chat_model)

    await agent.run(AgentRequest(instruction="send the reply", conversation_id="c3"))
    state = await agent.resume("c3", approved=True)

    assert state.status == AgentRunStatus.COMPLETED
    assert state.final_response == "The email has been sent."
    assert gmail_client.sent_draft_ids == ["draft-1"]

    history = await audit_service.get_history("c3")
    approved_record = next(r for r in history if r.approval_status == ApprovalStatus.APPROVED)
    assert approved_record.tool_name == "send_email"
    assert approved_record.status == ToolCallStatus.SUCCESS


@pytest.mark.asyncio
async def test_resume_with_rejection_never_sends() -> None:
    chat_model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_1"}],
            ),
            AIMessage(content="Okay, I did not send that email."),
        ]
    )
    agent, gmail_client, audit_service = _build_agent(chat_model)

    await agent.run(AgentRequest(instruction="send the reply", conversation_id="c4"))
    state = await agent.resume("c4", approved=False)

    assert state.status == AgentRunStatus.COMPLETED
    assert gmail_client.sent_draft_ids == []  # rejection must never send

    history = await audit_service.get_history("c4")
    rejected_record = next(r for r in history if r.approval_status == ApprovalStatus.REJECTED)
    assert rejected_record.tool_name == "send_email"
    assert rejected_record.status == ToolCallStatus.SKIPPED


@pytest.mark.asyncio
async def test_resume_without_pending_approval_raises() -> None:
    agent, _, _ = _build_agent(FakeChatModel([]))

    with pytest.raises(ValueError):
        await agent.resume("no-such-conversation", approved=True)


@pytest.mark.asyncio
async def test_resume_after_ttl_expires_raises_and_never_sends() -> None:
    """Phase 4.6: a pending approval that sits too long must expire rather
    than staying valid forever, and expiry must never let a send through."""
    chat_model = FakeChatModel(
        [AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_1"}])]
    )
    gmail_client = FakeGmailClient()
    tools = build_tools(gmail_client)
    audit_service = InMemoryAuditService()
    agent_graph = build_agent_graph(chat_model, tools, audit_service, checkpointer=MemorySaver())

    fake_clock = {"now": 1_000.0}
    agent = LangGraphAgent(
        agent_graph=agent_graph,
        tools=tools,
        approval_service=InMemoryApprovalService(),
        audit_service=audit_service,
        approval_ttl_seconds=60.0,
        clock=lambda: fake_clock["now"],
    )

    await agent.run(AgentRequest(instruction="send it", conversation_id="ttl-1"))
    fake_clock["now"] += 61.0  # advance past the TTL

    with pytest.raises(ApprovalExpiredError):
        await agent.resume("ttl-1", approved=True)

    assert gmail_client.sent_draft_ids == []
    history = await audit_service.get_history("ttl-1")
    assert any(record.approval_status == ApprovalStatus.EXPIRED for record in history)


@pytest.mark.asyncio
async def test_plan_decomposes_the_goal_without_executing_anything() -> None:
    outline = PlanOutline(steps=["Search the inbox for emails about x", "Summarize what's found"])
    chat_model = FakeChatModel([outline])
    agent, gmail_client, _ = _build_agent(chat_model)

    plan = await agent.plan(AgentRequest(instruction="find emails about x"))

    assert [step.description for step in plan.steps] == outline.steps
    assert gmail_client.calls == []  # plan() must not execute anything
