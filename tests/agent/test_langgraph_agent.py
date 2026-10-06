"""End-to-end tests of the LangGraph workflow and `LangGraphAgent`.

Uses `FakeChatModel` in place of a real Gemini model and `FakeGmailClient`
in place of the real Gmail API, so the graph's control flow -- tool
execution, the approval gate, resume-after-decision, and audit logging --
is exercised deterministically with no network access or credentials.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from mailpilot.agent.graph import AgentLimits, build_agent_graph
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
    assert ("search_messages", ("is:unread", 10)) in gmail_client.calls  # the tool's default max_results

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
async def test_approval_description_shows_cc_recipients() -> None:
    """An outsider hidden in Cc behind a plausible To must be visible to the approver."""
    chat_model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_draft",
                        "args": {
                            "to": ["alice@example.com"],
                            "cc": ["attacker@evil.com"],
                            "subject": "Fwd: invoice",
                            "body_text": "As requested.",
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
    agent, _, _ = _build_agent(chat_model)

    state = await agent.run(AgentRequest(instruction="forward the invoice", conversation_id="c-cc"))

    assert "Cc: attacker@evil.com" in state.pending_approval.description


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


def test_message_text_extracts_only_text_from_content_blocks() -> None:
    """Gemini 3.x returns block lists with opaque signatures; the user must only ever see the text."""
    from mailpilot.agent.langgraph_agent import _message_text

    message = AIMessage(
        content=[
            {"type": "text", "text": "Found 2 unread emails.", "extras": {"signature": "OPAQUE-BLOB"}},
            {"type": "text", "text": " Want me to summarise them?"},
        ]
    )

    assert _message_text(message) == "Found 2 unread emails. Want me to summarise them?"
    assert _message_text(AIMessage(content="plain")) == "plain"


@pytest.mark.asyncio
async def test_resume_reports_the_send_even_when_the_follow_up_model_call_fails() -> None:
    """If the wrap-up model call fails after an approved send, the caller must
    still learn the email went out (and must not get a 500 that invites a
    retry of /decision), and the conversation must stay usable afterwards."""
    chat_model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_1"}]),
            ValueError("model exploded"),  # permanent: no retries, no sleep
            AIMessage(content="Still here."),
        ]
    )
    agent, gmail_client, audit_service = _build_agent(chat_model)

    await agent.run(AgentRequest(instruction="send the reply", conversation_id="c-llm-fail"))
    state = await agent.resume("c-llm-fail", approved=True)

    # The run after the decision stopped early -- but the reply leads with
    # what the decision did.
    assert state.status == AgentRunStatus.FAILED
    assert state.final_response.startswith("The approved 'send_email' action was executed successfully.")
    assert "model exploded" in state.final_response
    assert gmail_client.sent_draft_ids == ["draft-1"]  # sent exactly once

    history = await audit_service.get_history("c-llm-fail")
    assert [r.tool_name for r in history][-2:] == ["send_email", "__llm_error__"]

    # The conversation state was still advanced past the approval, so a
    # follow-up instruction works normally.
    follow_up = await agent.run(AgentRequest(instruction="what happened?", conversation_id="c-llm-fail"))
    assert follow_up.status == AgentRunStatus.COMPLETED
    assert follow_up.final_response == "Still here."


@pytest.mark.asyncio
async def test_after_an_approval_the_agent_carries_on_with_follow_up_steps() -> None:
    chat_model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_1"}]),
            AIMessage(content="", tool_calls=[{"name": "apply_label", "args": {"message_id": "msg-1", "label_id": "STARRED"}, "id": "call_2"}]),
            AIMessage(content="Sent, and starred the original."),
        ]
    )
    agent, gmail_client, audit_service = _build_agent(chat_model)

    await agent.run(AgentRequest(instruction="send the reply, then star the email", conversation_id="c-follow"))
    state = await agent.resume("c-follow", approved=True)

    assert state.status == AgentRunStatus.COMPLETED
    assert state.final_response == "Sent, and starred the original."
    assert gmail_client.sent_draft_ids == ["draft-1"]
    assert ("apply_label", ("msg-1", "STARRED")) in gmail_client.calls  # the follow-up really ran
    history = await audit_service.get_history("c-follow")
    assert [r.tool_name for r in history] == ["send_email", "send_email", "apply_label"]


@pytest.mark.asyncio
async def test_a_second_send_after_an_approval_waits_for_its_own_approval() -> None:
    chat_model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_1"}]),
            AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-2"}, "id": "call_2"}]),
            AIMessage(content="Both sent."),
        ]
    )
    agent, gmail_client, _ = _build_agent(chat_model)

    await agent.run(AgentRequest(instruction="send both replies", conversation_id="c-two"))
    after_first = await agent.resume("c-two", approved=True)

    assert after_first.status == AgentRunStatus.AWAITING_APPROVAL
    assert after_first.pending_approval.tool_args == {"draft_id": "draft-2"}
    assert gmail_client.sent_draft_ids == ["draft-1"]  # the second is not sent yet

    final = await agent.resume("c-two", approved=True)

    assert final.status == AgentRunStatus.COMPLETED
    assert gmail_client.sent_draft_ids == ["draft-1", "draft-2"]


@pytest.mark.asyncio
async def test_continuing_after_an_approval_gets_a_fresh_time_budget() -> None:
    """A human may take longer to decide than a run is allowed to last."""
    chat_model = FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "call_1"}]),
            AIMessage(content="Sent."),
        ]
    )
    gmail_client = FakeGmailClient()
    tools = build_tools(gmail_client)
    audit_service = InMemoryAuditService()
    agent = LangGraphAgent(
        agent_graph=build_agent_graph(
            chat_model, tools, audit_service, checkpointer=MemorySaver(), limits=AgentLimits(max_execution_seconds=0.2)
        ),
        tools=tools,
        approval_service=InMemoryApprovalService(),
        audit_service=audit_service,
    )

    await agent.run(AgentRequest(instruction="send the reply", conversation_id="c-slow"))
    await asyncio.sleep(0.3)  # longer than the whole run budget
    state = await agent.resume("c-slow", approved=True)

    assert state.status == AgentRunStatus.COMPLETED
    assert state.final_response == "Sent."


def _send(draft_id: str, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": draft_id}, "id": call_id}])


@pytest.mark.asyncio
async def test_a_decision_only_acts_on_the_approval_it_names() -> None:
    """Approving an old card must never send whatever is pending now."""
    chat_model = FakeChatModel([_send("draft-1", "call_a"), _send("draft-2", "call_b"), AIMessage(content="Sent.")])
    agent, gmail_client, _ = _build_agent(chat_model)

    card_a = await agent.run(AgentRequest(instruction="reply to alice", conversation_id="c-cards"))
    card_b = await agent.run(AgentRequest(instruction="also email mallory", conversation_id="c-cards"))
    assert (card_a.pending_approval.approval_id, card_b.pending_approval.approval_id) == ("call_a", "call_b")

    with pytest.raises(ValueError, match="not pending"):
        await agent.resume("c-cards", approved=True, approval_id="call_a")
    assert gmail_client.sent_draft_ids == []

    state = await agent.resume("c-cards", approved=True, approval_id="call_b")
    assert state.status == AgentRunStatus.COMPLETED
    assert gmail_client.sent_draft_ids == ["draft-2"]


@pytest.mark.asyncio
async def test_a_new_instruction_withdraws_the_pending_action() -> None:
    chat_model = FakeChatModel([_send("draft-1", "call_a"), AIMessage(content="Here is your summary.")])
    agent, gmail_client, audit_service = _build_agent(chat_model)

    await agent.run(AgentRequest(instruction="reply to alice", conversation_id="c-new"))
    state = await agent.run(AgentRequest(instruction="actually, summarize my inbox", conversation_id="c-new"))

    assert state.final_response == "Here is your summary."
    with pytest.raises(ValueError):
        await agent.resume("c-new", approved=True)
    assert gmail_client.sent_draft_ids == []
    # The model was told the send didn't happen, before the new instruction:
    # every tool call it made has an answer.
    seen = chat_model.invocations[1]
    answer = next(m for m in seen if getattr(m, "tool_call_id", None) == "call_a")
    assert "withdrawn" in answer.content
    assert seen.index(answer) < len(seen) - 1 and seen[-1].content == "actually, summarize my inbox"
    withdrawn = (await audit_service.get_history("c-new"))[-1]
    assert (withdrawn.tool_name, withdrawn.approval_status) == ("send_email", ApprovalStatus.CANCELLED)
    assert await agent._approval_service.get_status("c-new", "call_a") is ApprovalStatus.CANCELLED


class _SlowFakeChatModel(FakeChatModel):
    """Takes a while to answer the calls numbered in `slow_calls` (0-based)."""

    def __init__(self, responses: list, slow_calls: set[int]) -> None:
        super().__init__(responses)
        self._slow_calls = slow_calls

    async def ainvoke(self, messages: list) -> object:
        call = self._index
        response = await super().ainvoke(messages)
        if call in self._slow_calls:
            await asyncio.sleep(0.2)
        return response


@pytest.mark.asyncio
async def test_a_run_waits_for_a_decision_in_progress_on_the_same_conversation() -> None:
    """Both continue the same conversation; neither may erase what the other did."""
    chat_model = _SlowFakeChatModel(
        [_send("draft-1", "call_a"), AIMessage(content="Sent."), AIMessage(content="Nothing else is new.")],
        slow_calls={1},  # the model call after the send
    )
    agent, gmail_client, _ = _build_agent(chat_model)
    await agent.run(AgentRequest(instruction="reply to alice", conversation_id="c-race"))

    decision = asyncio.create_task(agent.resume("c-race", approved=True))
    await asyncio.sleep(0.05)  # the email is sent; the decision's run is still going
    follow_up = await agent.run(AgentRequest(instruction="anything else new?", conversation_id="c-race"))
    resumed = await decision

    assert (resumed.final_response, follow_up.final_response) == ("Sent.", "Nothing else is new.")
    assert gmail_client.sent_draft_ids == ["draft-1"]
    snapshot = await agent._agent_graph.graph.aget_state({"configurable": {"thread_id": "c-race"}})
    history = [(type(m).__name__, getattr(m, "tool_call_id", None) or m.content) for m in snapshot.values["messages"]]
    assert history[-5:] == [
        ("AIMessage", ""),  # the send
        ("ToolMessage", "call_a"),  # its result: the conversation remembers the email went out
        ("AIMessage", "Sent."),
        ("HumanMessage", "anything else new?"),
        ("AIMessage", "Nothing else is new."),
    ]
