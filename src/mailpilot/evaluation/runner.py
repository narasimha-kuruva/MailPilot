"""Runs evaluation scenarios against the real agent (Phase 5.8).

`run_scenario` builds a fresh agent for each scenario -- the production
graph, tools, approval gate, guardrails, and audit trail -- wired to an
`InMemoryGmailClient` holding the scenario's mailbox, runs the instruction,
applies the scenario's approval decision if it stopped for one, and checks
every expectation. The chat model is the caller's: a scripted stand-in in
pytest, the configured provider's real model in `python -m
mailpilot.evaluation`.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from mailpilot.agent.graph import AgentLimits, build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.evaluation.expectations import ScenarioOutcome
from mailpilot.evaluation.mailbox import InMemoryGmailClient
from mailpilot.evaluation.scenarios import OWN_EMAIL, USER_LABELS, Scenario, default_mailbox
from mailpilot.intelligence.gemini_service import GeminiIntelligenceService
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.resilience import describe_error
from mailpilot.safety.in_memory_approval import InMemoryApprovalService
from mailpilot.schemas.agent import AgentRequest, AgentRunStatus


@dataclass
class ScenarioResult:
    scenario: Scenario
    passed: bool
    failures: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    tool_sequence: list[str] = field(default_factory=list)
    final_response: str | None = None


async def run_scenario(
    scenario: Scenario,
    chat_model: Any,
    *,
    limits: AgentLimits | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ScenarioResult:
    """Run one scenario end to end. Never raises: an exception is a failed result."""
    started = time.perf_counter()
    mailbox = InMemoryGmailClient(
        default_mailbox(), own_email=OWN_EMAIL, user_labels=USER_LABELS, failures=scenario.failures
    )
    tools = build_tools(mailbox, intelligence_service=GeminiIntelligenceService(chat_model))
    audit = InMemoryAuditService()
    graph = build_agent_graph(
        chat_model,
        tools,
        audit,
        checkpointer=MemorySaver(),
        limits=replace(limits or AgentLimits(), **scenario.limits),
        sleep=sleep,
    )
    agent = LangGraphAgent(
        agent_graph=graph,
        tools=tools,
        approval_service=InMemoryApprovalService(),
        audit_service=audit,
        gmail_client=mailbox,
    )

    try:
        state = await agent.run(AgentRequest(instruction=scenario.instruction, conversation_id=scenario.id))
        sent_before_decision = len(mailbox.sent)
        pending_draft = None
        if state.pending_approval is not None and state.pending_approval.tool_name == "send_email":
            draft = mailbox.drafts.get(str(state.pending_approval.tool_args.get("draft_id", "")))
            pending_draft = draft.message.model_copy(deep=True) if draft else None
        final_state = None
        if state.status is AgentRunStatus.AWAITING_APPROVAL and scenario.decision is not None:
            final_state = await agent.resume(scenario.id, scenario.decision)
    except Exception as exc:  # noqa: BLE001 - one broken scenario must not stop the suite
        return ScenarioResult(
            scenario=scenario,
            passed=False,
            failures=[f"run raised {type(exc).__name__}: {describe_error(exc)}"],
            duration_seconds=time.perf_counter() - started,
        )

    history = await audit.get_history(scenario.id)
    outcome = ScenarioOutcome(
        state=state,
        final_state=final_state,
        audit=history,
        mailbox=mailbox,
        sent_before_decision=sent_before_decision,
        pending_draft=pending_draft,
    )
    failures = [message for expectation in scenario.expectations if (message := expectation.check(outcome))]
    return ScenarioResult(
        scenario=scenario,
        passed=not failures,
        failures=failures,
        duration_seconds=time.perf_counter() - started,
        tool_sequence=[f"{r.tool_name}:{r.status}" for r in history],
        final_response=(final_state or state).final_response,
    )
