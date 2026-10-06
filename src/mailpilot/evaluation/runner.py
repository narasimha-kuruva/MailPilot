"""Runs evaluation scenarios against the real agent (Phase 5.8).

`run_scenario` builds a fresh agent for each scenario -- the production
graph, tools, approval gate, guardrails, and audit trail -- wired to an
`InMemoryGmailClient` holding the scenario's mailbox and, when an embedding
function is given, a throwaway knowledge store preloaded with the
scenario's documents and threads. It runs the instruction, applies the
scenario's approval decision if it stopped for one, and checks every
expectation. The chat model and embeddings are the caller's: stand-ins in
pytest, the configured provider's real ones in `python -m
mailpilot.evaluation`.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from mailpilot.agent.graph import AgentLimits, build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.agent.pending import memory_checkpointer
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.evaluation.expectations import ScenarioOutcome
from mailpilot.evaluation.mailbox import InMemoryGmailClient
from mailpilot.evaluation.scenarios import OWN_EMAIL, USER_LABELS, Scenario, default_mailbox
from mailpilot.intelligence.gemini_service import GeminiIntelligenceService
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.rag.embeddings import EmbeddingFunction
from mailpilot.rag.service import RAGService
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
    embedding_function: EmbeddingFunction | None = None,
    limits: AgentLimits | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ScenarioResult:
    """Run one scenario end to end. Never raises: an exception is a failed result.

    Without `embedding_function` the agent has no knowledge store, so the
    `index_thread` and `draft_grounded_reply` tools are absent.
    """
    started = time.perf_counter()
    # Chroma keeps its files open on Windows, so the cleanup is best effort.
    with tempfile.TemporaryDirectory(prefix="mailpilot-eval-", ignore_cleanup_errors=True) as store_dir:
        try:
            return await _run(scenario, chat_model, embedding_function, store_dir, limits, sleep, started)
        except Exception as exc:  # noqa: BLE001 - one broken scenario must not stop the suite
            return ScenarioResult(
                scenario=scenario,
                passed=False,
                failures=[f"run raised {type(exc).__name__}: {describe_error(exc)}"],
                duration_seconds=time.perf_counter() - started,
            )


async def _run(
    scenario: Scenario,
    chat_model: Any,
    embedding_function: EmbeddingFunction | None,
    store_dir: str,
    limits: AgentLimits | None,
    sleep: Callable[[float], Awaitable[None]] | None,
    started: float,
) -> ScenarioResult:
    mailbox = InMemoryGmailClient(
        default_mailbox(), own_email=OWN_EMAIL, user_labels=USER_LABELS, failures=scenario.failures
    )
    rag_service: RAGService | None = None
    if embedding_function is not None:
        rag_service = ChromaRAGService(persist_dir=store_dir, embedding_function=embedding_function)
        for document in scenario.knowledge_documents:
            await rag_service.ingest_document(document.document_id, document.text, document.metadata)
        for thread_id in scenario.knowledge_threads:
            await rag_service.ingest_thread(await mailbox.get_thread(thread_id))
        mailbox.calls.clear()  # setup isn't part of what the agent did

    tools = build_tools(
        mailbox,
        intelligence_service=GeminiIntelligenceService(chat_model),
        rag_service=rag_service,
        chat_model=chat_model,
        own_email=OWN_EMAIL,
    )
    audit = InMemoryAuditService()
    graph = build_agent_graph(
        chat_model,
        tools,
        audit,
        checkpointer=memory_checkpointer(),
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

    state = await agent.run(AgentRequest(instruction=scenario.instruction, conversation_id=scenario.id))
    sent_before_decision = len(mailbox.sent)
    pending_draft = None
    if state.pending_approval is not None and state.pending_approval.tool_name == "send_email":
        draft = mailbox.drafts.get(str(state.pending_approval.tool_args.get("draft_id", "")))
        pending_draft = draft.message.model_copy(deep=True) if draft else None
    final_state = None
    if state.pending_approval is not None and scenario.decision is not None:
        final_state = await agent.resume(
            scenario.id, scenario.decision, approval_id=state.pending_approval.approval_id
        )

    history = await audit.get_history(scenario.id)
    outcome = ScenarioOutcome(
        state=state,
        final_state=final_state,
        audit=history,
        mailbox=mailbox,
        sent_before_decision=sent_before_decision,
        pending_draft=pending_draft,
        knowledge=await rag_service.stats() if rag_service is not None else None,
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
