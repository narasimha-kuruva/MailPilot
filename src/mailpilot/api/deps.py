"""FastAPI dependency-injection helpers.

Routes depend on these aliases rather than importing concrete services
directly, so implementations (and test doubles, via
`app.dependency_overrides`) can be swapped without touching route code.

Services are constructed lazily (via `lru_cache`, mirroring `get_settings`)
so importing this module, or hitting unrelated routes like `/health`,
never requires Gmail OAuth or a Gemini API key to be configured -- only
actually calling an `/agent/*` route does.

With `STATE_BACKEND=sqlite` (the default) the audit trail, approvals,
pending approvals, completed sends and conversations live in SQLite under
`STATE_DIR` and survive a restart (`mailpilot.persistence.sqlite`); with
`memory` they live in this process only.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from mailpilot.agent.base import Agent
from mailpilot.agent.graph import AgentLimits, build_agent_graph
from mailpilot.agent.pending import memory_checkpointer
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.audit.service import AuditService
from mailpilot.config import Settings, get_settings
from mailpilot.gmail.client import GmailClient
from mailpilot.gmail.google_client import GoogleGmailClient
from mailpilot.intelligence.gemini_service import GeminiIntelligenceService
from mailpilot.intelligence.service import IntelligenceService
from mailpilot.llm.providers import LLMProviderError, build_chat_model, build_embedding_function
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.observability.metrics import LLMUsageCallback, MetricsRegistry
from mailpilot.persistence.sqlite import (
    PersistentState,
    SqliteApprovalService,
    SqliteAuditService,
    SqliteCompletedActions,
    SqlitePendingCallStore,
)
from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.rag.embeddings import EmbeddingFunction
from mailpilot.rag.service import RAGService
from mailpilot.safety.approval import ApprovalService
from mailpilot.safety.idempotency import IdempotencyGuard
from mailpilot.safety.in_memory_approval import InMemoryApprovalService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def _unavailable(exc: LLMProviderError) -> HTTPException:
    # Provider misconfiguration (missing API key, Ollama not running, model
    # not pulled) is an operator problem, not a client error: 503 with the
    # provider's own explanation of how to fix it.
    return HTTPException(status_code=503, detail=str(exc))


@lru_cache
def get_gmail_client() -> GmailClient:
    return GoogleGmailClient(get_settings())


@lru_cache
def get_metrics() -> MetricsRegistry:
    settings = get_settings()
    return MetricsRegistry(
        input_usd_per_million_tokens=settings.llm_input_usd_per_million_tokens,
        output_usd_per_million_tokens=settings.llm_output_usd_per_million_tokens,
    )


@lru_cache
def get_persistent_state() -> PersistentState | None:
    """The SQLite state under STATE_DIR, or None with STATE_BACKEND=memory."""
    settings = get_settings()
    return PersistentState.open(settings.state_dir) if settings.state_backend == "sqlite" else None


@lru_cache
def get_audit_service() -> AuditService:
    state = get_persistent_state()
    if state is not None:
        return SqliteAuditService(state.database, metrics=get_metrics())
    return InMemoryAuditService(metrics=get_metrics())


@lru_cache
def get_approval_service() -> ApprovalService:
    state = get_persistent_state()
    return SqliteApprovalService(state.database) if state is not None else InMemoryApprovalService()


@lru_cache
def get_chat_model() -> BaseChatModel:
    """The configured provider's chat model (Gemini or local Ollama) -- see `mailpilot.llm`."""
    try:
        return build_chat_model(get_settings(), callbacks=[LLMUsageCallback(get_metrics())])
    except LLMProviderError as exc:
        raise _unavailable(exc) from exc


@lru_cache
def get_embedding_function() -> EmbeddingFunction:
    try:
        return build_embedding_function(get_settings())
    except LLMProviderError as exc:
        raise _unavailable(exc) from exc


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    return ChromaRAGService(
        persist_dir=settings.chroma_persist_dir,
        embedding_function=get_embedding_function(),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


@lru_cache
def get_intelligence_service() -> IntelligenceService:
    return GeminiIntelligenceService(get_chat_model())


def _build_agent(checkpointer: BaseCheckpointSaver) -> Agent:
    """Everything the agent needs except the checkpointer. Blocking (Ollama
    readiness check, Chroma on disk), so `get_agent` runs it in a worker thread."""
    settings = get_settings()
    chat_model = get_chat_model()
    tools = build_tools(
        get_gmail_client(),
        intelligence_service=get_intelligence_service(),
        rag_service=get_rag_service(),
        chat_model=chat_model,
        max_context_chars=settings.rag_max_context_chars,
        top_k=settings.rag_top_k,
        own_email=settings.gmail_user_email,
    )
    state = get_persistent_state()
    graph = build_agent_graph(
        chat_model, tools, get_audit_service(), checkpointer=checkpointer, limits=AgentLimits.from_settings(settings)
    )
    return LangGraphAgent(
        agent_graph=graph,
        tools=tools,
        approval_service=get_approval_service(),
        audit_service=get_audit_service(),
        approval_ttl_seconds=settings.approval_ttl_seconds,
        tool_timeout_seconds=settings.agent_tool_timeout_seconds,
        gmail_client=get_gmail_client(),
        metrics=get_metrics(),
        pending_store=SqlitePendingCallStore(state.database) if state is not None else None,
        idempotency_guard=IdempotencyGuard(SqliteCompletedActions(state.database)) if state is not None else None,
    )


_agent: Agent | None = None


async def get_agent() -> Agent:
    """The process's one agent, built on first use.

    Async so it runs on the event loop: the SQLite checkpointer must be
    created there. The rest of the (blocking) setup runs in a worker thread.
    Two first requests racing may both build one; both get the first that
    finished, so there is still only one agent and one pending-approval store.
    """
    global _agent
    if _agent is None:
        state = get_persistent_state()
        checkpointer = state.checkpointer() if state is not None else memory_checkpointer()
        built = await asyncio.to_thread(_build_agent, checkpointer)
        if _agent is None:
            _agent = built
    return _agent


def reset_agent() -> None:
    """Forget the built agent (at shutdown, so a restarted app builds a fresh one)."""
    global _agent
    _agent = None


GmailClientDep = Annotated[GmailClient, Depends(get_gmail_client)]
AuditServiceDep = Annotated[AuditService, Depends(get_audit_service)]
ApprovalServiceDep = Annotated[ApprovalService, Depends(get_approval_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
IntelligenceServiceDep = Annotated[IntelligenceService, Depends(get_intelligence_service)]
AgentDep = Annotated[Agent, Depends(get_agent)]
MetricsDep = Annotated[MetricsRegistry, Depends(get_metrics)]
