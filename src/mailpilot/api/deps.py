"""FastAPI dependency-injection helpers.

Routes depend on these aliases rather than importing concrete services
directly, so implementations (and test doubles, via
`app.dependency_overrides`) can be swapped without touching route code.

Services are constructed lazily (via `lru_cache`, mirroring `get_settings`)
so importing this module, or hitting unrelated routes like `/health`,
never requires Gmail OAuth or a Gemini API key to be configured -- only
actually calling an `/agent/*` route does.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

from mailpilot.agent.base import Agent
from mailpilot.agent.graph import AgentLimits, build_agent_graph
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
from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.rag.embeddings import EmbeddingFunction
from mailpilot.rag.service import RAGService
from mailpilot.safety.approval import ApprovalService
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
def get_audit_service() -> AuditService:
    return InMemoryAuditService()


@lru_cache
def get_approval_service() -> ApprovalService:
    return InMemoryApprovalService()


@lru_cache
def get_chat_model() -> BaseChatModel:
    """The configured provider's chat model (Gemini or local Ollama) -- see `mailpilot.llm`."""
    try:
        return build_chat_model(get_settings())
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


@lru_cache
def get_agent() -> Agent:
    settings = get_settings()
    chat_model = get_chat_model()
    tools = build_tools(
        get_gmail_client(),
        intelligence_service=get_intelligence_service(),
        rag_service=get_rag_service(),
        chat_model=chat_model,
        max_context_chars=settings.rag_max_context_chars,
        own_email=settings.gmail_user_email,
    )
    limits = AgentLimits.from_settings(settings)
    graph = build_agent_graph(
        chat_model, tools, get_audit_service(), checkpointer=MemorySaver(), limits=limits
    )
    return LangGraphAgent(
        agent_graph=graph,
        tools=tools,
        approval_service=get_approval_service(),
        audit_service=get_audit_service(),
        approval_ttl_seconds=settings.approval_ttl_seconds,
        tool_timeout_seconds=settings.agent_tool_timeout_seconds,
        gmail_client=get_gmail_client(),
    )


GmailClientDep = Annotated[GmailClient, Depends(get_gmail_client)]
AuditServiceDep = Annotated[AuditService, Depends(get_audit_service)]
ApprovalServiceDep = Annotated[ApprovalService, Depends(get_approval_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
IntelligenceServiceDep = Annotated[IntelligenceService, Depends(get_intelligence_service)]
AgentDep = Annotated[Agent, Depends(get_agent)]
