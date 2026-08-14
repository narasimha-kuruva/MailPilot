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
from langchain_google_genai import ChatGoogleGenerativeAI
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
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.rag.embeddings import EmbeddingFunction, GeminiEmbeddingFunction
from mailpilot.rag.service import RAGService
from mailpilot.safety.approval import ApprovalService
from mailpilot.safety.in_memory_approval import InMemoryApprovalService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def _require_google_api_key(settings: Settings) -> str:
    if not settings.google_api_key:
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY is not configured. Set it in .env before using the agent endpoints.",
        )
    return settings.google_api_key


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
def get_chat_model() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    api_key = _require_google_api_key(settings)
    return ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=api_key)


@lru_cache
def get_embedding_function() -> EmbeddingFunction:
    settings = get_settings()
    api_key = _require_google_api_key(settings)
    return GeminiEmbeddingFunction(model=settings.rag_embedding_model, google_api_key=api_key)


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
    )
    limits = AgentLimits(
        max_steps=settings.agent_max_steps,
        max_tool_calls=settings.agent_max_tool_calls,
        max_tool_retries=settings.agent_max_tool_retries,
        tool_timeout_seconds=settings.agent_tool_timeout_seconds,
        max_execution_seconds=settings.agent_max_execution_seconds,
        max_output_chars=settings.agent_max_output_chars,
    )
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
