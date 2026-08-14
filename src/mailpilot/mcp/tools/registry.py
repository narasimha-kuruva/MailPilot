"""Builds the set of MCP tools available to the agent, wired to one GmailClient.

The Phase 3 intelligence/RAG tools (`classify_email`, `summarize_thread`,
`extract_tasks`, `draft_grounded_reply`) need extra dependencies beyond
`GmailClient`. They're added only when those dependencies are supplied, so
callers that only care about the Phase 2 Gmail tools (most existing tests)
don't need to construct an `IntelligenceService`/`RAGService`/chat model
just to call `build_tools`.
"""

from __future__ import annotations

from typing import Any

from mailpilot.gmail.client import GmailClient
from mailpilot.intelligence.service import IntelligenceService
from mailpilot.mcp.base import MCPTool
from mailpilot.mcp.tools.apply_label import ApplyLabelTool
from mailpilot.mcp.tools.classify_email import ClassifyEmailTool
from mailpilot.mcp.tools.create_draft import CreateDraftTool
from mailpilot.mcp.tools.draft_grounded_reply import DraftGroundedReplyTool
from mailpilot.mcp.tools.extract_tasks import ExtractTasksTool
from mailpilot.mcp.tools.list_labels import ListLabelsTool
from mailpilot.mcp.tools.read_email import ReadEmailTool
from mailpilot.mcp.tools.read_thread import ReadThreadTool
from mailpilot.mcp.tools.search_emails import SearchEmailsTool
from mailpilot.mcp.tools.send_email import SendEmailTool
from mailpilot.mcp.tools.summarize_thread import SummarizeThreadTool
from mailpilot.rag.service import RAGService


def build_tools(
    gmail_client: GmailClient,
    *,
    intelligence_service: IntelligenceService | None = None,
    rag_service: RAGService | None = None,
    chat_model: Any = None,
    max_context_chars: int = 6000,
) -> dict[str, MCPTool]:
    tools: list[MCPTool] = [
        SearchEmailsTool(gmail_client),
        ReadEmailTool(gmail_client),
        ReadThreadTool(gmail_client),
        ListLabelsTool(gmail_client),
        ApplyLabelTool(gmail_client),
        CreateDraftTool(gmail_client),
        SendEmailTool(gmail_client),
    ]

    if intelligence_service is not None:
        tools.append(ClassifyEmailTool(gmail_client, intelligence_service))
        tools.append(SummarizeThreadTool(gmail_client, intelligence_service))
        tools.append(ExtractTasksTool(gmail_client, intelligence_service))

    if rag_service is not None and chat_model is not None:
        tools.append(DraftGroundedReplyTool(gmail_client, rag_service, chat_model, max_context_chars))

    return {tool.name: tool for tool in tools}
