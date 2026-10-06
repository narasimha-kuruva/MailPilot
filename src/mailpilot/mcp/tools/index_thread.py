"""`index_thread` MCP tool: save a thread to the knowledge store for later grounded replies."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.rag.indexing import index_thread
from mailpilot.rag.service import RAGService
from mailpilot.schemas.rag import IndexedThread


class IndexThreadArgs(BaseModel):
    thread_id: str = Field(..., min_length=1, description="Gmail thread ID, as returned by search_emails.")


class IndexThreadTool(MCPTool):
    name = "index_thread"
    description = (
        "Save a thread to MailPilot's knowledge store, so later grounded replies "
        "(draft_grounded_reply) can draw on it as context. Stores the thread's text locally; "
        "it doesn't change the mailbox. Only index threads the user wants remembered."
    )
    args_schema = IndexThreadArgs

    def __init__(self, gmail_client: GmailClient, rag_service: RAGService) -> None:
        self._gmail_client = gmail_client
        self._rag_service = rag_service

    async def run(self, **kwargs: Any) -> IndexedThread:
        args = self.args_schema.model_validate(kwargs)
        return await index_thread(self._gmail_client, self._rag_service, args.thread_id)
