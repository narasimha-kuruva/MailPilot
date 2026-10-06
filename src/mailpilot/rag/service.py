"""RAG service interface.

Chunks and embeds email threads and documents, stores the vectors, and
answers similarity queries with cited sources via `RetrievedChunk`. Content
gets in only when someone asks for it -- the context API or the agent's
`index_thread` tool (see `mailpilot.rag.indexing`) -- never as a side effect
of reading mail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mailpilot.schemas.email import EmailThread
from mailpilot.schemas.rag import ContextStats, RetrievedChunk


class RAGService(ABC):
    """Interface for context ingestion and retrieval."""

    @abstractmethod
    async def ingest_thread(self, thread: EmailThread) -> int:
        """Index `thread`, replacing whatever was indexed for it before. Returns the chunk count."""
        raise NotImplementedError

    @abstractmethod
    async def ingest_document(self, document_id: str, text: str, metadata: dict) -> int:
        """Index a document, replacing any earlier version with the same id. Returns the chunk count."""
        raise NotImplementedError

    @abstractmethod
    async def query(
        self, query: str, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Similarity search, optionally filtered by metadata (e.g. `{"source_type": "document"}`)."""
        raise NotImplementedError

    @abstractmethod
    async def delete_thread(self, thread_id: str) -> int:
        """Remove a thread from the store. Returns how many chunks were removed (0 if it wasn't indexed)."""
        raise NotImplementedError

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    async def stats(self) -> ContextStats:
        raise NotImplementedError
