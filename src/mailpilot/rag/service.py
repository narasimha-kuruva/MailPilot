"""RAG service interface.

The concrete implementation (Phase 4+) will chunk and embed email threads
and documents, store vectors in ChromaDB, and answer similarity queries
with cited sources via `RetrievedChunk`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mailpilot.schemas.email import EmailThread
from mailpilot.schemas.rag import RetrievedChunk


class RAGService(ABC):
    """Interface for context ingestion and retrieval."""

    @abstractmethod
    async def ingest_thread(self, thread: EmailThread) -> None:
        raise NotImplementedError

    @abstractmethod
    async def ingest_document(
        self, document_id: str, text: str, metadata: dict
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def query(
        self, query: str, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Similarity search, optionally filtered by metadata (e.g. `{"source_type": "document"}`)."""
        raise NotImplementedError
