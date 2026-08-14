"""`RAGService` implementation backed by ChromaDB.

One collection holds both email-thread chunks and knowledge-document
chunks (distinguished by the `source_type` metadata field), so a single
similarity query ranks both kinds of context together -- see
`mailpilot.agent.reply_drafting` for how retrieval results are then
budgeted before going into a prompt.
"""

from __future__ import annotations

from typing import Any

import chromadb

from mailpilot.rag.chunking import chunk_text
from mailpilot.rag.embeddings import EmbeddingFunction
from mailpilot.rag.service import RAGService
from mailpilot.schemas.email import EmailThread
from mailpilot.schemas.rag import RetrievedChunk

_COLLECTION_NAME = "mailpilot_context"


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma only accepts primitive metadata values. Drop `None`s and
    stringify anything else, rather than persisting arbitrary objects
    alongside a chunk."""
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        sanitized[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return sanitized


class ChromaRAGService(RAGService):
    def __init__(
        self,
        persist_dir: str,
        embedding_function: EmbeddingFunction,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self._embedding_function = embedding_function
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            _COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    async def ingest_thread(self, thread: EmailThread) -> None:
        for message in thread.messages:
            text = f"{message.subject}\n\n{message.body_text or message.snippet or ''}".strip()
            chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
            if not chunks:
                continue
            ids = [f"thread:{message.message_id}:{i}" for i in range(len(chunks))]
            metadata = _sanitize_metadata(
                {
                    "source_type": "email_thread",
                    "thread_id": thread.thread_id,
                    "message_id": message.message_id,
                    "sender": message.sender.email,
                    "subject": message.subject,
                    "timestamp": message.received_at.isoformat() if message.received_at else "",
                }
            )
            embeddings = await self._embedding_function.embed_documents(chunks)
            self._collection.upsert(
                ids=ids, documents=chunks, metadatas=[metadata] * len(chunks), embeddings=embeddings
            )

    async def ingest_document(self, document_id: str, text: str, metadata: dict) -> None:
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            return
        ids = [f"document:{document_id}:{i}" for i in range(len(chunks))]
        merged_metadata = _sanitize_metadata({**metadata, "source_type": "document", "document_id": document_id})
        embeddings = await self._embedding_function.embed_documents(chunks)
        self._collection.upsert(
            ids=ids, documents=chunks, metadatas=[merged_metadata] * len(chunks), embeddings=embeddings
        )

    async def query(
        self, query: str, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        count = self._collection.count()
        if count == 0:
            return []

        query_embedding = await self._embedding_function.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
        )

        chunks: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            source_type = str(metadata.get("source_type", "unknown"))
            source_id = str(metadata.get("thread_id") or metadata.get("document_id") or chunk_id)
            # cosine distance -> similarity in [0, 1]; clamp for float noise near 0.
            score = max(0.0, 1.0 - distance)
            chunks.append(
                RetrievedChunk(
                    text=document,
                    score=score,
                    source_id=source_id,
                    source_type=source_type,
                    metadata=dict(metadata),
                )
            )
        return chunks
