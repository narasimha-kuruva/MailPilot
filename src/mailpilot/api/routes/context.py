"""Knowledge-store endpoints: index threads and documents, remove them, see what's stored.

Grounded replies (`draft_grounded_reply`) retrieve from this store. It holds
only what was explicitly indexed here or by the agent's `index_thread` tool.
"""

from __future__ import annotations

from fastapi import APIRouter

from mailpilot.api.deps import GmailClientDep, RAGServiceDep
from mailpilot.rag.indexing import index_threads, thread_ids_for_query
from mailpilot.schemas.rag import (
    ContextStats,
    IndexDocumentRequest,
    IndexedDocument,
    IndexThreadsRequest,
    IndexThreadsResponse,
    RemovedContext,
)

router = APIRouter(prefix="/context", tags=["context"])


@router.get("", response_model=ContextStats)
async def get_context_stats(rag_service: RAGServiceDep) -> ContextStats:
    """How many chunks, threads, and documents the store holds."""
    return await rag_service.stats()


@router.post("/threads", response_model=IndexThreadsResponse)
async def index_threads_route(
    request: IndexThreadsRequest, gmail_client: GmailClientDep, rag_service: RAGServiceDep
) -> IndexThreadsResponse:
    """Index threads by id, or the threads behind a Gmail search. Re-indexing replaces a
    thread's earlier version. A thread that fails is listed under `failed`; the rest still index."""
    if request.query is not None:
        thread_ids = await thread_ids_for_query(gmail_client, request.query, request.max_threads)
    else:
        thread_ids = request.thread_ids or []
    return await index_threads(gmail_client, rag_service, thread_ids)


@router.delete("/threads/{thread_id}", response_model=RemovedContext)
async def remove_thread(thread_id: str, rag_service: RAGServiceDep) -> RemovedContext:
    """Remove a thread from the store (0 chunks removed if it wasn't indexed)."""
    return RemovedContext(source_id=thread_id, chunks_removed=await rag_service.delete_thread(thread_id))


@router.post("/documents", response_model=IndexedDocument)
async def index_document(request: IndexDocumentRequest, rag_service: RAGServiceDep) -> IndexedDocument:
    """Index a document (a policy, a price list, notes). Re-using an id replaces the earlier version."""
    chunks = await rag_service.ingest_document(request.document_id, request.text, request.metadata)
    return IndexedDocument(document_id=request.document_id, chunks=chunks)


@router.delete("/documents/{document_id}", response_model=RemovedContext)
async def remove_document(document_id: str, rag_service: RAGServiceDep) -> RemovedContext:
    return RemovedContext(source_id=document_id, chunks_removed=await rag_service.delete_document(document_id))
