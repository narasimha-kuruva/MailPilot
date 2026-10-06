"""Putting email threads into the knowledge store -- shared by the context API and the agent.

Nothing is indexed implicitly: reading mail never stores it. Content enters
the store only through `POST /api/v1/context/...` or the agent's
`index_thread` tool, both of which call into this module, so what's in the
store is always something a person or the agent explicitly chose to keep.
`find_threads` only lists what a search matches, so a person can choose.
"""

from __future__ import annotations

from mailpilot.gmail.client import GmailClient
from mailpilot.rag.service import RAGService
from mailpilot.resilience import describe_error
from mailpilot.schemas.email import EmailMessage
from mailpilot.schemas.rag import IndexedThread, IndexFailure, IndexThreadsResponse, ThreadCandidate


async def index_thread(gmail_client: GmailClient, rag_service: RAGService, thread_id: str) -> IndexedThread:
    """Fetch a thread and (re)index it. Re-indexing replaces the thread's earlier chunks."""
    thread = await gmail_client.get_thread(thread_id)
    replaced = (await rag_service.indexed_threads([thread.thread_id])).get(thread.thread_id, 0)
    chunks = await rag_service.ingest_thread(thread)
    return IndexedThread(
        thread_id=thread.thread_id,
        subject=thread.subject,
        messages=len(thread.messages),
        chunks=chunks,
        replaced_chunks=replaced,
    )


async def _newest_message_per_thread(gmail_client: GmailClient, query: str, max_threads: int) -> list[EmailMessage]:
    """The newest message of each distinct thread among a Gmail search's newest matches, newest first.

    Searches for `max_threads` messages, so fewer threads come back when
    several matches share one.
    """
    newest: dict[str, EmailMessage] = {}
    for message in await gmail_client.search_messages(query, max_results=max_threads):
        newest.setdefault(message.thread_id, message)
    return list(newest.values())


async def thread_ids_for_query(gmail_client: GmailClient, query: str, max_threads: int) -> list[str]:
    """The distinct threads behind a Gmail search, newest first (see `_newest_message_per_thread`)."""
    return [message.thread_id for message in await _newest_message_per_thread(gmail_client, query, max_threads)]


async def find_threads(
    gmail_client: GmailClient, rag_service: RAGService, query: str, max_threads: int
) -> list[ThreadCandidate]:
    """The threads behind a Gmail search, each marked with what the store already holds for it."""
    messages = await _newest_message_per_thread(gmail_client, query, max_threads)
    indexed = await rag_service.indexed_threads([message.thread_id for message in messages])
    return [
        ThreadCandidate(
            thread_id=message.thread_id,
            subject=message.subject,
            sender=message.sender,
            received_at=message.received_at,
            snippet=message.snippet,
            indexed_chunks=indexed.get(message.thread_id, 0),
        )
        for message in messages
    ]


async def index_threads(gmail_client: GmailClient, rag_service: RAGService, thread_ids: list[str]) -> IndexThreadsResponse:
    """Index each thread in turn. One that can't be fetched or embedded is reported, not fatal."""
    response = IndexThreadsResponse()
    for thread_id in dict.fromkeys(thread_ids):
        try:
            response.indexed.append(await index_thread(gmail_client, rag_service, thread_id))
        except Exception as exc:  # noqa: BLE001 - reported per thread in the response
            response.failed.append(IndexFailure(thread_id=thread_id, error=describe_error(exc)))
    return response
