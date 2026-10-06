"""Grounded reply-drafting workflow (Phase 3.4).

    read thread -> retrieve context -> generate reply -> validate -> (caller creates the Gmail draft)

Context is retrieved via RAG, packed into the prompt by relevance (most
similar first) up to a character budget rather than dumping everything
retrieved, and the resulting draft is checked with
`mailpilot.safety.guardrails.find_unsupported_claims` before being handed
back. This function never calls Gmail -- see
`mailpilot.mcp.tools.draft_grounded_reply` for the tool that wraps this
with `create_draft` and the recipient guardrail.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mailpilot.prompts import GROUNDING_NOTICE, UNTRUSTED_CONTENT_NOTICE, wrap_untrusted
from mailpilot.rag.service import RAGService
from mailpilot.safety.guardrails import find_unsupported_claims
from mailpilot.schemas.email import EmailThread
from mailpilot.schemas.intelligence import GroundedDraft
from mailpilot.schemas.rag import RetrievedChunk


def _render_thread(thread: EmailThread) -> str:
    parts = []
    for message in thread.messages:
        body = message.body_text or message.snippet or ""
        parts.append(f"From: {message.sender.email}\nSubject: {message.subject}\n\n{body}")
    return "\n\n---\n\n".join(parts)


def exclude_thread(thread_id: str) -> dict[str, Any]:
    """Metadata filter: everything except one thread's chunks (documents included)."""
    return {"$or": [{"source_type": {"$ne": "email_thread"}}, {"thread_id": {"$ne": thread_id}}]}


def _pack_context(chunks: list[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    """Greedily keep the highest-scored chunks that fit in the character budget.

    This is the "prioritize relevant context rather than blindly inserting
    everything retrieved" behavior called for in Phase 3.3. Chunks are
    visited best-score first; one that doesn't fit is *skipped*, not
    treated as the end of the list, so a smaller lower-scored chunk after
    it can still be used. The single best chunk is always included even if
    it alone exceeds the budget -- some context beats none.
    """
    packed: list[RetrievedChunk] = []
    used = 0
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        if used + len(chunk.text) > max_chars and packed:
            continue
        packed.append(chunk)
        used += len(chunk.text)
    return packed


async def draft_grounded_reply(
    *,
    thread: EmailThread,
    intent: str,
    chat_model: Any,
    rag_service: RAGService,
    top_k: int = 5,
    max_context_chars: int = 6000,
) -> GroundedDraft:
    latest = thread.messages[-1] if thread.messages else None
    query_text = f"{thread.subject}\n{intent}\n{latest.body_text or latest.snippet if latest else ''}"

    # Context from *other* threads and documents: the thread being replied to
    # is already in the prompt in full, and its own chunks -- the most similar
    # ones in the store, if it was indexed -- would crowd everything else out.
    retrieved = await rag_service.query(query_text, top_k=top_k, where=exclude_thread(thread.thread_id))
    context_chunks = _pack_context(retrieved, max_context_chars)

    context_block = "\n\n".join(
        f"[source: {chunk.source_id}] {chunk.text}" for chunk in context_chunks
    ) or "(no relevant historical context found)"

    structured = chat_model.with_structured_output(GroundedDraft)
    prompt = [
        SystemMessage(
            content=(
                "You draft a reply to an email thread on the user's behalf. "
                f"The user's intent for this reply: {intent}\n\n"
                + UNTRUSTED_CONTENT_NOTICE
                + " "
                + GROUNDING_NOTICE
                + " List the source IDs you actually relied on in `source_references`."
            )
        ),
        HumanMessage(
            content=(
                wrap_untrusted("thread", f"thread_id: {thread.thread_id}\n\n{_render_thread(thread)}")
                + "\n\n"
                + wrap_untrusted("retrieved_context", context_block)
            )
        ),
    ]

    draft = await structured.ainvoke(prompt)

    source_texts = [_render_thread(thread), *(chunk.text for chunk in context_chunks)]
    notes = find_unsupported_claims(draft.body_text, source_texts)
    draft.validation_notes = notes
    draft.grounded = not notes
    draft.source_references = [chunk.source_id for chunk in context_chunks]

    return draft
