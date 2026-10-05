from __future__ import annotations

from pathlib import Path

import pytest

from mailpilot.agent.reply_drafting import _pack_context, draft_grounded_reply
from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.schemas.email import EmailAddress, EmailMessage, EmailThread
from mailpilot.schemas.intelligence import GroundedDraft
from mailpilot.schemas.rag import RetrievedChunk
from tests.fakes import FakeChatModel, FakeEmbeddingFunction


def _thread() -> EmailThread:
    message = EmailMessage(
        message_id="msg-1",
        thread_id="thread-1",
        subject="Renewal pricing",
        sender=EmailAddress(email="client@example.com"),
        body_text="Can you confirm the renewal price?",
    )
    return EmailThread(thread_id="thread-1", subject="Renewal pricing", messages=[message])


def _rag_service(tmp_path: Path) -> ChromaRAGService:
    return ChromaRAGService(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=FakeEmbeddingFunction(),
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.mark.asyncio
async def test_grounded_reply_marked_grounded_when_facts_match_context(tmp_path: Path) -> None:
    rag_service = _rag_service(tmp_path)
    await rag_service.ingest_document("pricing-doc", "The annual renewal price is $500.", {})

    draft_response = GroundedDraft(subject="Re: Renewal pricing", body_text="The renewal price is $500 per year.")
    chat_model = FakeChatModel([draft_response])

    result = await draft_grounded_reply(
        thread=_thread(), intent="confirm pricing", chat_model=chat_model, rag_service=rag_service
    )

    assert result.grounded is True
    assert result.validation_notes == []
    assert result.source_references  # at least one retrieved chunk was cited


@pytest.mark.asyncio
async def test_grounded_reply_flagged_when_model_invents_a_price(tmp_path: Path) -> None:
    rag_service = _rag_service(tmp_path)
    await rag_service.ingest_document("pricing-doc", "The annual renewal price is $500.", {})

    draft_response = GroundedDraft(subject="Re: Renewal pricing", body_text="The renewal price is $9999 per year.")
    chat_model = FakeChatModel([draft_response])

    result = await draft_grounded_reply(
        thread=_thread(), intent="confirm pricing", chat_model=chat_model, rag_service=rag_service
    )

    assert result.grounded is False
    assert any("$9999" in note for note in result.validation_notes)


@pytest.mark.asyncio
async def test_grounded_reply_works_with_no_retrieved_context(tmp_path: Path) -> None:
    rag_service = _rag_service(tmp_path)  # nothing ingested
    draft_response = GroundedDraft(subject="Re: Renewal pricing", body_text="Let me get back to you on pricing.")
    chat_model = FakeChatModel([draft_response])

    result = await draft_grounded_reply(
        thread=_thread(), intent="confirm pricing", chat_model=chat_model, rag_service=rag_service
    )

    assert result.source_references == []
    assert result.grounded is True  # no invented dates/amounts in this response


def _chunk(source_id: str, score: float, size: int) -> RetrievedChunk:
    return RetrievedChunk(text="x" * size, score=score, source_id=source_id, source_type="document")


def test_pack_context_skips_oversized_chunk_but_keeps_smaller_later_ones() -> None:
    """Budget 100: best chunk (80) fits; next-best (50) does not; the small
    low-scored one (15) must still be packed instead of being abandoned."""
    chunks = [_chunk("low-small", 0.3, 15), _chunk("high-big", 0.9, 80), _chunk("mid-big", 0.6, 50)]

    packed = _pack_context(chunks, max_chars=100)

    assert [c.source_id for c in packed] == ["high-big", "low-small"]


def test_pack_context_always_includes_the_single_best_chunk_even_if_oversized() -> None:
    chunks = [_chunk("huge", 0.9, 500), _chunk("small", 0.5, 10)]

    packed = _pack_context(chunks, max_chars=100)

    assert [c.source_id for c in packed] == ["huge"]


def test_pack_context_respects_budget_in_score_order() -> None:
    chunks = [_chunk("a", 0.9, 40), _chunk("b", 0.8, 40), _chunk("c", 0.7, 40)]

    packed = _pack_context(chunks, max_chars=100)

    assert [c.source_id for c in packed] == ["a", "b"]
