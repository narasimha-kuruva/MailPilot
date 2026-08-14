from __future__ import annotations

from pathlib import Path

import pytest

from mailpilot.agent.reply_drafting import draft_grounded_reply
from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.schemas.email import EmailAddress, EmailMessage, EmailThread
from mailpilot.schemas.intelligence import GroundedDraft
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
