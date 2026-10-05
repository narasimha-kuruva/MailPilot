from __future__ import annotations

from pathlib import Path

import pytest

from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.schemas.email import EmailAddress, EmailMessage, EmailThread
from tests.fakes import FakeEmbeddingFunction


def _service(tmp_path: Path) -> ChromaRAGService:
    return ChromaRAGService(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=FakeEmbeddingFunction(),
        chunk_size=200,
        chunk_overlap=20,
    )


def _thread(thread_id: str, body: str) -> EmailThread:
    message = EmailMessage(
        message_id=f"{thread_id}-msg-1",
        thread_id=thread_id,
        subject="Contract renewal",
        sender=EmailAddress(email="alice@example.com"),
        body_text=body,
    )
    return EmailThread(thread_id=thread_id, subject=message.subject, messages=[message])


@pytest.mark.asyncio
async def test_query_before_ingestion_returns_empty(tmp_path: Path) -> None:
    service = _service(tmp_path)

    results = await service.query("anything")

    assert results == []


@pytest.mark.asyncio
async def test_ingest_thread_and_query_returns_chunk_with_metadata(tmp_path: Path) -> None:
    service = _service(tmp_path)
    await service.ingest_thread(_thread("thread-1", "The contract renewal price is $500 per year."))

    results = await service.query("contract renewal price")

    assert len(results) == 1
    chunk = results[0]
    assert chunk.source_type == "email_thread"
    assert chunk.source_id == "thread-1"
    assert chunk.metadata["message_id"] == "thread-1-msg-1"
    assert chunk.metadata["sender"] == "alice@example.com"
    assert 0.0 <= chunk.score <= 1.0 + 1e-6


@pytest.mark.asyncio
async def test_ingest_document_and_query_returns_document_metadata(tmp_path: Path) -> None:
    service = _service(tmp_path)
    await service.ingest_document("doc-1", "Our refund policy allows returns within 30 days.", {"title": "Refund Policy"})

    results = await service.query("refund policy")

    assert len(results) == 1
    assert results[0].source_type == "document"
    assert results[0].source_id == "doc-1"
    assert results[0].metadata["title"] == "Refund Policy"


@pytest.mark.asyncio
async def test_query_with_where_filter_restricts_source_type(tmp_path: Path) -> None:
    service = _service(tmp_path)
    await service.ingest_thread(_thread("thread-1", "Meeting notes about the renewal."))
    await service.ingest_document("doc-1", "Meeting notes policy document.", {})

    thread_only = await service.query("meeting notes", top_k=5, where={"source_type": "email_thread"})
    doc_only = await service.query("meeting notes", top_k=5, where={"source_type": "document"})

    assert all(chunk.source_type == "email_thread" for chunk in thread_only)
    assert all(chunk.source_type == "document" for chunk in doc_only)
    assert thread_only and doc_only


@pytest.mark.asyncio
async def test_ingest_document_sanitizes_non_primitive_metadata(tmp_path: Path) -> None:
    service = _service(tmp_path)

    # Should not raise even though the caller passed a non-primitive value.
    await service.ingest_document("doc-2", "Some content here.", {"tags": ["a", "b"], "count": 3})

    results = await service.query("some content")
    assert results[0].metadata["count"] == 3
    assert results[0].metadata["tags"] == "['a', 'b']"


@pytest.mark.asyncio
async def test_reingesting_same_thread_upserts_rather_than_duplicates(tmp_path: Path) -> None:
    service = _service(tmp_path)
    thread = _thread("thread-1", "Original body text about pricing.")
    await service.ingest_thread(thread)
    await service.ingest_thread(thread)  # re-ingest the same thread

    results = await service.query("pricing", top_k=10)

    assert len(results) == 1  # not duplicated


@pytest.mark.asyncio
async def test_ingest_thread_embeds_every_message_in_a_single_batch(tmp_path: Path) -> None:
    """One embedding request per thread, not one per message -- on a metered
    embedding API that is the difference between 1 and N calls."""
    embedding = FakeEmbeddingFunction()
    service = ChromaRAGService(
        persist_dir=str(tmp_path / "chroma"), embedding_function=embedding, chunk_size=200, chunk_overlap=20
    )
    messages = [
        EmailMessage(
            message_id=f"m{i}",
            thread_id="thread-9",
            subject="Planning",
            sender=EmailAddress(email=f"p{i}@example.com"),
            body_text=f"Message number {i} about the renewal plan.",
        )
        for i in range(3)
    ]

    await service.ingest_thread(EmailThread(thread_id="thread-9", subject="Planning", messages=messages))

    assert embedding.document_calls == 1
    results = await service.query("renewal plan", top_k=10)
    assert sorted(chunk.metadata["message_id"] for chunk in results) == ["m0", "m1", "m2"]
