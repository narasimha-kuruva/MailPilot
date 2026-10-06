"""Explicit indexing into the knowledge store: the shared helpers, the agent tool, the API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mailpilot.agent.reply_drafting import exclude_thread
from mailpilot.api.deps import get_gmail_client, get_rag_service
from mailpilot.evaluation.mailbox import InMemoryGmailClient
from mailpilot.evaluation.scenarios import OWN_EMAIL, USER_LABELS, default_mailbox
from mailpilot.main import create_app
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.rag.indexing import index_threads, thread_ids_for_query
from mailpilot.schemas.email import EmailAddress, EmailMessage, EmailThread
from tests.fakes import FakeEmbeddingFunction


def _store(tmp_path: Path) -> ChromaRAGService:
    return ChromaRAGService(
        persist_dir=str(tmp_path / "chroma"), embedding_function=FakeEmbeddingFunction(), chunk_size=200, chunk_overlap=20
    )


def _mailbox() -> InMemoryGmailClient:
    return InMemoryGmailClient(default_mailbox(), own_email=OWN_EMAIL, user_labels=USER_LABELS)


def _thread(thread_id: str, *bodies: str) -> EmailThread:
    messages = [
        EmailMessage(
            message_id=f"{thread_id}-m{i}",
            thread_id=thread_id,
            subject="Renewal",
            sender=EmailAddress(email="alice@example.com"),
            body_text=body,
        )
        for i, body in enumerate(bodies)
    ]
    return EmailThread(thread_id=thread_id, subject="Renewal", messages=messages)


# --- The store ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reindexing_a_thread_replaces_its_old_chunks(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert await store.ingest_thread(_thread("t1", "first message", "second message")) == 2
    assert await store.ingest_thread(_thread("t1", "only message now")) == 1

    assert (await store.stats()).model_dump() == {"chunks": 1, "threads": 1, "documents": 0}


@pytest.mark.asyncio
async def test_delete_and_stats(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.ingest_thread(_thread("t1", "renewal price is $500"))
    await store.ingest_thread(_thread("t2", "lunch on thursday"))
    await store.ingest_document("pricing", "Standard plan: $500 per year.", {"title": "Pricing"})

    assert (await store.stats()).model_dump() == {"chunks": 3, "threads": 2, "documents": 1}
    assert await store.delete_thread("t1") == 1
    assert await store.delete_thread("t1") == 0  # already gone
    assert await store.delete_document("pricing") == 1
    assert (await store.stats()).model_dump() == {"chunks": 1, "threads": 1, "documents": 0}


@pytest.mark.asyncio
async def test_the_reply_filter_excludes_only_the_thread_being_replied_to(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.ingest_thread(_thread("current", "renewal price question"))
    await store.ingest_thread(_thread("older", "renewal price was $480 last year"))
    await store.ingest_document("pricing", "Renewal price list.", {})

    results = await store.query("renewal price", top_k=5, where=exclude_thread("current"))

    assert {chunk.source_id for chunk in results} == {"older", "pricing"}


# --- Helpers and the agent tool ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_search_yields_each_thread_once_newest_first() -> None:
    assert await thread_ids_for_query(_mailbox(), "from:alice", max_threads=10) == ["t-contract"]
    assert await thread_ids_for_query(_mailbox(), "is:unread", max_threads=10) == ["t-invoice", "t-lunch", "t-contract"]


@pytest.mark.asyncio
async def test_a_thread_that_fails_is_reported_and_the_rest_still_index(tmp_path: Path) -> None:
    store = _store(tmp_path)

    response = await index_threads(_mailbox(), store, ["t-lunch", "t-missing", "t-lunch"])

    assert [t.thread_id for t in response.indexed] == ["t-lunch"]  # the duplicate id is indexed once
    assert [f.thread_id for f in response.failed] == ["t-missing"]
    assert "404" in response.failed[0].error


@pytest.mark.asyncio
async def test_the_agent_tool_indexes_a_thread(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tools = build_tools(_mailbox(), rag_service=store)

    result = await tools["index_thread"].run(thread_id="t-contract")

    assert (result.thread_id, result.messages, result.chunks) == ("t-contract", 2, 2)
    assert (await store.stats()).threads == 1


# --- The API -------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_gmail_client] = _mailbox
    store = _store(tmp_path)
    app.dependency_overrides[get_rag_service] = lambda: store
    return TestClient(app)


def test_index_threads_by_id_and_by_search(api: TestClient) -> None:
    by_id = api.post("/api/v1/context/threads", json={"thread_ids": ["t-lunch"]})
    by_search = api.post("/api/v1/context/threads", json={"query": "label:clients", "max_threads": 5})

    assert by_id.status_code == 200
    assert [t["thread_id"] for t in by_id.json()["indexed"]] == ["t-lunch"]
    assert [t["thread_id"] for t in by_search.json()["indexed"]] == ["t-contract", "t-design"]
    assert api.get("/api/v1/context").json() == {"chunks": 4, "threads": 3, "documents": 0}


@pytest.mark.parametrize(
    "body",
    [{}, {"thread_ids": ["t1"], "query": "from:bob"}, {"thread_ids": []}, {"query": "x", "max_threads": 51}],
)
def test_index_threads_needs_exactly_one_valid_source(api: TestClient, body: dict) -> None:
    assert api.post("/api/v1/context/threads", json=body).status_code == 422


def test_documents_can_be_indexed_replaced_and_removed(api: TestClient) -> None:
    first = api.post("/api/v1/context/documents", json={"document_id": "pricing", "text": "Plan A: $500.", "metadata": {"title": "Pricing"}})
    api.post("/api/v1/context/documents", json={"document_id": "pricing", "text": "Plan A: $520."})
    removed = api.delete("/api/v1/context/documents/pricing")

    assert first.json() == {"document_id": "pricing", "chunks": 1}
    assert removed.json() == {"source_id": "pricing", "chunks_removed": 1}  # the replaced version left nothing behind
    assert api.get("/api/v1/context").json()["documents"] == 0


def test_removing_a_thread(api: TestClient) -> None:
    api.post("/api/v1/context/threads", json={"thread_ids": ["t-contract"]})

    assert api.delete("/api/v1/context/threads/t-contract").json() == {"source_id": "t-contract", "chunks_removed": 2}
