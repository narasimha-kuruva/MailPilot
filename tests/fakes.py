"""Test doubles shared across the Phase 2 test suite.

`FakeGmailClient` and `FakeChatModel` let agent/tool/graph logic be
exercised without any real network access, OAuth setup, or LLM API key.
"""

from __future__ import annotations

from typing import Any

from mailpilot.gmail.client import GmailClient
from mailpilot.schemas.email import (
    CreatedDraft,
    DraftEmail,
    EmailAddress,
    EmailMessage,
    EmailThread,
    Label,
)


class FakeGmailClient(GmailClient):
    """In-memory `GmailClient` double. Records calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.sent_draft_ids: list[str] = []
        self._drafts: dict[str, CreatedDraft] = {}
        self._next_draft_id = 1
        self._message = EmailMessage(
            message_id="msg-1",
            thread_id="thread-1",
            subject="Re: Project update",
            sender=EmailAddress(name="Alice", email="alice@example.com"),
            to=[EmailAddress(email="me@example.com")],
            snippet="Let's sync on this...",
            body_text="Let's sync on this urgently.",
            labels=["INBOX"],
        )
        self._thread_messages: list[EmailMessage] | None = None

    def set_message(self, message: EmailMessage) -> None:
        """Override what `search_messages`/`get_message` return (default: the fixture message)."""
        self._message = message

    def set_thread_messages(self, messages: list[EmailMessage]) -> None:
        """Override what `get_thread` returns (default: the single fixture message)."""
        self._thread_messages = messages

    async def search_messages(self, query: str, max_results: int = 25) -> list[EmailMessage]:
        self.calls.append(("search_messages", (query, max_results)))
        return [self._message]

    async def get_message(self, message_id: str) -> EmailMessage:
        self.calls.append(("get_message", (message_id,)))
        return self._message

    async def get_thread(self, thread_id: str) -> EmailThread:
        self.calls.append(("get_thread", (thread_id,)))
        messages = self._thread_messages if self._thread_messages is not None else [self._message]
        return EmailThread(thread_id=thread_id, subject=messages[0].subject, messages=messages)

    async def list_labels(self) -> list[Label]:
        self.calls.append(("list_labels", ()))
        return [Label(label_id="INBOX", name="INBOX", type="system")]

    async def apply_label(self, message_id: str, label_id: str) -> None:
        self.calls.append(("apply_label", (message_id, label_id)))

    async def create_draft(self, draft: DraftEmail) -> CreatedDraft:
        self.calls.append(("create_draft", (draft,)))
        draft_id = f"draft-{self._next_draft_id}"
        self._next_draft_id += 1
        message = EmailMessage(
            message_id=f"{draft_id}-msg",
            thread_id=draft.thread_id or "thread-1",
            subject=draft.subject,
            sender=EmailAddress(email="me@example.com"),
            to=draft.to,
            cc=draft.cc,
            body_text=draft.body_text,
        )
        created = CreatedDraft(draft_id=draft_id, message=message)
        self._drafts[draft_id] = created
        return created

    async def get_draft(self, draft_id: str) -> CreatedDraft:
        self.calls.append(("get_draft", (draft_id,)))
        if draft_id not in self._drafts:
            raise KeyError(f"No such draft: {draft_id}")
        return self._drafts[draft_id]

    async def send_email(self, draft_id: str) -> EmailMessage:
        self.calls.append(("send_email", (draft_id,)))
        self.sent_draft_ids.append(draft_id)
        return self._message


class FakeChatModel:
    """Duck-typed stand-in for a LangChain chat model.

    Only implements what `mailpilot.agent.graph` actually calls:
    `.bind_tools(tools)` (returning something with `.ainvoke`) and
    `.ainvoke(messages)`. Yields the given `AIMessage`s in order, one per call.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.bound_tools: list[Any] | None = None
        self.invocations: list[list[Any]] = []

    def bind_tools(self, tools: list[Any]) -> "FakeChatModel":
        self.bound_tools = tools
        return self

    def with_structured_output(self, schema: Any) -> "FakeChatModel":
        """Ignores `schema` -- tests queue the exact Pydantic instance they
        want returned, same as any other response."""
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        """Yield the next queued response; a queued exception instance is raised instead."""
        self.invocations.append(list(messages))
        response = self._responses[self._index]
        self._index += 1
        if isinstance(response, BaseException):
            raise response
        return response


class FakeEmbeddingFunction:
    """Deterministic, dependency-free stand-in for a real embedding model.

    Maps text to a small vector via a hash, so semantically unrelated texts
    get different (but stable) vectors -- good enough to exercise ChromaDB
    ingestion/query/filtering in tests without any network access. It is
    *not* semantically meaningful the way a real embedding model is.
    """

    def __init__(self, dimensions: int = 16) -> None:
        self._dimensions = dimensions
        self.document_calls = 0  # how many embed_documents() batches were requested

    def _vector(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._dimensions]]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
