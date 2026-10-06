"""A short-lived read cache in front of a `GmailClient`.

An agent run often reads the same mail more than once: it searches, then
reads one of the results; it summarizes a thread, then extracts its tasks.
Each read is a Gmail round trip and costs per-minute quota. This wrapper
keeps messages and threads for `ttl_seconds` (`GMAIL_CACHE_SECONDS`,
default 30):

- `get_message` and `get_thread` are served from the cache when fresh,
  and a search's results warm it, so search-then-read costs one fetch.
- Searches themselves always go to Gmail: "what's new" must be current.
- Any write -- a label, a draft, a send -- clears the cache, so nothing
  stale outlives a change MailPilot made.
- `get_draft` is never cached: an approval request must show the draft
  exactly as it is now.

Callers get copies, so changing a returned object can't change what the
next caller sees. A change made outside MailPilot (in Gmail itself) shows up
once its entry expires.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from mailpilot.gmail.client import GmailClient
from mailpilot.schemas.email import CreatedDraft, DraftEmail, EmailMessage, EmailThread, Label

M = TypeVar("M", bound=BaseModel)

_MAX_ENTRIES = 1000


class CachingGmailClient(GmailClient):
    def __init__(self, inner: GmailClient, ttl_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._inner = inner
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: dict[tuple[str, str], tuple[float, BaseModel]] = {}

    def _get(self, kind: str, key: str, model: type[M]) -> M | None:
        entry = self._entries.get((kind, key))
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at >= self._ttl:
            del self._entries[(kind, key)]
            return None
        return value.model_copy(deep=True)  # type: ignore[return-value]

    def _put(self, kind: str, key: str, value: BaseModel) -> None:
        if len(self._entries) >= _MAX_ENTRIES:
            now = self._clock()
            self._entries = {k: v for k, v in self._entries.items() if now - v[0] < self._ttl}
            if len(self._entries) >= _MAX_ENTRIES:
                self._entries.clear()
        self._entries[(kind, key)] = (self._clock(), value.model_copy(deep=True))

    def clear(self) -> None:
        self._entries.clear()

    # --- reads -----------------------------------------------------------------

    async def search_messages(self, query: str, max_results: int = 25) -> list[EmailMessage]:
        messages = await self._inner.search_messages(query, max_results)
        for message in messages:
            self._put("message", message.message_id, message)
        return messages

    async def get_message(self, message_id: str) -> EmailMessage:
        cached = self._get("message", message_id, EmailMessage)
        if cached is not None:
            return cached
        message = await self._inner.get_message(message_id)
        self._put("message", message_id, message)
        return message

    async def get_thread(self, thread_id: str) -> EmailThread:
        cached = self._get("thread", thread_id, EmailThread)
        if cached is not None:
            return cached
        thread = await self._inner.get_thread(thread_id)
        self._put("thread", thread_id, thread)
        return thread

    async def list_labels(self) -> list[Label]:
        return await self._inner.list_labels()

    async def get_draft(self, draft_id: str) -> CreatedDraft:
        return await self._inner.get_draft(draft_id)  # never cached: approvals must see the draft as it is

    # --- writes: each one clears the cache ---------------------------------------

    async def apply_label(self, message_id: str, label_id: str) -> None:
        try:
            await self._inner.apply_label(message_id, label_id)
        finally:
            self.clear()

    async def create_draft(self, draft: DraftEmail) -> CreatedDraft:
        try:
            return await self._inner.create_draft(draft)
        finally:
            self.clear()

    async def send_email(self, draft_id: str) -> EmailMessage:
        try:
            return await self._inner.send_email(draft_id)
        finally:
            self.clear()
