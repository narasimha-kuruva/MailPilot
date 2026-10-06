"""The short-lived Gmail read cache."""

from __future__ import annotations

import pytest

from mailpilot.config import Settings
from mailpilot.gmail.caching import CachingGmailClient
from mailpilot.schemas.email import DraftEmail, EmailAddress
from tests.fakes import FakeGmailClient


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _cached(ttl: float = 30.0) -> tuple[CachingGmailClient, FakeGmailClient, _Clock]:
    inner, clock = FakeGmailClient(), _Clock()
    return CachingGmailClient(inner, ttl_seconds=ttl, clock=clock), inner, clock


def _count(inner: FakeGmailClient, method: str) -> int:
    return sum(1 for name, _ in inner.calls if name == method)


@pytest.mark.asyncio
async def test_repeated_reads_are_served_from_the_cache_until_they_expire() -> None:
    cache, inner, clock = _cached(ttl=30)

    await cache.get_thread("thread-1")
    await cache.get_thread("thread-1")
    await cache.get_message("msg-1")
    await cache.get_message("msg-1")
    assert (_count(inner, "get_thread"), _count(inner, "get_message")) == (1, 1)

    clock.now += 30
    await cache.get_thread("thread-1")
    assert _count(inner, "get_thread") == 2


@pytest.mark.asyncio
async def test_a_search_warms_the_cache_but_is_never_cached_itself() -> None:
    cache, inner, _ = _cached()

    [found] = await cache.search_messages("is:unread")
    await cache.get_message(found.message_id)  # search-then-read: one Gmail round trip
    await cache.search_messages("is:unread")

    assert _count(inner, "get_message") == 0
    assert _count(inner, "search_messages") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("write", ["apply_label", "create_draft", "send_email"])
async def test_any_write_clears_the_cache(write: str) -> None:
    cache, inner, _ = _cached()
    await cache.get_thread("thread-1")

    if write == "apply_label":
        await cache.apply_label("msg-1", "STARRED")
    elif write == "create_draft":
        await cache.create_draft(DraftEmail(to=[EmailAddress(email="a@example.com")], subject="s", body_text="b"))
    else:
        await cache.send_email("draft-1")
    await cache.get_thread("thread-1")

    assert _count(inner, "get_thread") == 2


@pytest.mark.asyncio
async def test_drafts_are_never_cached() -> None:
    cache, inner, _ = _cached()
    created = await cache.create_draft(DraftEmail(to=[EmailAddress(email="a@example.com")], subject="s", body_text="b"))

    await cache.get_draft(created.draft_id)
    await cache.get_draft(created.draft_id)

    assert _count(inner, "get_draft") == 2


@pytest.mark.asyncio
async def test_callers_get_copies() -> None:
    cache, _, _ = _cached()

    first = await cache.get_thread("thread-1")
    first.messages[0].subject = "changed by a caller"

    assert (await cache.get_thread("thread-1")).messages[0].subject == "Re: Project update"


def test_the_cache_is_on_by_default_and_can_be_turned_off() -> None:
    assert Settings(_env_file=None).gmail_cache_seconds == 30
    assert Settings(_env_file=None, gmail_cache_seconds=0).gmail_cache_seconds == 0
