"""The in-memory mailbox the evaluation agent runs against."""

from __future__ import annotations

import pytest

from mailpilot.evaluation.mailbox import InMemoryGmailClient, MailboxError
from mailpilot.evaluation.scenarios import OWN_EMAIL, USER_LABELS, default_mailbox
from mailpilot.resilience import ErrorClass, classify_error
from mailpilot.schemas.email import DraftEmail, EmailAddress


def _mailbox(**kwargs) -> InMemoryGmailClient:
    return InMemoryGmailClient(default_mailbox(), own_email=OWN_EMAIL, user_labels=USER_LABELS, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_ids"),
    [
        ("is:unread", ["m-invoice", "m-lunch", "m-contract-2"]),  # newest first
        ("is:unread in:inbox", ["m-invoice", "m-lunch", "m-contract-2"]),
        ("from:bob", ["m-lunch"]),
        ("from:alice is:unread", ["m-contract-2"]),
        ('subject:"Q3 contract renewal"', ["m-contract-2", "m-contract-1"]),
        ("label:clients -is:unread", ["m-design", "m-contract-1"]),
        ("from:bob OR from:carol", ["m-lunch", "m-design"]),
        ("category:promotions", ["m-news"]),
        ("in:inbox newsletter", ["m-news"]),
        ("from:carol newer_than:7d", ["m-design"]),  # unmodelled operators are ignored
        ("from:nobody", []),
    ],
)
async def test_search_understands_common_gmail_operators(query: str, expected_ids: list[str]) -> None:
    found = await _mailbox().search_messages(query, max_results=10)

    assert [m.message_id for m in found] == expected_ids


@pytest.mark.asyncio
async def test_sending_consumes_the_draft_like_gmail() -> None:
    mailbox = _mailbox()
    draft = await mailbox.create_draft(
        DraftEmail(to=[EmailAddress(email="bob@example.com")], subject="Hi", body_text="Hello")
    )

    sent = await mailbox.send_email(draft.draft_id)

    assert sent.labels == ["SENT"] and mailbox.sent == [sent]
    with pytest.raises(MailboxError) as error:
        await mailbox.send_email(draft.draft_id)
    assert error.value.status_code == 404
    assert len(mailbox.sent) == 1


@pytest.mark.asyncio
async def test_failures_are_injected_per_call_then_clear() -> None:
    mailbox = _mailbox(failures={"search_messages": [503, 403]})

    with pytest.raises(MailboxError) as first:
        await mailbox.search_messages("is:unread")
    with pytest.raises(MailboxError) as second:
        await mailbox.search_messages("is:unread")
    third = await mailbox.search_messages("is:unread")

    assert classify_error(first.value) is ErrorClass.TRANSIENT
    assert classify_error(second.value) is ErrorClass.PERMANENT
    assert len(third) == 3
    assert mailbox.call_count("search_messages") == 3


@pytest.mark.asyncio
async def test_unknown_labels_and_messages_are_rejected() -> None:
    mailbox = _mailbox()

    with pytest.raises(MailboxError, match="Invalid label"):
        await mailbox.apply_label("m-lunch", "Label_nope")
    with pytest.raises(MailboxError, match="not found"):
        await mailbox.get_message("m-nope")
    await mailbox.apply_label("m-lunch", "STARRED")
    assert "STARRED" in (await mailbox.get_message("m-lunch")).labels
