"""The real Gmail API, read-only."""

from __future__ import annotations

import pytest

from mailpilot.gmail.google_client import GoogleGmailClient
from mailpilot.resilience import ErrorClass, classify_error, status_code

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_labels_include_the_system_inbox(gmail_client: GoogleGmailClient) -> None:
    labels = await gmail_client.list_labels()

    assert "INBOX" in {label.label_id for label in labels}


@pytest.mark.asyncio
async def test_search_returns_parsed_messages(gmail_client: GoogleGmailClient) -> None:
    messages = await gmail_client.search_messages("in:inbox", max_results=3)

    assert 0 < len(messages) <= 3
    for message in messages:
        assert message.message_id and message.thread_id
        assert "@" in message.sender.email
        assert "INBOX" in message.labels


@pytest.mark.asyncio
async def test_read_a_message_and_its_thread(gmail_client: GoogleGmailClient) -> None:
    [latest] = await gmail_client.search_messages("in:inbox", max_results=1)

    message = await gmail_client.get_message(latest.message_id)
    thread = await gmail_client.get_thread(latest.thread_id)

    assert message.message_id == latest.message_id
    assert message.body_text  # the text part, or text derived from an HTML-only body
    assert latest.message_id in {m.message_id for m in thread.messages}


@pytest.mark.asyncio
async def test_a_bad_message_id_is_a_permanent_client_error(gmail_client: GoogleGmailClient) -> None:
    with pytest.raises(Exception) as error:
        await gmail_client.get_message("0000000000000000")

    assert status_code(error.value) in (400, 404)
    assert classify_error(error.value) is ErrorClass.PERMANENT
