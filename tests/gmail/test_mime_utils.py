from __future__ import annotations

import base64

from mailpilot.gmail.mime_utils import (
    build_raw_draft,
    html_to_text,
    parse_address,
    parse_message,
    reply_all_recipients,
)
from mailpilot.schemas.email import EmailAddress, EmailMessage


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8")


def test_parse_message_extracts_headers_and_plain_text_body() -> None:
    raw = {
        "id": "msg-1",
        "threadId": "thread-1",
        "snippet": "Hello there",
        "internalDate": "1700000000000",
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": "Project update"},
                {"name": "From", "value": "Alice <alice@example.com>"},
                {"name": "To", "value": "me@example.com, Bob <bob@example.com>"},
            ],
            "body": {"data": _b64("Hello there, urgent update.")},
        },
    }

    message = parse_message(raw)

    assert message.message_id == "msg-1"
    assert message.thread_id == "thread-1"
    assert message.subject == "Project update"
    assert message.sender == EmailAddress(name="Alice", email="alice@example.com")
    assert [addr.email for addr in message.to] == ["me@example.com", "bob@example.com"]
    assert message.body_text == "Hello there, urgent update."
    assert message.labels == ["INBOX", "UNREAD"]
    assert message.received_at is not None


def test_parse_message_walks_multipart_payload_for_text_and_html() -> None:
    raw = {
        "id": "msg-2",
        "threadId": "thread-2",
        "snippet": "",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [{"name": "Subject", "value": "Multipart"}],
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("plain body")}},
                {"mimeType": "text/html", "body": {"data": _b64("<p>html body</p>")}},
            ],
        },
    }

    message = parse_message(raw)

    assert message.body_text == "plain body"
    assert message.body_html == "<p>html body</p>"


def test_parse_address_handles_display_name_and_bare_address() -> None:
    assert parse_address("Alice <alice@example.com>") == EmailAddress(name="Alice", email="alice@example.com")
    assert parse_address("bob@example.com") == EmailAddress(name=None, email="bob@example.com")


def test_build_raw_draft_roundtrips_headers_and_body() -> None:
    raw = build_raw_draft(
        to=[EmailAddress(name="Alice", email="alice@example.com")],
        cc=[EmailAddress(email="cc@example.com")],
        subject="Hi there",
        body_text="This is the body.",
    )

    decoded = base64.urlsafe_b64decode(raw.encode("utf-8")).decode("utf-8")

    assert "Alice <alice@example.com>" in decoded
    assert "cc@example.com" in decoded
    assert "Hi there" in decoded
    assert "This is the body." in decoded


def _message(sender: str, to: list[str], cc: list[str]) -> EmailMessage:
    return EmailMessage(
        message_id="m",
        thread_id="t",
        sender=EmailAddress(email=sender),
        to=[EmailAddress(email=a) for a in to],
        cc=[EmailAddress(email=a) for a in cc],
    )


def test_reply_all_includes_sender_to_and_cc_minus_self() -> None:
    message = _message("alice@example.com", ["me@example.com", "bob@example.com"], ["carol@example.com"])

    to, cc = reply_all_recipients(message, own_email="me@example.com")

    assert [a.email for a in to] == ["alice@example.com", "bob@example.com"]
    assert [a.email for a in cc] == ["carol@example.com"]


def test_reply_all_dedupes_case_insensitively_and_never_repeats_to_in_cc() -> None:
    message = _message(
        "Alice@Example.com", ["alice@example.com", "bob@example.com"], ["BOB@example.com", "dan@example.com"]
    )

    to, cc = reply_all_recipients(message)

    assert [a.email for a in to] == ["Alice@Example.com", "bob@example.com"]
    assert [a.email for a in cc] == ["dan@example.com"]


def test_reply_all_keeps_sender_when_excluding_self_would_empty_to() -> None:
    """Replying to your own sent message with no other recipients still produces a recipient."""
    message = _message("me@example.com", [], [])

    to, cc = reply_all_recipients(message, own_email="me@example.com")

    assert [a.email for a in to] == ["me@example.com"]
    assert cc == []


def test_reply_all_without_own_email_keeps_everyone() -> None:
    message = _message("alice@example.com", ["me@example.com"], [])

    to, _ = reply_all_recipients(message)

    assert [a.email for a in to] == ["alice@example.com", "me@example.com"]


def test_html_to_text_drops_markup_and_scripts_and_keeps_line_breaks() -> None:
    html = (
        "<html><head><style>p{color:red}</style></head><body>"
        "<p>Hello&nbsp;there,</p><script>alert(1)</script>"
        "<div>Your code is <b>1234</b>.</div><p>Thanks</p></body></html>"
    )

    assert html_to_text(html) == "Hello there,\nYour code is 1234.\nThanks"


def test_parse_message_derives_text_from_html_only_body() -> None:
    raw = {
        "id": "msg-3",
        "threadId": "thread-3",
        "snippet": "",
        "payload": {
            "mimeType": "text/html",
            "headers": [{"name": "Subject", "value": "OTP"}],
            "body": {"data": _b64("<p>Your login OTP is <b>987654</b></p>")},
        },
    }

    message = parse_message(raw)

    assert message.body_text == "Your login OTP is 987654"
    assert message.body_html == "<p>Your login OTP is <b>987654</b></p>"
