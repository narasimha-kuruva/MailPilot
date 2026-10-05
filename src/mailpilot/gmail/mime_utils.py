"""Conversion between raw Gmail API JSON/MIME and `mailpilot.schemas.email` models.

Kept separate from `GoogleGmailClient` so the parsing/building logic can be
unit-tested against fixture payloads without any network access or OAuth
setup.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from typing import Any

from mailpilot.schemas.email import EmailAddress, EmailMessage


def parse_address(raw: str) -> EmailAddress:
    """Parse a single `"Name <email@example.com>"` or bare address string."""
    name, email_addr = parseaddr(raw)
    return EmailAddress(name=name or None, email=email_addr)


def parse_address_list(raw: str) -> list[EmailAddress]:
    """Parse a comma-separated header value (e.g. a `To:` header) into addresses."""
    if not raw:
        return []
    return [parse_address(part) for part in raw.split(",") if part.strip()]


def reply_all_recipients(
    message: EmailMessage, own_email: str | None = None
) -> tuple[list[EmailAddress], list[EmailAddress]]:
    """Compute "reply all" recipients for a reply to `message`.

    To = the original sender followed by the original To list; Cc = the
    original Cc list. Addresses are de-duplicated case-insensitively, the
    user's own address (`own_email`, if known) is dropped so they don't
    mail themselves, and anything already in To is not repeated in Cc.
    If removing the user's own address would leave To empty (e.g. replying
    to a message the user sent), the original sender is kept so the draft
    always has at least one recipient.

    Every address returned comes from `message` itself, so the result is
    always safe under `mailpilot.safety.guardrails.validate_reply_recipients`.
    """
    own = own_email.strip().lower() if own_email else None
    seen: set[str] = set()

    def _take(candidates: list[EmailAddress]) -> list[EmailAddress]:
        kept: list[EmailAddress] = []
        for address in candidates:
            key = address.email.strip().lower()
            if not key or key in seen or key == own:
                continue
            seen.add(key)
            kept.append(address)
        return kept

    to = _take([message.sender, *message.to])
    if not to:
        seen.add(message.sender.email.strip().lower())
        to = [message.sender]
    cc = _take(message.cc)
    return to, cc


def _format_address(address: EmailAddress) -> str:
    # formataddr quotes the display name when needed (e.g. "Doe, John"), so
    # a comma in a name can't be mistaken for an address separator.
    return formataddr((address.name or "", address.email))


def _header(headers: list[dict[str, str]], name: str) -> str:
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def _decode_body(data: str) -> str:
    # Gmail base64url-encodes body data and may omit padding.
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _extract_bodies(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Walk the MIME part tree and return the first `(text_plain, text_html)` found."""
    text_plain: str | None = None
    text_html: str | None = None

    def _walk(part: dict[str, Any]) -> None:
        nonlocal text_plain, text_html
        mime_type = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data")
        if mime_type == "text/plain" and body_data and text_plain is None:
            text_plain = _decode_body(body_data)
        elif mime_type == "text/html" and body_data and text_html is None:
            text_html = _decode_body(body_data)
        for sub_part in part.get("parts", []) or []:
            _walk(sub_part)

    _walk(payload)
    return text_plain, text_html


def parse_message(raw_message: dict[str, Any]) -> EmailMessage:
    """Convert a Gmail API `Message` resource (format=`full`) into an `EmailMessage`."""
    payload = raw_message.get("payload", {})
    headers = payload.get("headers", [])
    body_text, body_html = _extract_bodies(payload)

    received_at = None
    internal_date = raw_message.get("internalDate")
    if internal_date:
        received_at = datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc)

    return EmailMessage(
        message_id=raw_message["id"],
        thread_id=raw_message["threadId"],
        subject=_header(headers, "Subject"),
        sender=parse_address(_header(headers, "From")),
        to=parse_address_list(_header(headers, "To")),
        cc=parse_address_list(_header(headers, "Cc")),
        snippet=raw_message.get("snippet", ""),
        body_text=body_text,
        body_html=body_html,
        labels=raw_message.get("labelIds", []),
        received_at=received_at,
    )


def build_raw_draft(
    to: list[EmailAddress],
    cc: list[EmailAddress],
    subject: str,
    body_text: str,
) -> str:
    """Build a base64url-encoded RFC 2822 message for the Gmail drafts/messages API."""
    message = MIMEText(body_text)
    message["To"] = ", ".join(_format_address(address) for address in to)
    if cc:
        message["Cc"] = ", ".join(_format_address(address) for address in cc)
    message["Subject"] = subject
    return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
