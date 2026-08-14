from __future__ import annotations

import pytest

from mailpilot.safety.guardrails import (
    GuardrailViolation,
    find_unsupported_claims,
    validate_reply_recipients,
)
from mailpilot.schemas.email import DraftEmail, EmailAddress, EmailMessage, EmailThread


def _thread() -> EmailThread:
    message = EmailMessage(
        message_id="msg-1",
        thread_id="thread-1",
        subject="Renewal",
        sender=EmailAddress(email="alice@example.com"),
        to=[EmailAddress(email="me@example.com")],
        cc=[EmailAddress(email="bob@example.com")],
        body_text="The renewal is due on 2026-09-01 for $500.",
    )
    return EmailThread(thread_id="thread-1", subject="Renewal", messages=[message])


def test_validate_reply_recipients_allows_known_addresses() -> None:
    draft = DraftEmail(to=[EmailAddress(email="alice@example.com")], subject="Re: Renewal", body_text="ok")

    validate_reply_recipients(draft, _thread())  # must not raise


def test_validate_reply_recipients_rejects_unknown_recipient() -> None:
    """This is the concrete guard against a prompt-injected 'also send to attacker@evil.com'."""
    draft = DraftEmail(
        to=[EmailAddress(email="alice@example.com"), EmailAddress(email="attacker@evil.com")],
        subject="Re: Renewal",
        body_text="ok",
    )

    with pytest.raises(GuardrailViolation, match="attacker@evil.com"):
        validate_reply_recipients(draft, _thread())


def test_find_unsupported_claims_flags_date_and_amount_not_in_sources() -> None:
    draft_body = "Sure, we can renew by 2026-12-25 for $9999."
    sources = ["The renewal is due on 2026-09-01 for $500."]

    notes = find_unsupported_claims(draft_body, sources)

    assert any("2026-12-25" in note for note in notes)
    assert any("$9999" in note for note in notes)


def test_find_unsupported_claims_allows_facts_present_in_sources() -> None:
    draft_body = "As discussed, the renewal is due on 2026-09-01 for $500."
    sources = ["The renewal is due on 2026-09-01 for $500."]

    notes = find_unsupported_claims(draft_body, sources)

    assert notes == []
