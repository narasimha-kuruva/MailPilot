"""Phase 5.1: secrets never reach log output or the stored audit trail.

Fake credentials are assembled at runtime (prefix + filler) rather than
written out, so nothing in this file looks like a real key to a secret
scanner.
"""

from __future__ import annotations

import json
import logging
import sys

import pytest

from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.logging_config import StructuredFormatter
from mailpilot.redaction import REDACTED, redact_text, redact_value
from mailpilot.schemas.audit import AuditRecord, ToolCallStatus

FAKE_API_KEY = "AIza" + "Sy" + "x" * 33
FAKE_ACCESS_TOKEN = "ya29." + "a0Ab" + "Q" * 40
FAKE_REFRESH_TOKEN = "1//" + "0g" + "Z" * 40
FAKE_CLIENT_SECRET = "GOCSPX-" + "k" * 28
FAKE_JWT = "eyJ" + "h" * 20 + ".eyJ" + "p" * 30 + "." + "s" * 30


@pytest.mark.parametrize(
    "secret", [FAKE_API_KEY, FAKE_ACCESS_TOKEN, FAKE_REFRESH_TOKEN, FAKE_CLIENT_SECRET, FAKE_JWT]
)
def test_known_credential_shapes_are_redacted_in_place(secret: str) -> None:
    redacted = redact_text(f"request failed (credential {secret}) -- retrying")

    assert secret not in redacted
    assert redacted == f"request failed (credential {REDACTED}) -- retrying"


def test_bearer_header_is_redacted() -> None:
    redacted = redact_text("Authorization: Bearer abcdefghijklmnop1234567890")

    assert redacted == f"Authorization: Bearer {REDACTED}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GET /token?access_token=opaque-value-1&alt=json", f"GET /token?access_token={REDACTED}&alt=json"),
        ('{"refresh_token": "opaque-value-2", "expiry": 1}', f'{{"refresh_token": "{REDACTED}", "expiry": 1}}'),
        ("{'client_secret': 'opaque-value-3'}", f"{{'client_secret': '{REDACTED}'}}"),
        ('{\\"api_key\\": \\"opaque-value-4\\"}', f'{{\\"api_key\\": \\"{REDACTED}\\"}}'),
        ("password: hunter2", f"password: {REDACTED}"),
        ("GOOGLE_API_KEY=opaque-value-5", f"GOOGLE_API_KEY={REDACTED}"),
        # The label can end a longer name.
        ("GMAIL_REFRESH_TOKEN=opaque-value-6", f"GMAIL_REFRESH_TOKEN={REDACTED}"),
        ("oauth_access_token: opaque-value-7", f"oauth_access_token: {REDACTED}"),
        ("MAILPILOT_API_KEY=opaque-value-8", f"MAILPILOT_API_KEY={REDACTED}"),
        # A quoted value is redacted whole, spaces, commas and escaped quotes included.
        ('{"password": "correct horse, battery"}', f'{{"password": "{REDACTED}"}}'),
        ('{"password": "say \\"hi\\" twice", "user": "bo"}', f'{{"password": "{REDACTED}", "user": "bo"}}'),
        ("{'password': 'two words'}", f"{{'password': '{REDACTED}'}}"),
        ('{\\"password\\": \\"two words\\"}', f'{{\\"password\\": \\"{REDACTED}\\"}}'),
        # An unquoted value keeps its commas.
        ("login?password=ab,cd&next=/", f"login?password={REDACTED}&next=/"),
    ],
)
def test_values_labelled_as_secrets_are_redacted(text: str, expected: str) -> None:
    assert redact_text(text) == expected


@pytest.mark.parametrize("text", ["api_keys_count=3", "password_hint: blue", "the passwords: rotated"])
def test_names_that_merely_contain_a_secret_label_are_left_alone(text: str) -> None:
    assert redact_text(text) == text


def test_a_shaped_secret_under_a_secret_label_is_redacted_once() -> None:
    assert redact_text(f"api_key={FAKE_API_KEY}") == f"api_key={REDACTED}"


@pytest.mark.parametrize(
    "text",
    [
        "message 18c2f0a9b3d4e5f6 in thread 18c2f0a9b3d4e5f6",  # Gmail ids
        "conversation 3f1c9a2e-8b7d-4c6e-9f0a-1b2c3d4e5f60",  # uuid4
        "draft r-1234567890123456789 for alice@example.com",
        "max_tokens=512, num_ctx=16384, input_tokens: 2048",
        "bearer tokens are rotated hourly; the password is never logged",
        "token: expired",  # a bare "token:" in prose is not treated as a label
    ],
)
def test_ordinary_text_is_left_alone(text: str) -> None:
    assert redact_text(text) == text


def test_redact_value_handles_nested_structures_and_sensitive_keys() -> None:
    value = {
        "tool": "search_emails",
        "Access-Token": "anything at all",
        "google_api_key": None,  # empty: left visible, so "not set" stays debuggable
        "nested": {"headers": [{"Authorization": "Basic dXNlcjpwYXNz"}], "note": f"key {FAKE_API_KEY}"},
        "count": 3,
    }

    assert redact_value(value) == {
        "tool": "search_emails",
        "Access-Token": REDACTED,
        "google_api_key": None,
        "nested": {"headers": [{"Authorization": REDACTED}], "note": f"key {REDACTED}"},
        "count": 3,
    }


def _format(record: logging.LogRecord) -> tuple[str, dict]:
    line = StructuredFormatter().format(record)
    return line, json.loads(line)  # must still be valid JSON after redaction


def test_log_message_and_extra_fields_are_redacted() -> None:
    record = logging.LogRecord(
        "mailpilot.test", logging.INFO, __file__, 1, "calling Gemini with key %s", (FAKE_API_KEY,), None
    )
    record.extra_fields = {"refresh_token": FAKE_REFRESH_TOKEN, "detail": {"note": f"token {FAKE_ACCESS_TOKEN}"}}

    line, payload = _format(record)

    for secret in (FAKE_API_KEY, FAKE_REFRESH_TOKEN, FAKE_ACCESS_TOKEN):
        assert secret not in line
    assert payload["message"] == f"calling Gemini with key {REDACTED}"
    assert payload["refresh_token"] == REDACTED
    assert payload["detail"] == {"note": f"token {REDACTED}"}


def test_exception_text_in_a_log_line_is_redacted() -> None:
    try:
        raise RuntimeError(f"401 from token endpoint: client_secret={FAKE_CLIENT_SECRET}")
    except RuntimeError:
        record = logging.LogRecord("mailpilot.test", logging.ERROR, __file__, 1, "refresh failed", None, sys.exc_info())

    line, payload = _format(record)

    assert FAKE_CLIENT_SECRET not in line
    assert f"client_secret={REDACTED}" in payload["exception"]


def test_objects_json_cannot_encode_are_redacted_after_str() -> None:
    class Credentials:
        def __str__(self) -> str:
            return f"Credentials(token={FAKE_ACCESS_TOKEN})"

    record = logging.LogRecord("mailpilot.test", logging.INFO, __file__, 1, "loaded", None, None)
    record.extra_fields = {"creds": Credentials()}

    line, payload = _format(record)

    assert FAKE_ACCESS_TOKEN not in line
    assert payload["creds"] == f"Credentials(token={REDACTED})"


@pytest.mark.asyncio
async def test_audit_records_are_stored_redacted() -> None:
    audit = InMemoryAuditService()

    await audit.record(
        AuditRecord(
            conversation_id="c1",
            agent_request=f"email my key {FAKE_API_KEY} to bob",
            tool_name="create_draft",
            tool_args={"to": ["bob@example.com"], "body_text": f"here it is: {FAKE_API_KEY}"},
            status=ToolCallStatus.SUCCESS,
            result_summary=f'{{"draft_id": "draft-1", "body": "password: hunter2 {FAKE_API_KEY}"}}',
        )
    )

    [stored] = await audit.get_history("c1")
    assert FAKE_API_KEY not in stored.model_dump_json()
    assert stored.agent_request == f"email my key {REDACTED} to bob"
    assert stored.tool_args == {"to": ["bob@example.com"], "body_text": f"here it is: {REDACTED}"}
    assert "hunter2" not in (stored.result_summary or "")
    assert stored.status is ToolCallStatus.SUCCESS


@pytest.mark.asyncio
async def test_audit_ids_are_kept_so_records_can_still_be_found() -> None:
    audit = InMemoryAuditService()

    await audit.record(
        AuditRecord(
            conversation_id="ticket-api_key:7781",
            step_id="call-password=1",
            agent_request="check the ticket",
            tool_name="search_emails",
            status=ToolCallStatus.SUCCESS,
        )
    )

    [stored] = await audit.get_history("ticket-api_key:7781")
    assert stored.step_id == "call-password=1"
