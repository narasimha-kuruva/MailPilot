from __future__ import annotations

from mailpilot.safety.policy import requires_approval


def test_send_email_requires_approval() -> None:
    assert requires_approval("send_email") is True


def test_other_tools_do_not_require_approval() -> None:
    for name in ["search_emails", "read_email", "read_thread", "list_labels", "apply_label", "create_draft"]:
        assert requires_approval(name) is False
