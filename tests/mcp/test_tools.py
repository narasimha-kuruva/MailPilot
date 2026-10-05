from __future__ import annotations

import pytest
from pydantic import ValidationError

from mailpilot.mcp.tools.apply_label import ApplyLabelTool
from mailpilot.mcp.tools.create_draft import CreateDraftTool
from mailpilot.mcp.tools.list_labels import ListLabelsTool
from mailpilot.mcp.tools.read_email import ReadEmailTool
from mailpilot.mcp.tools.read_thread import ReadThreadTool
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.mcp.tools.search_emails import SearchEmailsTool
from mailpilot.mcp.tools.send_email import SendEmailTool
from mailpilot.schemas.email import EmailSummary
from mailpilot.safety.guardrails import GuardrailViolation
from tests.fakes import FakeGmailClient


@pytest.mark.asyncio
async def test_search_emails_delegates_to_gmail_client() -> None:
    client = FakeGmailClient()
    tool = SearchEmailsTool(client)

    results = await tool.run(query="is:unread", max_results=10)

    assert client.calls == [("search_messages", ("is:unread", 10))]
    assert len(results) == 1
    summary = results[0]
    assert isinstance(summary, EmailSummary)
    assert summary.message_id == "msg-1"
    assert summary.sender.email == "alice@example.com"
    assert summary.snippet == "Let's sync on this..."
    assert not hasattr(summary, "body_text")  # bodies are fetched via read_email


@pytest.mark.asyncio
async def test_read_email_and_read_thread_strip_html_bodies() -> None:
    client = FakeGmailClient()
    client._message = client._message.model_copy(update={"body_html": "<p>big markup</p>"})

    message = await ReadEmailTool(client).run(message_id="msg-1")
    thread = await ReadThreadTool(client).run(thread_id="thread-1")

    assert message.body_html is None
    assert message.body_text == "Let's sync on this urgently."
    assert all(m.body_html is None for m in thread.messages)


@pytest.mark.asyncio
async def test_search_emails_rejects_invalid_args() -> None:
    tool = SearchEmailsTool(FakeGmailClient())

    with pytest.raises(ValidationError):
        await tool.run(max_results=10)  # missing required "query"


@pytest.mark.asyncio
async def test_read_email_and_read_thread() -> None:
    client = FakeGmailClient()

    message = await ReadEmailTool(client).run(message_id="msg-1")
    thread = await ReadThreadTool(client).run(thread_id="thread-1")

    assert message.message_id == "msg-1"
    assert thread.thread_id == "thread-1"


@pytest.mark.asyncio
async def test_list_labels_and_apply_label() -> None:
    client = FakeGmailClient()

    labels = await ListLabelsTool(client).run()
    outcome = await ApplyLabelTool(client).run(message_id="msg-1", label_id="INBOX")

    assert labels[0].label_id == "INBOX"
    assert outcome == {"status": "applied", "message_id": "msg-1", "label_id": "INBOX"}
    assert ("apply_label", ("msg-1", "INBOX")) in client.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("label_id", ["TRASH", "SPAM"])
async def test_apply_label_refuses_trash_and_spam_before_touching_gmail(label_id: str) -> None:
    """The guardrail is a hard block in the tool, not an approval round-trip:
    the Gmail client must never even be asked."""
    client = FakeGmailClient()

    with pytest.raises(GuardrailViolation):
        await ApplyLabelTool(client).run(message_id="msg-1", label_id=label_id)

    assert not any(name == "apply_label" for name, _ in client.calls)


@pytest.mark.asyncio
async def test_apply_label_rejects_empty_ids() -> None:
    with pytest.raises(ValidationError):
        await ApplyLabelTool(FakeGmailClient()).run(message_id="", label_id="INBOX")


@pytest.mark.asyncio
async def test_create_draft_converts_address_strings() -> None:
    client = FakeGmailClient()
    tool = CreateDraftTool(client)

    created = await tool.run(
        to=["Alice <alice@example.com>"],
        cc=["cc@example.com"],
        subject="Re: hello",
        body_text="Sounds good.",
    )

    assert created.draft_id == "draft-1"
    ((_, (draft,)),) = [c for c in client.calls if c[0] == "create_draft"]
    assert draft.to[0].email == "alice@example.com"
    assert draft.cc[0].email == "cc@example.com"


@pytest.mark.asyncio
async def test_send_email_tool_calls_gmail_client_directly() -> None:
    """The tool itself performs no approval check -- that's the graph's job."""
    client = FakeGmailClient()
    tool = SendEmailTool(client)

    await tool.run(draft_id="draft-1")

    assert client.sent_draft_ids == ["draft-1"]


def test_registry_builds_all_seven_tools_keyed_by_name() -> None:
    tools = build_tools(FakeGmailClient())

    assert set(tools) == {
        "search_emails",
        "read_email",
        "read_thread",
        "list_labels",
        "apply_label",
        "create_draft",
        "send_email",
    }
