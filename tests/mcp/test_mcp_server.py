"""The MCP server, driven through the MCP SDK's own client (in process)."""

from __future__ import annotations

import pytest
from mcp import Client

from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.mcp.server import NOT_EXPOSED, build_server
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.schemas.audit import ToolCallStatus
from tests.fakes import FakeGmailClient


def _server(gmail: FakeGmailClient, audit: InMemoryAuditService):
    return build_server(build_tools(gmail), audit, session_id="test")


def _text(result) -> str:
    return "".join(block.text for block in result.content)


@pytest.mark.asyncio
async def test_the_tools_are_listed_with_their_schemas_and_without_send_email() -> None:
    async with Client(_server(FakeGmailClient(), InMemoryAuditService())) as client:
        listed = (await client.list_tools()).tools

    names = {tool.name for tool in listed}
    assert "search_emails" in names and "create_draft" in names and "apply_label" in names
    assert not names & NOT_EXPOSED
    search = next(tool for tool in listed if tool.name == "search_emails")
    assert "query" in search.input_schema["properties"]
    assert search.input_schema["required"] == ["query"]


@pytest.mark.asyncio
async def test_a_call_runs_the_tool_fences_the_result_and_is_audited() -> None:
    gmail, audit = FakeGmailClient(), InMemoryAuditService()

    async with Client(_server(gmail, audit)) as client:
        result = await client.call_tool("search_emails", {"query": "is:unread", "max_results": 3})

    assert not result.is_error
    text = _text(result)
    assert text.startswith("<untrusted_tool_result>") and "alice@example.com" in text
    assert ("search_messages", ("is:unread", 3)) in gmail.calls
    [record] = await audit.get_history("mcp-test")
    assert (record.tool_name, record.status, record.agent_request) == ("search_emails", ToolCallStatus.SUCCESS, "(MCP client)")
    assert record.duration_ms is not None


@pytest.mark.asyncio
async def test_sending_is_refused_over_mcp_and_never_reaches_gmail() -> None:
    gmail = FakeGmailClient()

    async with Client(_server(gmail, InMemoryAuditService())) as client:
        result = await client.call_tool("send_email", {"draft_id": "draft-1"})

    assert result.is_error
    assert "isn't available over MCP" in _text(result)
    assert gmail.sent_draft_ids == [] and not [c for c in gmail.calls if c[0] == "send_email"]


@pytest.mark.asyncio
async def test_guardrails_still_hold_over_mcp() -> None:
    gmail, audit = FakeGmailClient(), InMemoryAuditService()

    async with Client(_server(gmail, audit)) as client:
        result = await client.call_tool("apply_label", {"message_id": "msg-1", "label_id": "TRASH"})

    assert result.is_error
    assert "hide or delete" in _text(result)
    assert not [c for c in gmail.calls if c[0] == "apply_label"]
    assert (await audit.get_history("mcp-test"))[0].status is ToolCallStatus.FAILURE


@pytest.mark.asyncio
async def test_invalid_arguments_and_unknown_tools_are_errors() -> None:
    async with Client(_server(FakeGmailClient(), InMemoryAuditService())) as client:
        bad_args = await client.call_tool("search_emails", {"query": ""})
        unknown = await client.call_tool("delete_everything", {})

    assert bad_args.is_error and "query" in _text(bad_args)
    assert unknown.is_error and "Unknown tool" in _text(unknown)


def test_without_gmail_authorization_the_stdio_server_refuses_to_start(tmp_path, monkeypatch, capsys) -> None:
    """Over stdio, stdout is the protocol: the consent flow (which prints there) must never run."""
    from mailpilot.api import deps
    from mailpilot.config import Settings
    from mailpilot.mcp.__main__ import main

    settings = Settings(_env_file=None, google_oauth_token_file=str(tmp_path / "missing-token.json"))
    monkeypatch.setattr(deps, "get_settings", lambda: settings)

    assert main([]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Authorize once first" in captured.err
