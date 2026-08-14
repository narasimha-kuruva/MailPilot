from __future__ import annotations

from pathlib import Path

import pytest

from mailpilot.intelligence.gemini_service import GeminiIntelligenceService
from mailpilot.mcp.tools.classify_email import ClassifyEmailTool
from mailpilot.mcp.tools.draft_grounded_reply import DraftGroundedReplyTool
from mailpilot.mcp.tools.extract_tasks import ExtractTasksTool
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.mcp.tools.summarize_thread import SummarizeThreadTool
from mailpilot.rag.chroma_service import ChromaRAGService
from mailpilot.schemas.intelligence import (
    EmailCategory,
    EmailClassification,
    GroundedDraft,
    Priority,
    TaskExtractionResult,
    ThreadSummary,
)
from tests.fakes import FakeChatModel, FakeEmbeddingFunction, FakeGmailClient


def _rag_service(tmp_path: Path) -> ChromaRAGService:
    return ChromaRAGService(
        persist_dir=str(tmp_path / "chroma"), embedding_function=FakeEmbeddingFunction(), chunk_size=200
    )


@pytest.mark.asyncio
async def test_classify_email_tool_fetches_then_classifies() -> None:
    gmail_client = FakeGmailClient()
    expected = EmailClassification(
        category=EmailCategory.CLIENT, priority=Priority.HIGH, is_urgent=True, requires_action=True, reasoning="x"
    )
    intelligence = GeminiIntelligenceService(FakeChatModel([expected]))
    tool = ClassifyEmailTool(gmail_client, intelligence)

    result = await tool.run(message_id="msg-1")

    assert result == expected
    assert ("get_message", ("msg-1",)) in gmail_client.calls


@pytest.mark.asyncio
async def test_summarize_thread_tool() -> None:
    gmail_client = FakeGmailClient()
    expected = ThreadSummary(thread_id="", summary="summary text", key_points=[])
    intelligence = GeminiIntelligenceService(FakeChatModel([expected]))
    tool = SummarizeThreadTool(gmail_client, intelligence)

    result = await tool.run(thread_id="thread-1")

    assert result.summary == "summary text"
    assert result.thread_id == "thread-1"


@pytest.mark.asyncio
async def test_extract_tasks_tool() -> None:
    gmail_client = FakeGmailClient()
    expected = TaskExtractionResult(tasks=[])
    intelligence = GeminiIntelligenceService(FakeChatModel([expected]))
    tool = ExtractTasksTool(gmail_client, intelligence)

    result = await tool.run(thread_id="thread-1")

    assert result == expected


@pytest.mark.asyncio
async def test_draft_grounded_reply_tool_creates_a_draft(tmp_path: Path) -> None:
    gmail_client = FakeGmailClient()
    rag_service = _rag_service(tmp_path)
    draft_response = GroundedDraft(subject="Re: Project update", body_text="Sounds good, thanks!")
    chat_model = FakeChatModel([draft_response])
    tool = DraftGroundedReplyTool(gmail_client, rag_service, chat_model)

    result = await tool.run(thread_id="thread-1", intent="acknowledge and thank them")

    assert result.draft_id == "draft-1"
    created_calls = [c for c in gmail_client.calls if c[0] == "create_draft"]
    assert len(created_calls) == 1
    draft_arg = created_calls[0][1][0]
    assert draft_arg.to[0].email == "alice@example.com"  # the thread's sender, from FakeGmailClient


@pytest.mark.asyncio
async def test_draft_grounded_reply_tool_never_lets_the_model_choose_the_recipient(tmp_path: Path) -> None:
    """`GroundedDraft` (the model's structured output) has no recipient field --
    the tool always addresses the reply to the thread's own last sender. This
    is a stronger guarantee than validating the model's choice after the
    fact: there is no code path for the model to pick a recipient at all.
    `validate_reply_recipients` (unit-tested in tests/safety/test_guardrails.py)
    is still called as defense in depth."""
    gmail_client = FakeGmailClient()
    rag_service = _rag_service(tmp_path)
    draft_response = GroundedDraft(subject="Re: Project update", body_text="ok")
    tool = DraftGroundedReplyTool(gmail_client, rag_service, FakeChatModel([draft_response]))

    await tool.run(thread_id="thread-1", intent="reply")

    created_calls = [c for c in gmail_client.calls if c[0] == "create_draft"]
    draft_arg = created_calls[0][1][0]
    assert [addr.email for addr in draft_arg.to] == ["alice@example.com"]
    assert draft_arg.cc == []


def test_registry_includes_intelligence_and_rag_tools_when_deps_provided(tmp_path: Path) -> None:
    gmail_client = FakeGmailClient()
    intelligence = GeminiIntelligenceService(FakeChatModel([]))
    rag_service = _rag_service(tmp_path)
    chat_model = FakeChatModel([])

    tools = build_tools(
        gmail_client, intelligence_service=intelligence, rag_service=rag_service, chat_model=chat_model
    )

    assert set(tools) == {
        "search_emails",
        "read_email",
        "read_thread",
        "list_labels",
        "apply_label",
        "create_draft",
        "send_email",
        "classify_email",
        "summarize_thread",
        "extract_tasks",
        "draft_grounded_reply",
    }


def test_registry_omits_phase3_tools_when_deps_not_provided() -> None:
    tools = build_tools(FakeGmailClient())

    assert "classify_email" not in tools
    assert "draft_grounded_reply" not in tools
    assert len(tools) == 7
