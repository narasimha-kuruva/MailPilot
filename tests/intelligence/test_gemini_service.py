from __future__ import annotations

import pytest

from mailpilot.intelligence.gemini_service import GeminiIntelligenceService
from mailpilot.schemas.email import EmailAddress, EmailMessage, EmailThread
from mailpilot.schemas.intelligence import (
    EmailCategory,
    EmailClassification,
    ExtractedTask,
    Priority,
    TaskExtractionResult,
    ThreadSummary,
)
from tests.fakes import FakeChatModel


def _message(body: str = "Please send the proposal ASAP.") -> EmailMessage:
    return EmailMessage(
        message_id="msg-1",
        thread_id="thread-1",
        subject="Urgent: proposal needed",
        sender=EmailAddress(email="client@example.com"),
        body_text=body,
    )


def _thread(messages: list[EmailMessage] | None = None) -> EmailThread:
    return EmailThread(thread_id="thread-1", subject="Urgent: proposal needed", messages=messages or [_message()])


@pytest.mark.asyncio
async def test_classify_email_returns_structured_classification() -> None:
    expected = EmailClassification(
        category=EmailCategory.CLIENT,
        priority=Priority.URGENT,
        is_urgent=True,
        requires_action=True,
        reasoning="Client is asking for a proposal urgently.",
    )
    chat_model = FakeChatModel([expected])
    service = GeminiIntelligenceService(chat_model)

    result = await service.classify_email(_message())

    assert result == expected
    # the raw email body must have reached the prompt fenced as untrusted content
    prompt = chat_model.invocations[0]
    assert any("untrusted_email_body" in str(m.content) for m in prompt)


@pytest.mark.asyncio
async def test_summarize_thread_fills_in_thread_id() -> None:
    expected = ThreadSummary(thread_id="", summary="Client wants a proposal.", key_points=["urgent request"])
    chat_model = FakeChatModel([expected])
    service = GeminiIntelligenceService(chat_model)

    result = await service.summarize_thread(_thread())

    assert result.thread_id == "thread-1"
    assert result.summary == "Client wants a proposal."


@pytest.mark.asyncio
async def test_extract_tasks_backfills_source_thread_id_when_missing() -> None:
    expected = TaskExtractionResult(
        tasks=[ExtractedTask(action="Send the proposal", deadline=None, owner=None)]
    )
    chat_model = FakeChatModel([expected])
    service = GeminiIntelligenceService(chat_model)

    result = await service.extract_tasks(_thread())

    assert result.tasks[0].source_thread_id == "thread-1"
    assert result.tasks[0].deadline is None  # never invented
    assert result.tasks[0].owner is None
