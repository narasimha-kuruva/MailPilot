"""`IntelligenceService` backed by a LangChain chat model's structured output.

Takes any `BaseChatModel`-shaped object (duck-typed: just needs
`.with_structured_output(schema)` returning something with `.ainvoke()`),
so tests can inject a fake instead of a real Gemini model -- see
`tests/fakes.py::FakeChatModel`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mailpilot.intelligence.service import IntelligenceService
from mailpilot.prompts import GROUNDING_NOTICE, UNTRUSTED_CONTENT_NOTICE, wrap_untrusted
from mailpilot.schemas.email import EmailMessage, EmailThread
from mailpilot.schemas.intelligence import (
    EmailClassification,
    TaskExtractionResult,
    ThreadSummary,
)


def _render_message(message: EmailMessage) -> str:
    body = message.body_text or message.snippet or ""
    return (
        f"From: {message.sender.email}\n"
        f"Subject: {message.subject}\n"
        f"Received: {message.received_at.isoformat() if message.received_at else 'unknown'}\n\n"
        f"{body}"
    )


def _render_thread(thread: EmailThread) -> str:
    return "\n\n---\n\n".join(_render_message(message) for message in thread.messages)


class GeminiIntelligenceService(IntelligenceService):
    def __init__(self, chat_model: Any) -> None:
        self._chat_model = chat_model

    async def classify_email(self, message: EmailMessage) -> EmailClassification:
        structured = self._chat_model.with_structured_output(EmailClassification)
        prompt = [
            SystemMessage(
                content=(
                    "You classify a single email's category, priority, urgency, and "
                    "whether it requires action. " + UNTRUSTED_CONTENT_NOTICE
                )
            ),
            HumanMessage(content=wrap_untrusted("email_body", _render_message(message))),
        ]
        return await structured.ainvoke(prompt)

    async def summarize_thread(self, thread: EmailThread) -> ThreadSummary:
        structured = self._chat_model.with_structured_output(ThreadSummary)
        prompt = [
            SystemMessage(
                content=(
                    "You summarize an email thread: what it's about, where it "
                    "stands, and the key points a busy reader needs. "
                    + UNTRUSTED_CONTENT_NOTICE
                    + " "
                    + GROUNDING_NOTICE
                )
            ),
            HumanMessage(content=wrap_untrusted("thread", f"thread_id: {thread.thread_id}\n\n{_render_thread(thread)}")),
        ]
        result = await structured.ainvoke(prompt)
        result.thread_id = thread.thread_id
        return result

    async def extract_tasks(self, thread: EmailThread) -> TaskExtractionResult:
        structured = self._chat_model.with_structured_output(TaskExtractionResult)
        prompt = [
            SystemMessage(
                content=(
                    "You extract concrete, actionable tasks from an email thread. "
                    "Only extract a deadline or owner if the text states one "
                    "explicitly -- leave them null rather than guessing. "
                    + UNTRUSTED_CONTENT_NOTICE
                )
            ),
            HumanMessage(content=wrap_untrusted("thread", f"thread_id: {thread.thread_id}\n\n{_render_thread(thread)}")),
        ]
        result = await structured.ainvoke(prompt)
        for task in result.tasks:
            task.source_thread_id = task.source_thread_id or thread.thread_id
        return result
