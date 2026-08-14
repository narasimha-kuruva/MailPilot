"""IntelligenceService interface.

The concrete implementation (`GeminiIntelligenceService`) asks the LLM for
structured output (`BaseChatModel.with_structured_output`) rather than
free-form text, so callers get typed Pydantic models instead of parsing
prose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mailpilot.schemas.email import EmailMessage, EmailThread
from mailpilot.schemas.intelligence import (
    EmailClassification,
    TaskExtractionResult,
    ThreadSummary,
)


class IntelligenceService(ABC):
    @abstractmethod
    async def classify_email(self, message: EmailMessage) -> EmailClassification:
        raise NotImplementedError

    @abstractmethod
    async def summarize_thread(self, thread: EmailThread) -> ThreadSummary:
        raise NotImplementedError

    @abstractmethod
    async def extract_tasks(self, thread: EmailThread) -> TaskExtractionResult:
        raise NotImplementedError
