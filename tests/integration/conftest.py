"""Integration tests: the real Gmail account and the real model from `.env`.

Opt-in only (they're skipped otherwise, see tests/conftest.py):

    MAILPILOT_RUN_INTEGRATION_TESTS=1 pytest -m integration

They only ever READ from Gmail -- no draft is created, nothing is sent or
relabelled -- and make a handful of model calls with the configured
provider (mind the Gemini free tier's ~20 requests per model per day).
Run them from the repository root so the relative paths in `.env` resolve.
Missing credentials fail the run rather than skip it: having asked for the
integration tests, a broken setup should be loud.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel

from mailpilot.config import Settings
from mailpilot.gmail.google_client import GoogleGmailClient
from mailpilot.llm.providers import LLMProviderError, build_chat_model


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture
def gmail_client(settings: Settings) -> GoogleGmailClient:
    if not Path(settings.google_oauth_token_file).exists():
        pytest.fail(
            f"No Gmail OAuth token at {settings.google_oauth_token_file}: start the app once and "
            "complete the consent flow (see README, 'Running locally')."
        )
    return GoogleGmailClient(settings)


@pytest.fixture
def chat_model(settings: Settings) -> BaseChatModel:
    try:
        return build_chat_model(settings)
    except LLMProviderError as exc:
        pytest.fail(f"LLM provider '{settings.llm_provider}' is not usable: {exc}")
