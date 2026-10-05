"""Provider selection tests. None of these need a Gemini key, Ollama, or the network:
Ollama's model-list endpoint is replaced with an in-memory stub."""

from __future__ import annotations

import urllib.error

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from mailpilot.api import deps
from mailpilot.config import Settings
from mailpilot.llm import providers
from mailpilot.llm.providers import (
    LLMProviderError,
    OllamaEmbeddingFunction,
    build_chat_model,
    build_embedding_function,
    ensure_ollama_ready,
)
from mailpilot.rag.embeddings import GeminiEmbeddingFunction


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _tags(*names: str):
    return lambda _base_url: {"models": [{"name": n} for n in names]}


def _unreachable(_base_url: str):
    raise urllib.error.URLError("connection refused")


# --- configuration -----------------------------------------------------------


def test_provider_defaults_to_gemini_with_ollama_defaults_present() -> None:
    settings = _settings()

    assert settings.llm_provider == "gemini"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "gemma4:e2b"


def test_provider_is_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")

    settings = _settings()

    assert settings.llm_provider == "ollama"
    assert settings.ollama_model == "gemma4:e4b"
    assert settings.ollama_base_url == "http://gpu-box:11434"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(llm_provider="openai")


# --- gemini ------------------------------------------------------------------


def test_gemini_without_api_key_gives_clear_error() -> None:
    with pytest.raises(LLMProviderError, match="GOOGLE_API_KEY"):
        build_chat_model(_settings(llm_provider="gemini", google_api_key=None))


def test_gemini_builds_chat_model_and_embeddings_with_configured_names() -> None:
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = _settings(google_api_key="test-key", gemini_model="gemini-3.7-flash")

    chat_model = build_chat_model(settings)
    embeddings = build_embedding_function(settings)

    assert isinstance(chat_model, ChatGoogleGenerativeAI)
    assert chat_model.model.endswith("gemini-3.7-flash")
    assert isinstance(embeddings, GeminiEmbeddingFunction)


# --- ollama readiness --------------------------------------------------------


def test_ollama_not_reachable_gives_install_hint() -> None:
    with pytest.raises(LLMProviderError, match="not reachable at http://localhost:11434") as info:
        ensure_ollama_ready(_settings(llm_provider="ollama"), fetch_tags=_unreachable)

    assert "https://ollama.com" in str(info.value)


def test_ollama_model_not_installed_tells_user_to_pull_it() -> None:
    with pytest.raises(LLMProviderError, match=r"ollama pull gemma4:e2b") as info:
        ensure_ollama_ready(_settings(llm_provider="ollama"), fetch_tags=_tags("llama3:latest"))

    assert "Installed models: llama3:latest" in str(info.value)


def test_ollama_ready_when_model_installed() -> None:
    ensure_ollama_ready(_settings(llm_provider="ollama"), fetch_tags=_tags("gemma4:e2b"))  # must not raise


def test_ollama_tagless_model_name_matches_latest() -> None:
    settings = _settings(llm_provider="ollama", ollama_model="gemma4")

    ensure_ollama_ready(settings, fetch_tags=_tags("gemma4:latest"))  # must not raise
    with pytest.raises(LLMProviderError):
        ensure_ollama_ready(settings, fetch_tags=_tags("gemma4:e2b"))  # a specific tag is not "latest"


# --- ollama factories ----------------------------------------------------------


def test_ollama_builds_chat_model_with_configured_model_and_url(monkeypatch) -> None:
    from langchain_ollama import ChatOllama

    monkeypatch.setattr(providers, "_fetch_tags", _tags("gemma4:e2b"))
    settings = _settings(llm_provider="ollama", ollama_base_url="http://gpu-box:11434")

    chat_model = build_chat_model(settings)

    assert isinstance(chat_model, ChatOllama)
    assert chat_model.model == "gemma4:e2b"
    assert chat_model.base_url == "http://gpu-box:11434"
    # The agent graph needs both of these from whatever model it gets.
    assert callable(chat_model.bind_tools)
    assert callable(chat_model.with_structured_output)


def test_ollama_chat_model_is_refused_when_server_is_down(monkeypatch) -> None:
    monkeypatch.setattr(providers, "_fetch_tags", _unreachable)

    with pytest.raises(LLMProviderError, match="not reachable"):
        build_chat_model(_settings(llm_provider="ollama"))


@pytest.mark.asyncio
async def test_ollama_embeddings_defer_readiness_check_until_first_use(monkeypatch) -> None:
    """Constructing the agent must not require the embedding model; using RAG does."""
    monkeypatch.setattr(providers, "_fetch_tags", _tags("gemma4:e2b"))  # chat model present, embedding model not
    settings = _settings(llm_provider="ollama", ollama_embedding_model="embeddinggemma")

    embeddings = build_embedding_function(settings)  # must not raise
    assert isinstance(embeddings, OllamaEmbeddingFunction)

    with pytest.raises(LLMProviderError, match=r"ollama pull embeddinggemma"):
        await embeddings.embed_query("hello")


# --- API wiring --------------------------------------------------------------


def test_deps_translate_provider_error_into_503(monkeypatch) -> None:
    monkeypatch.setattr(providers, "_fetch_tags", _unreachable)
    monkeypatch.setattr(deps, "get_settings", lambda: _settings(llm_provider="ollama"))
    deps.get_chat_model.cache_clear()
    try:
        with pytest.raises(HTTPException) as info:
            deps.get_chat_model()
        assert info.value.status_code == 503
        assert "Ollama" in info.value.detail
    finally:
        deps.get_chat_model.cache_clear()


def test_deps_gemini_without_key_is_still_503(monkeypatch) -> None:
    monkeypatch.setattr(deps, "get_settings", lambda: _settings(llm_provider="gemini", google_api_key=None))
    deps.get_chat_model.cache_clear()
    try:
        with pytest.raises(HTTPException) as info:
            deps.get_chat_model()
        assert info.value.status_code == 503
        assert "GOOGLE_API_KEY" in info.value.detail
    finally:
        deps.get_chat_model.cache_clear()
