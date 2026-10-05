"""Build the chat model and embedding function for the configured provider.

    LLM_PROVIDER=ollama  -> ChatOllama + Ollama embeddings (local, no API quota; the default)
    LLM_PROVIDER=gemini  -> ChatGoogleGenerativeAI + Gemini embeddings (needs GOOGLE_API_KEY)

Selection is driven by configuration alone and is deterministic: if the
chosen provider is unavailable the error says so, and nothing ever falls
back to the other provider. Both return LangChain `BaseChatModel`s (with
`bind_tools` / `with_structured_output`) and `EmbeddingFunction`s, so the
agent graph, tools, RAG and intelligence code are provider-agnostic.

The Ollama path never downloads models: `ensure_ollama_ready` checks that
the server is reachable and that the configured model has already been
pulled, and raises `LLMProviderError` with the exact fix otherwise, so a
missing install surfaces as one clear sentence instead of a connection
stack trace from deep inside LangChain.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from mailpilot.config import Settings
from mailpilot.rag.embeddings import EmbeddingFunction, GeminiEmbeddingFunction

OLLAMA_INSTALL_HINT = "Install Ollama from https://ollama.com and start it (it runs as a background service)"


class LLMProviderError(RuntimeError):
    """The configured LLM provider can't be used; the message says why and how to fix it."""


# ---------------------------------------------------------------------------
# Ollama readiness
# ---------------------------------------------------------------------------


def _fetch_tags(base_url: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """GET /api/tags from the Ollama server (its 'list installed models' endpoint)."""
    url = base_url.rstrip("/") + "/api/tags"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310 - local URL from config
        return json.loads(response.read().decode("utf-8"))


def _model_matches(configured: str, installed: str) -> bool:
    """`gemma4:e2b` matches exactly; a tag-less `gemma4` matches `gemma4:latest`."""
    if configured == installed:
        return True
    return ":" not in configured and installed == f"{configured}:latest"


def ensure_ollama_ready(
    settings: Settings,
    model: str | None = None,
    *,
    fetch_tags: Callable[[str], dict[str, Any]] | None = None,
) -> None:
    """Raise `LLMProviderError` unless Ollama is reachable and `model` is installed.

    `fetch_tags` is injectable for tests; production uses `_fetch_tags`.
    """
    model = model or settings.ollama_model
    base_url = settings.ollama_base_url
    try:
        tags = (fetch_tags or _fetch_tags)(base_url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LLMProviderError(
            f"LLM_PROVIDER=ollama but Ollama is not reachable at {base_url} ({exc}). "
            f"{OLLAMA_INSTALL_HINT}, or set OLLAMA_BASE_URL to where it is running. "
            "To use the cloud model instead, set LLM_PROVIDER=gemini (with GOOGLE_API_KEY)."
        ) from exc

    installed = [str(item.get("name", "")) for item in tags.get("models", [])]
    if not any(_model_matches(model, name) for name in installed):
        installed_text = ", ".join(sorted(installed)) if installed else "none"
        raise LLMProviderError(
            f"Ollama is running at {base_url} but the model '{model}' is not installed. "
            f"Run `ollama pull {model}` (MailPilot never downloads models itself). "
            f"Installed models: {installed_text}. "
            "To use the cloud model instead, set LLM_PROVIDER=gemini (with GOOGLE_API_KEY)."
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _require_google_api_key(settings: Settings) -> str:
    if not settings.google_api_key:
        raise LLMProviderError(
            "GOOGLE_API_KEY is not configured. Set it in .env before using the agent endpoints, "
            "or set LLM_PROVIDER=ollama to use a local model instead."
        )
    return settings.google_api_key


def build_chat_model(settings: Settings) -> BaseChatModel:
    """The chat model the agent graph, planner, intelligence service and drafting all share."""
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries=1 disables the client's own retries (6 by default, with
        # ~30s of backoff -- it retried a daily-quota 429 five times before
        # giving up). `mailpilot.resilience` is the single retry layer: it
        # knows a daily quota from a rate limit and respects the run's
        # deadline. 1, not 0: the SDK treats 0 as "use my defaults".
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model, google_api_key=_require_google_api_key(settings), max_retries=1
        )

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        ensure_ollama_ready(settings, settings.ollama_model)
        # Ollama's server default context window is 4096 tokens. The system
        # prompt, the tool schemas and one full email already fill that, and
        # the model is then left a handful of tokens for its answer (seen
        # live: done_reason=length, empty reply). The models themselves
        # support far more; OLLAMA_NUM_CTX sets what we actually ask for.
        return ChatOllama(
            model=settings.ollama_model, base_url=settings.ollama_base_url, num_ctx=settings.ollama_num_ctx
        )

    raise LLMProviderError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'ollama'.")


class OllamaEmbeddingFunction:
    """`EmbeddingFunction` backed by a local Ollama embedding model.

    Readiness (server up, model pulled) is checked on first use rather than
    at construction, so the agent can start -- and every non-RAG tool can
    run -- without the embedding model installed. The first RAG call then
    fails with a clear `LLMProviderError` instead of an HTTP error.
    """

    def __init__(self, settings: Settings) -> None:
        from langchain_ollama import OllamaEmbeddings

        self._settings = settings
        self._model = settings.ollama_embedding_model
        self._embeddings = OllamaEmbeddings(model=self._model, base_url=settings.ollama_base_url)
        self._checked = False

    def _ensure_ready(self) -> None:
        if not self._checked:
            ensure_ollama_ready(self._settings, self._model)
            self._checked = True

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_ready()
        return await self._embeddings.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        self._ensure_ready()
        return await self._embeddings.aembed_query(text)


def build_embedding_function(settings: Settings) -> EmbeddingFunction:
    if settings.llm_provider == "gemini":
        return GeminiEmbeddingFunction(model=settings.rag_embedding_model, google_api_key=_require_google_api_key(settings))

    if settings.llm_provider == "ollama":
        return OllamaEmbeddingFunction(settings)

    raise LLMProviderError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'ollama'.")
