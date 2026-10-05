"""Application configuration, sourced from environment variables / .env.

All secrets (API keys, OAuth client files) are read from the environment at
runtime and are never hardcoded. See `.env.example` for the full list of
supported variables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, typed application configuration.

    Instantiated once via `get_settings()` and injected wherever needed
    rather than imported as a module-level singleton, so tests can override
    it via FastAPI dependency overrides.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- LLM provider selection (Phase 5) ---
    # "ollama" (default): local model via Ollama, no API quota; models must be
    # pulled separately (`ollama pull <model>`), MailPilot never downloads them.
    # "gemini": hosted Google Gemini, needs GOOGLE_API_KEY. Selection is
    # deterministic -- there is no automatic fallback from one to the other.
    # See `mailpilot.llm.providers`.
    llm_provider: Literal["gemini", "ollama"] = "ollama"

    # --- LLM Provider (Google Gemini) ---
    google_api_key: str | None = None
    gemini_model: str = "gemini-3.7-flash"

    # --- Local LLM (Ollama) -- used only when llm_provider == "ollama" ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"
    ollama_embedding_model: str = "embeddinggemma"  # only needed by the RAG tools
    ollama_num_ctx: int = 16384  # context window; Ollama's own default (4096) can't hold one full email

    # --- Gmail OAuth ---
    google_oauth_client_secrets_file: str = "./secrets/client_secret.json"
    google_oauth_token_file: str = "./secrets/token.json"
    gmail_user_email: str | None = None

    # --- ChromaDB / RAG (Phase 3) ---
    chroma_persist_dir: str = "./data/chroma"
    rag_embedding_model: str = "models/gemini-embedding-2"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 100
    rag_top_k: int = 5
    rag_max_context_chars: int = 6000

    # --- Safety ---
    require_approval_before_send: bool = Field(
        default=True,
        description=(
            "Must remain True in all environments. Sending email without "
            "explicit human approval is not a supported configuration."
        ),
    )
    approval_ttl_seconds: int = 1800  # Phase 4: how long a pending approval stays valid

    # --- Agent execution limits (Phase 4) ---
    agent_max_steps: int = 20  # max agent (LLM) turns per run
    agent_max_tool_calls: int = 15  # max tool executions per run
    agent_max_tool_retries: int = 2  # per-call retries on transient tool errors
    agent_tool_timeout_seconds: float = 30.0
    agent_max_execution_seconds: float = 120.0
    agent_max_output_chars: int = 12000  # cap on a single tool result / generated response
    agent_max_llm_retries: int = 3  # per model call, on transient 503/429/timeouts
    agent_llm_retry_base_delay_seconds: float = 2.0
    agent_llm_timeout_seconds: float = 60.0  # raise for slow local (CPU-only) models

    # --- Reliability (Phase 5) ---
    gmail_max_retries: int = 3  # per request; rate-limit 403/429s are retried with backoff
    gmail_retry_base_delay_seconds: float = 1.0  # 1s, 2s, 4s
    gmail_max_concurrent_fetches: int = 5  # message bodies fetched in parallel per search

    @field_validator("require_approval_before_send")
    @classmethod
    def _approval_before_send_cannot_be_disabled(cls, value: bool) -> bool:
        # Not env-configurable to False on purpose: the field exists to
        # document the invariant, not to offer a way to turn it off. The
        # actual gate is `mailpilot.safety.policy.SENSITIVE_TOOL_NAMES`,
        # which doesn't read this setting at all -- so if this were allowed
        # to silently become False, an operator could believe they'd
        # disabled the approval gate when nothing had actually changed.
        if value is not True:
            raise ValueError(
                "require_approval_before_send cannot be set to False. Sending email "
                "without human approval is not a supported configuration."
            )
        return value

    @model_validator(mode="after")
    def _rag_overlap_must_be_smaller_than_chunk_size(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                f"rag_chunk_overlap ({self.rag_chunk_overlap}) must be smaller than "
                f"rag_chunk_size ({self.rag_chunk_size})."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached with `lru_cache` so environment parsing happens once per process;
    tests can bypass this by constructing `Settings(...)` directly or by
    clearing the cache with `get_settings.cache_clear()`.
    """
    return Settings()
