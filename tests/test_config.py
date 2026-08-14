from __future__ import annotations

import pytest
from pydantic import ValidationError

from mailpilot.config import Settings


def test_defaults_are_safe() -> None:
    settings = Settings(_env_file=None)

    assert settings.require_approval_before_send is True
    assert settings.app_env == "development"


def test_settings_reads_env_override(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_PORT", "9000")

    settings = Settings(_env_file=None)

    assert settings.app_env == "production"
    assert settings.api_port == 9000


def test_require_approval_before_send_cannot_be_disabled() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, require_approval_before_send=False)


def test_rag_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rag_chunk_size=100, rag_chunk_overlap=100)
