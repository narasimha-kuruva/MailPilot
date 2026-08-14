"""Domain models for retrieval-augmented context lookups."""

from __future__ import annotations

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """One retrieved passage, with enough metadata to cite its source."""

    text: str
    score: float
    source_id: str
    source_type: str  # e.g. "email_thread" | "document"
    metadata: dict = {}
