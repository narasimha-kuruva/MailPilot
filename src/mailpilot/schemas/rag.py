"""Domain models for retrieval-augmented context: lookups and indexing."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from mailpilot.schemas.email import EmailAddress


class RetrievedChunk(BaseModel):
    """One retrieved passage, with enough metadata to cite its source."""

    text: str
    score: float
    source_id: str
    source_type: str  # e.g. "email_thread" | "document"
    metadata: dict = {}


class ContextStats(BaseModel):
    """What the knowledge store holds."""

    chunks: int
    threads: int
    documents: int


class IndexThreadsRequest(BaseModel):
    """Threads to index: listed by id, or found with a Gmail search -- exactly one of the two."""

    thread_ids: list[str] | None = Field(default=None, min_length=1, max_length=50)
    query: str | None = Field(default=None, min_length=1, description="Gmail search, e.g. 'from:alice newer_than:90d'.")
    max_threads: int = Field(default=10, ge=1, le=50, description="With `query`: index at most this many threads.")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "IndexThreadsRequest":
        if (self.thread_ids is None) == (self.query is None):
            raise ValueError("Give either `thread_ids` or `query`, not both or neither.")
        return self


class ThreadSearchRequest(BaseModel):
    """A Gmail search whose threads to list for choosing what to index. Nothing is stored."""

    query: str = Field(..., min_length=1, description="Gmail search, e.g. 'from:alice newer_than:90d'.")
    max_threads: int = Field(default=10, ge=1, le=50, description="List at most this many threads.")


class ThreadCandidate(BaseModel):
    """A thread found by a search, described by its newest matching message."""

    thread_id: str
    subject: str
    sender: EmailAddress
    received_at: datetime | None = None
    snippet: str = ""
    indexed_chunks: int = Field(0, description="Chunks the store already holds for this thread; 0 if it isn't indexed.")


class ThreadSearchResponse(BaseModel):
    threads: list[ThreadCandidate] = Field(default_factory=list)


class IndexedThread(BaseModel):
    thread_id: str
    subject: str
    messages: int
    chunks: int
    replaced_chunks: int = Field(
        0, description="Chunks from an earlier indexing of this thread, now replaced; 0 if it wasn't indexed before."
    )


class IndexFailure(BaseModel):
    thread_id: str
    error: str


class IndexThreadsResponse(BaseModel):
    indexed: list[IndexedThread] = Field(default_factory=list)
    failed: list[IndexFailure] = Field(default_factory=list)


class IndexDocumentRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=500_000)
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Stored with every chunk, e.g. {'title': 'Pricing 2026'}."
    )


class IndexedDocument(BaseModel):
    document_id: str
    chunks: int


class RemovedContext(BaseModel):
    source_id: str
    chunks_removed: int
