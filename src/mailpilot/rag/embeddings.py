"""Embedding generation for RAG.

`EmbeddingFunction` is a minimal `Protocol` so `ChromaRAGService` doesn't
depend on a concrete provider, and tests can inject a deterministic fake
(see `tests/fakes.py::FakeEmbeddingFunction`) instead of calling a real
Gemini embeddings endpoint.
"""

from __future__ import annotations

from typing import Protocol

from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingFunction(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingFunction:
    """`EmbeddingFunction` backed by Google's Gemini embedding model."""

    def __init__(self, model: str, google_api_key: str) -> None:
        self._embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=google_api_key)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)
