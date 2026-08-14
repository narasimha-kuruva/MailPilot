"""Text chunking for RAG ingestion.

A small hand-rolled recursive splitter (no extra dependency): split on
paragraph breaks first, then sentences, then raw characters, packing
pieces up to `chunk_size` with `overlap` characters repeated between
consecutive chunks so a fact split across a boundary is still retrievable
from at least one chunk.
"""

from __future__ import annotations

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_into_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= 400:
            units.append(paragraph)
        else:
            units.extend(s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip())
    return units


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split `text` into overlapping chunks of at most `chunk_size` characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # overlap must never reach chunk_size, or the hard-slice step below
    # becomes zero/negative (ValueError / an infinite-looking range).
    overlap = max(0, min(overlap, chunk_size - 1))

    units = _split_into_units(text)
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + unit if overlap else unit
        else:
            # a single unit longer than chunk_size: hard-slice it
            for start in range(0, len(unit), chunk_size - overlap):
                chunks.append(unit[start : start + chunk_size])
            current = ""

        if len(current) > chunk_size:
            # the overlap-prefixed remainder can still exceed chunk_size; flush it
            chunks.append(current)
            current = ""

    if current:
        chunks.append(current)

    return chunks
