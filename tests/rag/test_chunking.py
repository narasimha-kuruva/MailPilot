from __future__ import annotations

from mailpilot.rag.chunking import chunk_text


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("short text", chunk_size=200) == ["short text"]


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("", chunk_size=200) == []
    assert chunk_text("   ", chunk_size=200) == []


def test_long_text_is_split_within_chunk_size_bound() -> None:
    text = " ".join(f"This is sentence number {i}." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200 + 30  # allow for the overlap prefix on continuation chunks


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"This is sentence number {i}." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=30)

    # the tail of one chunk should reappear at the start of the next
    overlap_sample = chunks[0][-20:]
    assert overlap_sample in chunks[1]


def test_single_oversized_unit_is_hard_sliced() -> None:
    text = "x" * 500
    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    assert "".join(chunks).replace("", "") != ""
    for chunk in chunks:
        assert len(chunk) <= 100


def test_overlap_equal_to_chunk_size_does_not_crash() -> None:
    """A misconfigured overlap >= chunk_size must degrade gracefully, not raise."""
    chunks = chunk_text("x" * 500, chunk_size=100, overlap=100)

    assert chunks  # did not crash, produced something
    for chunk in chunks:
        assert len(chunk) <= 100


def test_overlap_greater_than_chunk_size_does_not_crash() -> None:
    chunks = chunk_text("x" * 500, chunk_size=100, overlap=150)

    assert chunks
