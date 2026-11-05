"""Split long documents into overlapping, paragraph-aware chunks.

all-MiniLM-L6-v2 truncates input at 256 word-pieces, so chunks are kept to
roughly 150 words: long enough for context, short enough to be fully embedded.
"""
from __future__ import annotations

import re


def chunk_text(text: str, chunk_words: int = 150, overlap_words: int = 30) -> list[str]:
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not text:
        return []

    step = chunk_words - overlap_words
    chunks: list[str] = []
    current: list[str] = []

    for para in re.split(r"\n\s*\n", text):
        words = para.split()
        if not words:
            continue

        if len(words) > chunk_words:
            # Oversized paragraph: flush, then slide a window across it.
            if current:
                chunks.append(" ".join(current))
            windows = [words[s:s + chunk_words] for s in range(0, len(words), step)
                       if s == 0 or s + overlap_words < len(words)]
            chunks.extend(" ".join(w) for w in windows[:-1])
            current = windows[-1]          # last window may still absorb the next paragraph
            continue

        if current and len(current) + len(words) > chunk_words:
            chunks.append(" ".join(current))
            current = current[-overlap_words:] if overlap_words else []
        current.extend(words)

    if current and (not chunks or len(current) > overlap_words):
        chunks.append(" ".join(current))
    return chunks
