"""Deterministically split extracted local materials into retrievable chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from university_agent.material_text import ExtractedDocument

_MIN_MAX_CHARS = 100
_PARAGRAPH_SEPARATOR = "\n\n"
_PARAGRAPH_BREAK_PATTERN = re.compile(r"\r?\n[^\S\r\n]*\r?\n+")


@dataclass(frozen=True, slots=True)
class MaterialChunk:
    """A deterministic text chunk with local material provenance."""

    course_name: str
    relative_path: Path
    format: str
    text: str
    page_number: int | None
    chunk_index: int


def chunk_document(
    document: ExtractedDocument,
    *,
    max_chars: int = 2000,
) -> list[MaterialChunk]:
    """Split one extracted document without crossing PDF page boundaries."""
    if max_chars < _MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be at least {_MIN_MAX_CHARS}")

    if document.page_texts is None:
        text_sections = ((None, document.text),)
    else:
        text_sections = tuple(enumerate(document.page_texts, start=1))

    chunks: list[MaterialChunk] = []
    for page_number, text in text_sections:
        for chunk_text in _split_text(text, max_chars=max_chars):
            chunks.append(
                MaterialChunk(
                    course_name=document.course_name,
                    relative_path=document.relative_path,
                    format=document.format,
                    text=chunk_text,
                    page_number=page_number,
                    chunk_index=len(chunks),
                )
            )

    return chunks


def _split_text(text: str, *, max_chars: int) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_BREAK_PATTERN.split(text)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    pending_paragraphs: list[str] = []
    pending_length = 0

    def flush_pending() -> None:
        nonlocal pending_length
        if pending_paragraphs:
            chunks.append(_PARAGRAPH_SEPARATOR.join(pending_paragraphs))
            pending_paragraphs.clear()
            pending_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush_pending()
            chunks.extend(_split_long_paragraph(paragraph, max_chars=max_chars))
            continue

        separator_length = len(_PARAGRAPH_SEPARATOR) if pending_paragraphs else 0
        if pending_length + separator_length + len(paragraph) <= max_chars:
            pending_paragraphs.append(paragraph)
            pending_length += separator_length + len(paragraph)
        else:
            flush_pending()
            pending_paragraphs.append(paragraph)
            pending_length = len(paragraph)

    flush_pending()
    return chunks


def _split_long_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    pending_words: list[str] = []
    pending_length = 0

    def flush_pending() -> None:
        nonlocal pending_length
        if pending_words:
            chunks.append(" ".join(pending_words))
            pending_words.clear()
            pending_length = 0

    for word in paragraph.split():
        if len(word) > max_chars:
            flush_pending()
            chunks.extend(
                word[offset : offset + max_chars]
                for offset in range(0, len(word), max_chars)
            )
            continue

        separator_length = 1 if pending_words else 0
        if pending_length + separator_length + len(word) <= max_chars:
            pending_words.append(word)
            pending_length += separator_length + len(word)
        else:
            flush_pending()
            pending_words.append(word)
            pending_length = len(word)

    flush_pending()
    return chunks
