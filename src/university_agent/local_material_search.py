"""Compose local discovery, extraction, chunking, and lexical search."""

from __future__ import annotations

from university_agent.local_materials import LocalMaterialsSource
from university_agent.material_chunks import MaterialChunk, chunk_document
from university_agent.material_search import MaterialSearchResult, search_chunks
from university_agent.material_text import (
    UnsupportedMaterialFormatError,
    _extract_discovered_text,
)


def search_local_materials(
    source: LocalMaterialsSource,
    *,
    query: str,
    course_name: str | None = None,
    limit: int = 10,
    max_chunk_chars: int = 2000,
) -> list[MaterialSearchResult]:
    """Search supported local materials for one exact course or all courses."""
    search_chunks((), query, limit=limit)
    if max_chunk_chars < 100:
        raise ValueError("max_chunk_chars must be at least 100")

    courses = source.list_courses()
    if course_name is not None:
        matching_courses = [
            course for course in courses if course.name == course_name
        ]
        if not matching_courses:
            raise ValueError("unknown course")
        courses = matching_courses

    chunks: list[MaterialChunk] = []
    for course in courses:
        for material in source.list_materials(course.name):
            try:
                document = _extract_discovered_text(material)
            except UnsupportedMaterialFormatError:
                continue
            chunks.extend(
                chunk_document(document, max_chars=max_chunk_chars)
            )

    return search_chunks(chunks, query, limit=limit)
