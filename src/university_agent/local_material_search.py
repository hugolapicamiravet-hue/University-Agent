"""Compose local discovery, extraction, chunking, and lexical search."""

from __future__ import annotations

from university_agent.local_material_cache import LocalMaterialCache
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
    cache: LocalMaterialCache | None = None,
) -> list[MaterialSearchResult]:
    """Search supported local materials for one exact course or all courses."""
    search_chunks((), query, limit=limit)
    if max_chunk_chars < 100:
        raise ValueError("max_chunk_chars must be at least 100")

    course_names = (
        [course.name for course in source.list_courses()]
        if course_name is None
        else [course_name]
    )

    chunks: list[MaterialChunk] = []
    for selected_course in course_names:
        for material in source.list_materials(selected_course):
            try:
                document = (
                    _extract_discovered_text(material)
                    if cache is None
                    else cache._get_document(material)
                )
            except UnsupportedMaterialFormatError:
                continue
            chunks.extend(
                chunk_document(document, max_chars=max_chunk_chars)
            )

    return search_chunks(chunks, query, limit=limit)
