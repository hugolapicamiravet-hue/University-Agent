"""Public document-focused knowledge engine for academic materials."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from university_agent.local_material_cache import LocalMaterialCache
from university_agent.local_material_search import search_local_materials
from university_agent.local_materials import LocalMaterialsSource


@dataclass(frozen=True, slots=True)
class MaterialReference:
    """One discoverable material without its absolute filesystem location."""

    course_name: str
    relative_path: str
    filename: str
    extension: str


@dataclass(frozen=True, slots=True)
class MaterialPassage:
    """One ranked academic passage with relative source provenance."""

    course_name: str
    relative_path: str
    page_number: int | None
    text: str
    score: float


class AcademicKnowledgeEngine:
    """Discover and lexically retrieve evidence from local academic documents."""

    def __init__(
        self,
        materials: str | Path | LocalMaterialsSource,
        *,
        excluded_relative_paths: Iterable[str | Path] = (),
        max_chunk_chars: int = 2000,
        cache: LocalMaterialCache | None = None,
    ) -> None:
        if max_chunk_chars < 100:
            raise ValueError("max_chunk_chars must be at least 100")
        if isinstance(materials, LocalMaterialsSource):
            if tuple(excluded_relative_paths):
                raise ValueError(
                    "excluded_relative_paths cannot be used with an existing source"
                )
            source = materials
        else:
            source = LocalMaterialsSource(
                materials,
                excluded_relative_paths=excluded_relative_paths,
            )

        self._source = source
        self._max_chunk_chars = max_chunk_chars
        self._cache = cache or LocalMaterialCache()

    def list_courses(self) -> list[str]:
        """Return configured course names without filesystem locations."""
        return [course.name for course in self._source.list_courses()]

    def list_materials(self, course: str) -> list[MaterialReference]:
        """Return relative references for materials in one course."""
        return [
            MaterialReference(
                course_name=material.course_name,
                relative_path=material.relative_path.as_posix(),
                filename=material.filename,
                extension=material.extension,
            )
            for material in self._source.list_materials(course)
        ]

    def search(
        self,
        query: str,
        *,
        course: str | None = None,
        limit: int = 10,
    ) -> list[MaterialPassage]:
        """Return lexically ranked passages using the established algorithm."""
        results = search_local_materials(
            self._source,
            query=query,
            course_name=course,
            limit=limit,
            max_chunk_chars=self._max_chunk_chars,
            cache=self._cache,
        )
        return [
            MaterialPassage(
                course_name=result.chunk.course_name,
                relative_path=result.chunk.relative_path.as_posix(),
                page_number=result.chunk.page_number,
                text=result.chunk.text,
                score=result.score,
            )
            for result in results
        ]

    def clear_cache(self) -> None:
        """Discard extracted documents retained by this engine instance."""
        self._cache.clear()
