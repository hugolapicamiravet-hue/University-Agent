"""Transparent deterministic lexical search over local material chunks."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from university_agent.material_chunks import MaterialChunk

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class MaterialSearchResult:
    """One lexically ranked material chunk."""

    chunk: MaterialChunk
    score: float


def search_chunks(
    chunks: Iterable[MaterialChunk],
    query: str,
    *,
    limit: int = 10,
) -> list[MaterialSearchResult]:
    """Return chunks matching normalized query terms in deterministic order."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if limit < 1:
        raise ValueError("limit must be positive")

    query_terms = tuple(dict.fromkeys(_tokenize(query)))
    if not query_terms:
        raise ValueError("query must contain a searchable term")

    ranked: list[tuple[int, int, MaterialSearchResult]] = []
    for chunk in chunks:
        token_counts = Counter(_tokenize(chunk.text))
        distinct_matches = sum(term in token_counts for term in query_terms)
        if distinct_matches == 0:
            continue

        occurrences = sum(token_counts[term] for term in query_terms)
        score = distinct_matches + occurrences / (occurrences + 1)
        ranked.append(
            (
                distinct_matches,
                occurrences,
                MaterialSearchResult(chunk=chunk, score=score),
            )
        )

    ranked.sort(key=_ranking_key)
    return [result for _, _, result in ranked[:limit]]


def _tokenize(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def _ranking_key(
    ranked_result: tuple[int, int, MaterialSearchResult],
) -> tuple[int, int, str, str, int, int]:
    distinct_matches, occurrences, result = ranked_result
    chunk = result.chunk
    page_number = chunk.page_number if chunk.page_number is not None else -1
    return (
        -distinct_matches,
        -occurrences,
        chunk.course_name,
        chunk.relative_path.as_posix(),
        page_number,
        chunk.chunk_index,
    )
