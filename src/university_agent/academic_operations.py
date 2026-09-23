"""Narrow deterministic application operations for a future agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast

from university_agent.connectors.gmail import GmailConnector
from university_agent.local_material_search import search_local_materials
from university_agent.local_materials import LocalMaterialsSource
from university_agent.recent_course_notices import find_recent_course_notices
from university_agent.upcoming_deadlines import find_upcoming_deadlines_until


@dataclass(frozen=True, slots=True)
class DeadlineResult:
    """Deadline information useful to an agent without Gmail identifiers."""

    title: str
    due_at: datetime


@dataclass(frozen=True, slots=True)
class CourseNoticeResult:
    """Explicit course-notice fields without Gmail identifiers or headers."""

    course_codes: tuple[str, ...]
    academic_year: str
    text: str
    message_date: str


@dataclass(frozen=True, slots=True)
class MaterialPassage:
    """One ranked local passage with citation-oriented relative provenance."""

    course_name: str
    relative_path: str
    page_number: int | None
    text: str
    score: float


def get_deadlines(
    connector: GmailConnector,
    *,
    now: datetime,
    until: datetime,
    max_results: int = 100,
) -> list[DeadlineResult]:
    """Return narrowed deadlines within an explicit time window."""
    deadlines = find_upcoming_deadlines_until(
        connector,
        now=now,
        until=until,
        max_results=max_results,
    )
    return [
        DeadlineResult(
            title=cast(str, deadline.notification.title),
            due_at=cast(datetime, deadline.notification.event_at),
        )
        for deadline in deadlines
    ]


def get_course_notices(
    connector: GmailConnector,
    *,
    course_code: str,
    academic_year: str,
    lookback_days: int = 120,
    max_results: int = 100,
) -> list[CourseNoticeResult]:
    """Return narrowed recent notices for an exact course code and year."""
    notices = find_recent_course_notices(
        connector,
        course_code=course_code,
        academic_year=academic_year,
        lookback_days=lookback_days,
        max_results=max_results,
    )
    return [
        CourseNoticeResult(
            course_codes=notice.notice.course_codes,
            academic_year=notice.notice.academic_year,
            text=notice.notice.notice_text,
            message_date=notice.message_date,
        )
        for notice in notices
    ]


def search_materials(
    source: LocalMaterialsSource,
    *,
    query: str,
    course_name: str | None = None,
    limit: int = 10,
    max_chunk_chars: int = 2000,
) -> list[MaterialPassage]:
    """Return narrowed lexical passages from supported local materials."""
    results = search_local_materials(
        source,
        query=query,
        course_name=course_name,
        limit=limit,
        max_chunk_chars=max_chunk_chars,
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
