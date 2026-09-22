"""Read recent course-prefixed notices from Gmail."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from university_agent.connectors.gmail import GmailConnector
from university_agent.course_notices import (
    ParsedCourseNotice,
    parse_course_notice_subject,
)

_COURSE_CODE_PATTERN = re.compile(r"(?:EI|MT)[0-9]{4}")
_ACADEMIC_YEAR_PATTERN = re.compile(
    r"(?P<start_year>[0-9]{4})-(?P<end_year>[0-9]{4})"
)


@dataclass(frozen=True, slots=True)
class DetectedCourseNotice:
    """A parsed course notice together with its Gmail provenance."""

    message_id: str
    thread_id: str
    message_date: str
    notice: ParsedCourseNotice


def find_recent_course_notices(
    connector: GmailConnector,
    *,
    course_code: str,
    academic_year: str,
    lookback_days: int = 120,
    max_results: int = 100,
) -> list[DetectedCourseNotice]:
    """Return recent parsed notices explicitly prefixed by a course code."""
    if not _COURSE_CODE_PATTERN.fullmatch(course_code):
        raise ValueError("course_code must match (?:EI|MT)[0-9]{4}")

    year_match = _ACADEMIC_YEAR_PATTERN.fullmatch(academic_year)
    if (
        year_match is None
        or int(year_match.group("end_year"))
        != int(year_match.group("start_year")) + 1
    ):
        raise ValueError("academic_year must contain consecutive YYYY-YYYY years")

    if lookback_days < 1:
        raise ValueError("lookback_days must be at least 1")

    query = f'newer_than:{lookback_days}d subject:"{academic_year}"'
    messages = connector.search_messages(query, max_results=max_results)
    seen_message_ids: set[str] = set()
    results: list[DetectedCourseNotice] = []

    for message in messages:
        message_id = message["id"]
        if message_id in seen_message_ids:
            continue
        seen_message_ids.add(message_id)

        notice = parse_course_notice_subject(message["subject"])
        if (
            notice is not None
            and notice.academic_year == academic_year
            and course_code in notice.course_codes
        ):
            results.append(
                DetectedCourseNotice(
                    message_id=message_id,
                    thread_id=message["thread_id"],
                    message_date=message["date"],
                    notice=notice,
                )
            )

    dated_results: list[tuple[datetime, DetectedCourseNotice]] = []
    undated_results: list[DetectedCourseNotice] = []
    for result in results:
        parsed_date = _parse_message_date(result.message_date)
        if parsed_date is None:
            undated_results.append(result)
        else:
            dated_results.append((parsed_date, result))

    dated_results.sort(key=lambda item: item[1].message_id)
    dated_results.sort(key=lambda item: item[0], reverse=True)
    undated_results.sort(key=lambda result: result.message_id)

    return [result for _, result in dated_results] + undated_results


def _parse_message_date(value: str) -> datetime | None:
    if not value:
        return None

    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)
