"""Deterministic parsing for course-prefixed university email subjects."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedCourseNotice:
    """Explicit fields extracted from a supported course-notice subject."""

    raw_subject: str
    course_codes: tuple[str, ...]
    academic_year: str
    notice_text: str


_COURSE_CODE_PATTERN = r"(?:EI|MT)[0-9]{4}"
_COURSE_NOTICE_PATTERN = re.compile(
    rf"^(?P<course_codes>{_COURSE_CODE_PATTERN}"
    rf"(?:-{_COURSE_CODE_PATTERN})*)-"
    r"(?P<academic_year>"
    r"(?P<start_year>[0-9]{4})-(?P<end_year>[0-9]{4})"
    r")"
    r"\s*:\s*"
    r"(?P<notice_text>.+)$"
)


def parse_course_notice_subject(subject: str) -> ParsedCourseNotice | None:
    """Parse a supported course-prefixed subject, or return None."""
    raw_subject = subject
    normalized_subject = unicodedata.normalize("NFC", " ".join(subject.split()))
    match = _COURSE_NOTICE_PATTERN.fullmatch(normalized_subject)
    if not match:
        return None

    start_year = int(match.group("start_year"))
    end_year = int(match.group("end_year"))
    if end_year != start_year + 1:
        return None

    return ParsedCourseNotice(
        raw_subject=raw_subject,
        course_codes=tuple(match.group("course_codes").split("-")),
        academic_year=match.group("academic_year"),
        notice_text=match.group("notice_text"),
    )
