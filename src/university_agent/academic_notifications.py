"""Deterministic parsing for observed UJI academic notification subjects."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class NotificationCategory(StrEnum):
    """Notification categories proven by observed subject templates."""

    DEADLINE = "deadline"
    TASK_OVERDUE = "task_overdue"
    ACTIVITY_OPENS = "activity_opens"
    TASK_SUBMISSION_CONFIRMED = "task_submission_confirmed"
    UPCOMING_TASKS_SUMMARY = "upcoming_tasks_summary"
    ACCOUNT_LOGIN_NOTICE = "account_login_notice"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ParsedNotification:
    """Structured fields extracted conservatively from a notification subject."""

    category: NotificationCategory
    raw_subject: str
    title: str | None = None
    date_text: str | None = None
    event_at: datetime | None = None
    days_until_due: int | None = None


_WEEKDAYS = {
    "dilluns",
    "dimarts",
    "dimecres",
    "dijous",
    "divendres",
    "dissabte",
    "diumenge",
}

_MONTHS = {
    "gener": 1,
    "febrer": 2,
    "març": 3,
    "abril": 4,
    "maig": 5,
    "juny": 6,
    "juliol": 7,
    "agost": 8,
    "setembre": 9,
    "octubre": 10,
    "novembre": 11,
    "desembre": 12,
}

_LOCALIZED_DATE_PATTERN = (
    r"(?P<date>\S+,\s+\d{1,2}\s+de\s+\S+\s+\d{4},\s+\S+?)"
)
_DEADLINE_PATTERN = re.compile(
    rf"^Venciment el {_LOCALIZED_DATE_PATTERN}:\s+(?P<title>.+)$"
)
_ACTIVITY_OPENS_PATTERN = re.compile(
    rf"^Opens on {_LOCALIZED_DATE_PATTERN}:\s+(?P<title>.+)$"
)
_TASK_OVERDUE_PATTERN = re.compile(
    r"^Tasca vençuda:\s+(?P<title>.+)\s+"
    r"\(fecha límite (?P<date>\d{1,2}/\d{1,2}/\d{4} a las \S+)\)$"
)
_TASK_OVERDUE_WITHOUT_DATE_PATTERN = re.compile(
    r"^Tasca vençuda:\s+(?P<title>.+)$"
)
_TASK_SUBMISSION_PATTERN = re.compile(
    r"^Heu realitzat la tramesa de la tasca\s+(?P<title>.+)$"
)
_UPCOMING_TASKS_PATTERN = re.compile(
    r"^Teniu tasques que vencen en (?P<days>\d+) dies$"
)
_ACCOUNT_LOGIN_SUBJECT = "Inici de sessió nou del vostre compte Aula Virtual UJI"

_LOCALIZED_DATE_VALUE_PATTERN = re.compile(
    r"^(?P<weekday>\S+),\s+"
    r"(?P<day>\d{1,2})\s+de\s+"
    r"(?P<month>\S+)\s+"
    r"(?P<year>\d{4}),\s+"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})$"
)
_NUMERIC_DATE_VALUE_PATTERN = re.compile(
    r"^(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})"
    r"\s+a las\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})$"
)


def parse_notification_subject(subject: str) -> ParsedNotification:
    """Parse one known notification template, returning UNKNOWN otherwise."""
    raw_subject = subject
    normalized_subject = unicodedata.normalize("NFC", " ".join(subject.split()))

    match = _DEADLINE_PATTERN.fullmatch(normalized_subject)
    if match:
        date_text = match.group("date")
        return ParsedNotification(
            category=NotificationCategory.DEADLINE,
            raw_subject=raw_subject,
            title=match.group("title"),
            date_text=date_text,
            event_at=_parse_localized_datetime(date_text),
        )

    match = _TASK_OVERDUE_PATTERN.fullmatch(normalized_subject)
    if match:
        date_text = match.group("date")
        return ParsedNotification(
            category=NotificationCategory.TASK_OVERDUE,
            raw_subject=raw_subject,
            title=match.group("title"),
            date_text=date_text,
            event_at=_parse_numeric_datetime(date_text),
        )

    match = _TASK_OVERDUE_WITHOUT_DATE_PATTERN.fullmatch(normalized_subject)
    if match:
        return ParsedNotification(
            category=NotificationCategory.TASK_OVERDUE,
            raw_subject=raw_subject,
            title=match.group("title"),
        )

    match = _ACTIVITY_OPENS_PATTERN.fullmatch(normalized_subject)
    if match:
        date_text = match.group("date")
        return ParsedNotification(
            category=NotificationCategory.ACTIVITY_OPENS,
            raw_subject=raw_subject,
            title=match.group("title"),
            date_text=date_text,
            event_at=_parse_localized_datetime(date_text),
        )

    match = _TASK_SUBMISSION_PATTERN.fullmatch(normalized_subject)
    if match:
        return ParsedNotification(
            category=NotificationCategory.TASK_SUBMISSION_CONFIRMED,
            raw_subject=raw_subject,
            title=match.group("title"),
        )

    match = _UPCOMING_TASKS_PATTERN.fullmatch(normalized_subject)
    if match:
        return ParsedNotification(
            category=NotificationCategory.UPCOMING_TASKS_SUMMARY,
            raw_subject=raw_subject,
            days_until_due=int(match.group("days")),
        )

    if normalized_subject == _ACCOUNT_LOGIN_SUBJECT:
        return ParsedNotification(
            category=NotificationCategory.ACCOUNT_LOGIN_NOTICE,
            raw_subject=raw_subject,
        )

    return ParsedNotification(
        category=NotificationCategory.UNKNOWN,
        raw_subject=raw_subject,
    )


def _parse_localized_datetime(date_text: str) -> datetime | None:
    match = _LOCALIZED_DATE_VALUE_PATTERN.fullmatch(date_text)
    if not match:
        return None

    weekday = match.group("weekday").casefold()
    month = _MONTHS.get(match.group("month").casefold())
    if weekday not in _WEEKDAYS or month is None:
        return None

    return _build_datetime(
        year=match.group("year"),
        month=month,
        day=match.group("day"),
        hour=match.group("hour"),
        minute=match.group("minute"),
    )


def _parse_numeric_datetime(date_text: str) -> datetime | None:
    match = _NUMERIC_DATE_VALUE_PATTERN.fullmatch(date_text)
    if not match:
        return None

    return _build_datetime(
        year=match.group("year"),
        month=match.group("month"),
        day=match.group("day"),
        hour=match.group("hour"),
        minute=match.group("minute"),
    )


def _build_datetime(
    *,
    year: str,
    month: str | int,
    day: str,
    hour: str,
    minute: str,
) -> datetime | None:
    try:
        return datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hour),
            minute=int(minute),
        )
    except ValueError:
        return None
