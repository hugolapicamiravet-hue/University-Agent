"""Compose Gmail search with notification parsing for upcoming deadlines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime

from university_agent.academic_body_deadlines import parse_academic_body_deadlines
from university_agent.academic_notifications import (
    NotificationCategory,
    ParsedNotification,
    parse_notification_subject,
)
from university_agent.connectors.gmail import GmailConnector
from university_agent.course_notices import parse_course_notice_subject

_DEADLINE_QUERY = 'subject:"Venciment el"'


@dataclass(frozen=True, slots=True)
class DetectedDeadline:
    """A parsed future deadline together with its Gmail identity."""

    message_id: str
    thread_id: str
    notification: ParsedNotification


def find_upcoming_deadlines(
    connector: GmailConnector,
    *,
    now: datetime,
    max_results: int = 100,
) -> list[DetectedDeadline]:
    """Return detected Gmail deadlines strictly later than ``now``."""
    if now.tzinfo is not None:
        raise ValueError("now must be timezone-naive")

    academic_year = _academic_year_for(now)
    messages = connector.search_messages(
        f'{{{_DEADLINE_QUERY} subject:"{academic_year}"}}',
        max_results=max_results,
    )
    seen_message_ids: set[str] = set()
    unique_messages = []
    for message in messages:
        if message["id"] in seen_message_ids:
            continue
        seen_message_ids.add(message["id"])
        unique_messages.append(message)

    seen_deadlines: set[tuple[str, str, datetime]] = set()
    subject_deadline_keys: set[tuple[str, datetime]] = set()
    deadlines: list[DetectedDeadline] = []

    for message in unique_messages:
        message_id = message["id"]
        notification = parse_notification_subject(message["subject"])
        if (
            notification.category is NotificationCategory.DEADLINE
            and notification.event_at is not None
            and notification.event_at > now
        ):
            subject_deadline_keys.add(
                (
                    " ".join((notification.title or "").casefold().split()),
                    notification.event_at,
                )
            )
            _append_deadline(
                deadlines,
                seen_deadlines,
                message_id=message_id,
                thread_id=message["thread_id"],
                notification=notification,
            )

        course_notice = parse_course_notice_subject(message["subject"])
        if course_notice is None or course_notice.academic_year != academic_year:
            continue
        body = connector.get_plain_text_body(message_id)
        for body_deadline in parse_academic_body_deadlines(
            body,
            message_at=_parse_message_datetime(message["date"]),
        ):
            if body_deadline.due_at <= now:
                continue
            if (
                " ".join(body_deadline.title.casefold().split()),
                body_deadline.due_at,
            ) in subject_deadline_keys:
                continue
            _append_deadline(
                deadlines,
                seen_deadlines,
                message_id=message_id,
                thread_id=message["thread_id"],
                notification=ParsedNotification(
                    category=NotificationCategory.DEADLINE,
                    raw_subject=message["subject"],
                    title=body_deadline.title,
                    date_text=body_deadline.date_text,
                    event_at=body_deadline.due_at,
                ),
            )

    deadlines.sort(
        key=lambda deadline: (
            deadline.notification.event_at,
            deadline.message_id,
            deadline.notification.title or "",
        )
    )
    return deadlines


def find_upcoming_deadlines_until(
    connector: GmailConnector,
    *,
    now: datetime,
    until: datetime,
    max_results: int = 100,
) -> list[DetectedDeadline]:
    """Return deadlines strictly after ``now`` and no later than ``until``."""
    if now.tzinfo is not None:
        raise ValueError("now must be timezone-naive")
    if until.tzinfo is not None:
        raise ValueError("until must be timezone-naive")
    if until <= now:
        raise ValueError("until must be later than now")

    deadlines = find_upcoming_deadlines(
        connector,
        now=now,
        max_results=max_results,
    )
    return [
        deadline
        for deadline in deadlines
        if deadline.notification.event_at is not None
        and deadline.notification.event_at <= until
    ]


def _append_deadline(
    deadlines: list[DetectedDeadline],
    seen: set[tuple[str, str, datetime]],
    *,
    message_id: str,
    thread_id: str,
    notification: ParsedNotification,
) -> None:
    if notification.title is None or notification.event_at is None:
        return
    normalized_title = " ".join(notification.title.casefold().split())
    key = (message_id, normalized_title, notification.event_at)
    if key in seen:
        return
    seen.add(key)
    deadlines.append(
        DetectedDeadline(
            message_id=message_id,
            thread_id=thread_id,
            notification=notification,
        )
    )


def _academic_year_for(now: datetime) -> str:
    start_year = now.year if now.month >= 8 else now.year - 1
    return f"{start_year:04d}-{start_year + 1:04d}"


def _parse_message_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed.replace(tzinfo=None)
