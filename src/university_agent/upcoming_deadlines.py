"""Compose Gmail search with notification parsing for upcoming deadlines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from university_agent.academic_notifications import (
    NotificationCategory,
    ParsedNotification,
    parse_notification_subject,
)
from university_agent.connectors.gmail import GmailConnector

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

    messages = connector.search_messages(
        _DEADLINE_QUERY,
        max_results=max_results,
    )
    seen_message_ids: set[str] = set()
    deadlines: list[DetectedDeadline] = []

    for message in messages:
        message_id = message["id"]
        if message_id in seen_message_ids:
            continue
        seen_message_ids.add(message_id)

        notification = parse_notification_subject(message["subject"])
        if (
            notification.category is NotificationCategory.DEADLINE
            and notification.event_at is not None
            and notification.event_at > now
        ):
            deadlines.append(
                DetectedDeadline(
                    message_id=message_id,
                    thread_id=message["thread_id"],
                    notification=notification,
                )
            )

    deadlines.sort(
        key=lambda deadline: (
            deadline.notification.event_at,
            deadline.message_id,
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
