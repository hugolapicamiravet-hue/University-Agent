"""Deterministic host-owned calendar windows for application operations."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def resolve_remaining_week_window(
    *,
    now: datetime,
    timezone_name: str,
) -> tuple[datetime, datetime]:
    """Return naïve local bounds from ``now`` through the ISO week end.

    The host supplies both a trusted timezone-aware current time and an IANA
    timezone name. The returned bounds are local wall-clock values compatible
    with the timezone-naïve dates parsed from current academic notifications.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not timezone_name or timezone_name != timezone_name.strip():
        raise ValueError("timezone_name must be a non-empty IANA timezone name")

    local_now = now.astimezone(ZoneInfo(timezone_name))
    days_until_sunday = 6 - local_now.weekday()
    week_end = (local_now + timedelta(days=days_until_sunday)).replace(
        hour=23,
        minute=59,
        second=59,
        microsecond=999999,
    )
    if local_now >= week_end:
        raise ValueError("no time remains in the current week")

    return local_now.replace(tzinfo=None), week_end.replace(tzinfo=None)
