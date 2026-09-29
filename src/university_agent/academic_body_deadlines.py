"""Conservative deterministic parsing of academic announcement bodies."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class AcademicBodyDeadline:
    """One explicit action and deadline parsed from an announcement body."""

    title: str
    due_at: datetime
    date_text: str


_MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
    "gener": 1,
    "febrer": 2,
    "març": 3,
    "maig": 5,
    "juny": 6,
    "juliol": 7,
    "agost": 8,
    "setembre": 9,
    "desembre": 12,
}

_WEEKDAY = (
    r"(?:lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|"
    r"dilluns|dimarts|dimecres|dijous|divendres|dissabte|diumenge)"
)
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE = rf"""
    (?P<date>
        (?:{_WEEKDAY}\s*,?\s*)?
        (?:d[ií]a\s+)?
        (?P<day>[0-9]{{1,2}})\s+
        (?:de\s+|d['’]\s*)
        (?P<month>{_MONTH})
        (?:\s+de\s+(?P<year>[0-9]{{4}}))?
        \s*,?\s*
        (?:(?:a\s+las|a\s+les)\s+)?
        (?P<hour>[0-9]{{1,2}}):(?P<minute>[0-9]{{2}})
    )
"""
_DEADLINE_PATTERN = re.compile(
    rf"""
    (?P<marker>
        antes\s+(?:del|de\s+la|de)\s+(?:(?:próxim[oa])\s+)?|
        abans\s+(?:del|de\s+la|de)\s+(?:(?:proper[ae]?)\s+)?|
        fecha\s+l[ií]mite\s*:\s*|
        data\s+l[ií]mit\s*:\s*|
        vence\s+el\s+|
        venç\s+el\s+|
        deadline\s*:\s*
    )
    {_DATE}
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ACTION_VERB = (
    r"(?:subir|subid|entregar|entregad|completar|completad|rellenar|rellenad|"
    r"responder|responded|enviar|enviad|pujar|pugeu|lliurar|lliureu|completeu|"
    r"omplir|ompliu|respondre|responeu|envieu|upload|submit|complete|fill\s+in)"
)
_ACTION_PATTERN = re.compile(
    rf"\b(?P<action>{_ACTION_VERB}\b.+)",
    re.IGNORECASE,
)
_ACTION_SEPARATOR = re.compile(
    rf"\s+(?:y|i|and)\s+(?={_ACTION_VERB}\b)",
    re.IGNORECASE,
)
_SEGMENT_SEPARATOR = re.compile(r"\n+|(?<=[.!?])\s+|\s*;\s*")
_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
_EXAMPLE_PATTERN = re.compile(
    r"\b(?:ejemplo|example|per\s+exemple)\b",
    re.IGNORECASE,
)
_QUOTED_BOUNDARY_PATTERN = re.compile(
    r"^(?:-{2,}\s*(?:original message|forwarded message).*-{2,}|"
    r"missatge original:|mensaje original:)$",
    re.IGNORECASE,
)


def parse_academic_body_deadlines(
    body: str,
    *,
    message_at: datetime | None,
) -> list[AcademicBodyDeadline]:
    """Return explicit actionable deadlines from a plain-text email body.

    A missing year is resolved to the first matching future calendar date
    relative to ``message_at``. The system clock is never consulted.
    """
    if not body.strip():
        return []

    text = _sanitize_body(body)
    results: list[AcademicBodyDeadline] = []
    seen: set[tuple[str, datetime]] = set()

    for match in _DEADLINE_PATTERN.finditer(text):
        due_at = _parse_due_at(match, message_at=message_at)
        if due_at is None:
            continue

        before = text[: match.start()]
        after = text[match.end() :]
        actions = _nearby_actions(before, after)
        for title in actions:
            key = (" ".join(title.casefold().split()), due_at)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                AcademicBodyDeadline(
                    title=title,
                    due_at=due_at,
                    date_text=" ".join(match.group("date").split()),
                )
            )

    results.sort(key=lambda result: (result.due_at, result.title.casefold()))
    return results


def _sanitize_body(body: str) -> str:
    normalized = unicodedata.normalize("NFC", body).replace("\r\n", "\n")
    lines: list[str] = []
    for raw_line in normalized.replace("\r", "\n").split("\n"):
        line = " ".join(raw_line.split())
        if _QUOTED_BOUNDARY_PATTERN.fullmatch(line):
            break
        if line.startswith(">"):
            continue
        if line and _URL_PATTERN.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _nearby_actions(before: str, after: str) -> list[str]:
    before_block = before.rsplit("\n\n", maxsplit=1)[-1]
    next_deadline = _DEADLINE_PATTERN.search(after)
    if next_deadline is not None:
        after = after[: next_deadline.start()]
    before_segments = _SEGMENT_SEPARATOR.split(before_block[-1200:])
    after_segments = _SEGMENT_SEPARATOR.split(after[:1200])
    candidates = before_segments[-5:] + after_segments[:12]
    actions: list[str] = []

    for candidate in candidates:
        if _URL_PATTERN.search(candidate) or _EXAMPLE_PATTERN.search(candidate):
            continue
        match = _ACTION_PATTERN.search(candidate)
        if match is None:
            continue
        for action in _ACTION_SEPARATOR.split(match.group("action")):
            title = action.strip(" \t\n-–—:,.")
            if 2 <= len(title) <= 240:
                actions.append(title)

    return actions


def _parse_due_at(
    match: re.Match[str],
    *,
    message_at: datetime | None,
) -> datetime | None:
    month = _MONTHS.get(match.group("month").casefold())
    if month is None:
        return None

    year_text = match.group("year")
    if year_text is not None:
        return _build_datetime(
            year=int(year_text),
            month=month,
            day=int(match.group("day")),
            hour=int(match.group("hour")),
            minute=int(match.group("minute")),
        )

    if message_at is None:
        return None
    reference = message_at.replace(tzinfo=None)
    candidate = _build_datetime(
        year=reference.year,
        month=month,
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
    )
    if candidate is None:
        return None
    if candidate < reference:
        if candidate.date() == reference.date():
            return None
        candidate = _build_datetime(
            year=reference.year + 1,
            month=month,
            day=int(match.group("day")),
            hour=int(match.group("hour")),
            minute=int(match.group("minute")),
        )
    if candidate is None or candidate > reference + timedelta(days=366):
        return None
    return candidate


def _build_datetime(
    *,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
) -> datetime | None:
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None
