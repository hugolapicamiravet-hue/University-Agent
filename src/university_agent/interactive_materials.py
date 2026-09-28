"""Normalize already-captured interactive states into safe local text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


_MAX_STATES = 100
_MAX_STATE_CHARS = 100_000
_MAX_TOTAL_CHARS = 1_000_000
_MAX_HEADING_CHARS = 200
_BLANK_LINE_PATTERN = re.compile(r"\n[\t ]*\n+")


@dataclass(frozen=True, slots=True)
class InteractiveState:
    """Visible academic text captured from one interactive state."""

    title: str
    text: str


def normalize_interactive_material(
    resource_title: str,
    states: Sequence[InteractiveState],
) -> str:
    """Return deterministic UTF-8-ready text for captured interactive states."""
    normalized_resource_title = _normalize_heading(
        resource_title,
        field="resource title",
    )
    if not states:
        raise ValueError("at least one interactive state is required")
    if len(states) > _MAX_STATES:
        raise ValueError(f"interactive states must not exceed {_MAX_STATES}")

    normalized_states: list[tuple[str, list[str]]] = []
    seen_previous_blocks: set[str] = set()
    total_characters = 0

    for state in states:
        normalized_state_title = _normalize_heading(
            state.title,
            field="state title",
        )
        if len(state.text) > _MAX_STATE_CHARS:
            raise ValueError(
                f"state text must not exceed {_MAX_STATE_CHARS} characters"
            )
        total_characters += len(state.text)
        if total_characters > _MAX_TOTAL_CHARS:
            raise ValueError(
                f"total state text must not exceed {_MAX_TOTAL_CHARS} characters"
            )

        blocks = _normalized_blocks(state.text)
        unique_blocks = [
            block
            for block in blocks
            if _is_heading(block) or block not in seen_previous_blocks
        ]
        seen_previous_blocks.update(
            block for block in blocks if not _is_heading(block)
        )
        normalized_states.append((normalized_state_title, unique_blocks))

    document_parts = [f"# {normalized_resource_title}"]
    for state_title, blocks in normalized_states:
        document_parts.append(f"## {state_title}")
        if blocks:
            document_parts.append("\n\n".join(blocks))

    return "\n\n".join(document_parts) + "\n"


def write_interactive_material(
    destination_directory: str | Path,
    filename: str,
    resource_title: str,
    states: Sequence[InteractiveState],
) -> Path:
    """Create one normalized TXT artifact inside an explicit directory."""
    safe_filename = _safe_txt_filename(filename)
    destination = Path(destination_directory)
    if not destination.exists():
        raise FileNotFoundError("destination directory does not exist")
    if not destination.is_dir():
        raise NotADirectoryError("destination is not a directory")

    text = normalize_interactive_material(resource_title, states)
    resolved_destination = destination.resolve()
    output_path = resolved_destination / safe_filename

    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(text)

    return output_path


def _normalize_heading(value: str, *, field: str) -> str:
    normalized = unicodedata.normalize("NFC", " ".join(value.split()))
    if not normalized:
        raise ValueError(f"{field} must not be blank")
    if len(normalized) > _MAX_HEADING_CHARS:
        raise ValueError(
            f"{field} must not exceed {_MAX_HEADING_CHARS} characters"
        )
    return normalized


def _normalized_blocks(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = normalized.strip()
    if not normalized:
        return []
    return [
        block.strip()
        for block in _BLANK_LINE_PATTERN.split(normalized)
        if block.strip()
    ]


def _is_heading(block: str) -> bool:
    first_line = block.split("\n", maxsplit=1)[0]
    return bool(re.match(r"^#{1,6}(?:\s|$)", first_line))


def _safe_txt_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.strip())
    windows_path = PureWindowsPath(normalized)
    if (
        not normalized
        or normalized in {".", ".."}
        or normalized.startswith(".")
        or "/" in normalized
        or "\\" in normalized
        or "\0" in normalized
        or any(ord(character) < 32 for character in normalized)
        or any(character in '<>:"|?*' for character in normalized)
        or windows_path.is_absolute()
        or windows_path.drive
        or _is_windows_reserved_filename(windows_path)
    ):
        raise ValueError("filename must be a safe relative file name")

    path = Path(normalized)
    if not path.suffix:
        return f"{normalized}.txt"
    if path.suffix.casefold() != ".txt":
        raise ValueError("filename must use the .txt extension")
    return f"{normalized[: -len(path.suffix)]}.txt"


def _is_windows_reserved_filename(path: PureWindowsPath) -> bool:
    stem = path.name.split(".", maxsplit=1)[0].rstrip(" .").casefold()
    return stem in {"con", "prn", "aux", "nul"} or bool(
        re.fullmatch(r"(?:com|lpt)[1-9]", stem)
    )
