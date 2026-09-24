"""Discover user-managed course materials on the local filesystem."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LocalCourse:
    """An immediate course directory beneath the configured materials root."""

    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class LocalMaterial:
    """A material file discovered beneath one local course directory."""

    course_name: str
    relative_path: Path
    path: Path
    filename: str
    extension: str


class LocalMaterialsSource:
    """Discover courses and files without reading their contents."""

    def __init__(
        self,
        root: str | Path,
        *,
        excluded_relative_paths: Iterable[str | Path] = (),
    ) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError("materials root does not exist")
        if not root_path.is_dir():
            raise NotADirectoryError("materials root is not a directory")

        self.root = root_path.resolve()

        self._excluded_relative_paths = frozenset(
            _validate_excluded_relative_path(path)
            for path in excluded_relative_paths
        )

    def list_courses(self) -> list[LocalCourse]:
        """Return visible, non-symlinked immediate course directories."""
        courses = [
            LocalCourse(name=entry.name, path=entry)
            for entry in self.root.iterdir()
            if not entry.name.startswith(".")
            and not entry.is_symlink()
            and entry.is_dir()
        ]
        return sorted(courses, key=lambda course: course.name)

    def list_materials(self, course: str) -> list[LocalMaterial]:
        """Return files for one unambiguous normalized course lookup."""
        lookup_key = _course_lookup_key(course)
        matching_courses = [
            discovered
            for discovered in self.list_courses()
            if _course_lookup_key(discovered.name) == lookup_key
        ]
        if not matching_courses:
            raise ValueError("unknown course")
        if len(matching_courses) > 1:
            raise ValueError("ambiguous course")
        discovered_course = matching_courses[0]

        materials: list[LocalMaterial] = []
        directories = [discovered_course.path]

        while directories:
            directory = directories.pop()
            for entry in directory.iterdir():
                if entry.name.startswith(".") or entry.is_symlink():
                    continue
                if entry.is_dir():
                    relative_directory = entry.relative_to(
                        discovered_course.path
                    )
                    if relative_directory not in self._excluded_relative_paths:
                        directories.append(entry)
                elif entry.is_file():
                    relative_path = entry.relative_to(discovered_course.path)
                    materials.append(
                        LocalMaterial(
                            course_name=discovered_course.name,
                            relative_path=relative_path,
                            path=entry,
                            filename=entry.name,
                            extension=entry.suffix,
                        )
                    )

        return sorted(
            materials,
            key=lambda material: material.relative_path.as_posix(),
        )


def _validate_excluded_relative_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(
            "excluded paths must be non-empty relative paths without parent traversal"
        )
    return path


def _course_lookup_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().casefold()
