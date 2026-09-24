"""Extract deterministic text from discovered local course materials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from university_agent.local_materials import LocalMaterial, LocalMaterialsSource


class UnsupportedMaterialFormatError(ValueError):
    """Raised when text extraction does not support a material format."""


class MaterialTextExtractionError(Exception):
    """Raised when a supported material cannot be read or extracted."""


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Text and local provenance extracted from one course material."""

    course_name: str
    relative_path: Path
    format: str
    text: str
    page_texts: tuple[str, ...] | None = None

    @property
    def page_count(self) -> int | None:
        """Return the number of PDF pages, or None for non-paged text."""
        if self.page_texts is None:
            return None
        return len(self.page_texts)


def extract_text(
    source: LocalMaterialsSource,
    material: LocalMaterial,
) -> ExtractedDocument:
    """Extract text from a material currently discoverable in ``source``."""
    discovered_material = _rediscover_material(source, material)
    return _extract_discovered_text(discovered_material)


def _extract_discovered_text(material: LocalMaterial) -> ExtractedDocument:
    """Extract a material returned directly by LocalMaterialsSource."""
    material_format = material.path.suffix.casefold()

    if material_format == ".txt":
        return _extract_txt(material)
    if material_format == ".pdf":
        return _extract_pdf(material)

    display_format = material_format or "<none>"
    raise UnsupportedMaterialFormatError(
        f"unsupported material format: {display_format}"
    )


def _rediscover_material(
    source: LocalMaterialsSource,
    material: LocalMaterial,
) -> LocalMaterial:
    try:
        discovered_materials = source.list_materials(material.course_name)
    except ValueError as error:
        raise ValueError("material does not belong to source") from error

    for discovered_material in discovered_materials:
        if discovered_material == material:
            return discovered_material

    raise ValueError("material does not belong to source")


def _extract_txt(material: LocalMaterial) -> ExtractedDocument:
    try:
        text = material.path.read_text(encoding="utf-8")
    except Exception as error:
        raise MaterialTextExtractionError("could not extract TXT text") from error

    return ExtractedDocument(
        course_name=material.course_name,
        relative_path=material.relative_path,
        format=".txt",
        text=text,
    )


def _extract_pdf(material: LocalMaterial) -> ExtractedDocument:
    try:
        reader = PdfReader(material.path)
        page_texts = tuple(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise MaterialTextExtractionError("could not extract PDF text") from error

    return ExtractedDocument(
        course_name=material.course_name,
        relative_path=material.relative_path,
        format=".pdf",
        text="\n\n".join(page_texts),
        page_texts=page_texts,
    )
