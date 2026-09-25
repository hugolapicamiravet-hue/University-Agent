"""Bounded process-local cache for extracted course materials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from university_agent.local_materials import LocalMaterial
from university_agent.material_text import (
    ExtractedDocument,
    MaterialTextExtractionError,
    _extract_discovered_text,
)


@dataclass(frozen=True, slots=True)
class _FileSignature:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    signature: _FileSignature
    document: ExtractedDocument


class LocalMaterialCache:
    """Cache extracted documents with metadata-based invalidation."""

    def __init__(self, *, max_entries: int = 256) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._entries: dict[Path, _CacheEntry] = {}

    def _get_document(self, material: LocalMaterial) -> ExtractedDocument:
        """Return one current extracted document, reusing it when unchanged."""
        identity, signature = _material_identity_and_signature(material)
        cached = self._entries.get(identity)
        if cached is not None and cached.signature == signature:
            return cached.document

        document = _extract_discovered_text(material)
        if identity not in self._entries:
            self._evict_if_full()
        self._entries[identity] = _CacheEntry(
            signature=signature,
            document=document,
        )
        return document

    def clear(self) -> None:
        """Discard every process-local cached document."""
        self._entries.clear()

    def _evict_if_full(self) -> None:
        if len(self._entries) < self._max_entries:
            return
        oldest_identity = next(iter(self._entries))
        del self._entries[oldest_identity]


def _material_identity_and_signature(
    material: LocalMaterial,
) -> tuple[Path, _FileSignature]:
    try:
        metadata = material.path.stat()
        identity = material.path.resolve(strict=True)
    except OSError as error:
        raise MaterialTextExtractionError(
            "could not inspect material for extraction"
        ) from error

    return identity, _FileSignature(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )
