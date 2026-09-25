"""Tests for bounded process-local material extraction caching."""

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from university_agent import local_material_cache
from university_agent.agent_tools import AgentToolRuntime
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_material_cache import LocalMaterialCache
from university_agent.local_material_search import search_local_materials
from university_agent.local_materials import LocalMaterialsSource
from university_agent.material_text import MaterialTextExtractionError


class LocalMaterialCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "materials"
        self.alpha = self.root / "Fictional Course Alpha"
        self.beta = self.root / "Fictional Course Beta"
        self.alpha.mkdir(parents=True)
        self.beta.mkdir()
        self.source = LocalMaterialsSource(self.root)
        self.cache = LocalMaterialCache()

    def create_text(
        self,
        course: Path,
        relative_path: str,
        text: str,
    ) -> Path:
        path = course / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def search(
        self,
        query: str,
        *,
        max_chunk_chars: int = 2000,
        source: LocalMaterialsSource | None = None,
    ):
        return search_local_materials(
            self.source if source is None else source,
            query=query,
            limit=100,
            max_chunk_chars=max_chunk_chars,
            cache=self.cache,
        )

    def extraction_patch(self):
        return patch.object(
            local_material_cache,
            "_extract_discovered_text",
            wraps=local_material_cache._extract_discovered_text,
        )

    def test_first_search_extracts_and_repeated_search_reuses_document(self):
        self.create_text(self.alpha, "notes.txt", "fictional shared memory")

        with self.extraction_patch() as extract:
            first = self.search("shared")
            second = self.search("memory")

        self.assertEqual(extract.call_count, 1)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)

    def test_changed_file_size_invalidates_document(self):
        path = self.create_text(self.alpha, "notes.txt", "fictional alpha")

        with self.extraction_patch() as extract:
            self.search("alpha")
            path.write_text("fictional beta content", encoding="utf-8")
            results = self.search("beta")

        self.assertEqual(extract.call_count, 2)
        self.assertEqual(len(results), 1)

    def test_same_size_change_with_new_mtime_invalidates_document(self):
        path = self.create_text(self.alpha, "notes.txt", "fictional alpha")
        original = path.stat()

        with self.extraction_patch() as extract:
            self.search("alpha")
            path.write_text("fictional bravo", encoding="utf-8")
            os.utime(
                path,
                ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000),
            )
            results = self.search("bravo")

        self.assertEqual(extract.call_count, 2)
        self.assertEqual(len(results), 1)

    def test_deleted_file_does_not_return_stale_content(self):
        path = self.create_text(self.alpha, "notes.txt", "fictional removable")
        self.search("removable")

        path.unlink()

        self.assertEqual(self.search("removable"), [])

    def test_replaced_file_invalidates_document(self):
        path = self.create_text(self.alpha, "notes.txt", "fictional original")
        replacement = self.alpha / "replacement.txt"

        with self.extraction_patch() as extract:
            self.search("original")
            replacement.write_text("fictional replaced", encoding="utf-8")
            os.replace(replacement, path)
            results = self.search("replaced")

        self.assertEqual(extract.call_count, 2)
        self.assertEqual(len(results), 1)

    def test_chunk_sizes_are_rebuilt_without_reextracting(self):
        self.create_text(
            self.alpha,
            "long.txt",
            " ".join(["fictional"] * 80),
        )

        with self.extraction_patch() as extract:
            small = self.search("fictional", max_chunk_chars=100)
            large = self.search("fictional", max_chunk_chars=200)

        self.assertEqual(extract.call_count, 1)
        self.assertGreater(len(small), len(large))
        self.assertTrue(all(len(result.chunk.text) <= 100 for result in small))
        self.assertTrue(all(len(result.chunk.text) <= 200 for result in large))

    def test_materials_in_different_courses_do_not_collide(self):
        self.create_text(self.alpha, "notes.txt", "fictional alpha")
        self.create_text(self.beta, "notes.txt", "fictional beta")

        with self.extraction_patch() as extract:
            alpha_results = self.search("alpha")
            beta_results = self.search("beta")

        self.assertEqual(extract.call_count, 2)
        self.assertEqual(
            alpha_results[0].chunk.course_name,
            "Fictional Course Alpha",
        )
        self.assertEqual(
            beta_results[0].chunk.course_name,
            "Fictional Course Beta",
        )

    def test_relative_provenance_is_unchanged(self):
        self.create_text(
            self.alpha,
            "nested/notes.txt",
            "fictional provenance",
        )

        result = self.search("provenance")[0]

        self.assertEqual(result.chunk.relative_path, Path("nested/notes.txt"))
        self.assertFalse(result.chunk.relative_path.is_absolute())
        self.assertFalse(hasattr(result.chunk, "path"))

    def test_corrupt_supported_document_still_raises(self):
        (self.alpha / "corrupt.pdf").write_bytes(b"not a PDF")

        with self.assertLogs("pypdf", level="WARNING"):
            with self.assertRaises(MaterialTextExtractionError):
                self.search("fictional")

    def test_unsupported_formats_and_exclusions_remain_skipped(self):
        self.create_text(self.alpha, "unsupported.docx", "fictional unsupported")
        self.create_text(self.alpha, "visible.txt", "fictional visible")
        self.create_text(self.alpha, "Generated/private.txt", "fictional private")
        source = LocalMaterialsSource(
            self.root,
            excluded_relative_paths=("Generated",),
        )

        with self.extraction_patch() as extract:
            visible = self.search("visible", source=source)

        self.assertEqual(extract.call_count, 2)
        self.assertEqual(len(visible), 1)
        self.assertFalse(
            any(
                "private" in result.chunk.text
                for result in visible
            )
        )

    def test_clear_forces_reextraction(self):
        self.create_text(self.alpha, "notes.txt", "fictional clear")

        with self.extraction_patch() as extract:
            self.search("clear")
            self.cache.clear()
            self.search("clear")

        self.assertEqual(extract.call_count, 2)

    def test_entry_bound_uses_deterministic_fifo_eviction(self):
        self.cache = LocalMaterialCache(max_entries=1)
        self.create_text(self.alpha, "first.txt", "fictional first")
        self.create_text(self.beta, "second.txt", "fictional second")

        with self.extraction_patch() as extract:
            search_local_materials(
                self.source,
                query="first",
                course_name="Fictional Course Alpha",
                cache=self.cache,
            )
            search_local_materials(
                self.source,
                query="second",
                course_name="Fictional Course Beta",
                cache=self.cache,
            )
            search_local_materials(
                self.source,
                query="first",
                course_name="Fictional Course Alpha",
                cache=self.cache,
            )

        self.assertEqual(extract.call_count, 3)

    def test_invalid_entry_bounds_are_rejected(self):
        for value in (0, -1, True, False, 1.5, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "max_entries must be a positive integer",
                ):
                    LocalMaterialCache(max_entries=value)

    def test_agent_runtime_reuses_its_process_local_cache(self):
        self.create_text(self.alpha, "notes.txt", "fictional runtime")
        runtime = AgentToolRuntime(
            connector=Mock(spec=GmailConnector),
            materials_source=self.source,
            timezone_name="Europe/Madrid",
            max_gmail_results=100,
            max_lookback_days=120,
            max_material_results=10,
            max_chunk_chars=2000,
        )
        arguments = {"query": "runtime", "course_name": None, "limit": 10}
        now = datetime(2054, 9, 23, tzinfo=timezone.utc)

        with self.extraction_patch() as extract:
            first = runtime.execute("search_local_materials", arguments, now=now)
            second = runtime.execute("search_local_materials", arguments, now=now)

        self.assertEqual(extract.call_count, 1)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
