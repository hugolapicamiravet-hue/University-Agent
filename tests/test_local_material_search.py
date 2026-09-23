"""Tests for the high-level local material search workflow."""

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from university_agent.local_material_search import search_local_materials
from university_agent.local_materials import LocalMaterialsSource
from university_agent.material_text import MaterialTextExtractionError


class LocalMaterialSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "materials"
        self.alpha = self.root / "Fictional Course Alpha"
        self.beta = self.root / "Fictional Course Beta"
        self.alpha.mkdir(parents=True)
        self.beta.mkdir()
        self.source = LocalMaterialsSource(self.root)

    def create_text(
        self,
        course_path: Path,
        relative_path: str,
        text: str,
    ) -> Path:
        path = course_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def create_pdf(
        self,
        course_path: Path,
        relative_path: str,
        page_texts: tuple[str | None, ...],
    ) -> Path:
        path = course_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = PdfWriter()

        for page_text in page_texts:
            page = writer.add_blank_page(width=612, height=792)
            if page_text is None:
                continue

            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            font_reference = writer._add_object(font)
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): font_reference}
                    )
                }
            )
            content = DecodedStreamObject()
            content.set_data(
                f"BT /F1 12 Tf 72 720 Td ({page_text}) Tj ET".encode("ascii")
            )
            page[NameObject("/Contents")] = writer._add_object(content)

        with path.open("wb") as output:
            writer.write(output)
        return path

    def test_searches_one_txt_material(self):
        self.create_text(
            self.alpha,
            "notes.txt",
            "Fictional notes about concurrent systems",
        )

        results = search_local_materials(
            self.source,
            query="concurrent",
            course_name="Fictional Course Alpha",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.relative_path, Path("notes.txt"))

    def test_searches_pdf_and_preserves_page_provenance(self):
        self.create_pdf(
            self.alpha,
            "slides.pdf",
            ("Fictional introduction", "Fictional distributed systems"),
        )

        results = search_local_materials(
            self.source,
            query="distributed",
            course_name="Fictional Course Alpha",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.relative_path, Path("slides.pdf"))
        self.assertEqual(results[0].chunk.page_number, 2)

    def test_searches_nested_material(self):
        self.create_text(
            self.alpha,
            "laboratory/week-01.txt",
            "Fictional laboratory protocol",
        )

        results = search_local_materials(self.source, query="protocol")

        self.assertEqual(
            results[0].chunk.relative_path,
            Path("laboratory/week-01.txt"),
        )

    def test_course_scope_excludes_other_courses(self):
        self.create_text(self.alpha, "alpha.txt", "shared fictional term")
        self.create_text(self.beta, "beta.txt", "shared fictional term")

        results = search_local_materials(
            self.source,
            query="shared",
            course_name="Fictional Course Beta",
        )

        self.assertEqual(
            [result.chunk.course_name for result in results],
            ["Fictional Course Beta"],
        )

    def test_all_course_search_includes_each_course(self):
        self.create_text(self.alpha, "alpha.txt", "shared fictional term")
        self.create_text(self.beta, "beta.txt", "shared fictional term")

        results = search_local_materials(self.source, query="shared")

        self.assertEqual(
            [result.chunk.course_name for result in results],
            ["Fictional Course Alpha", "Fictional Course Beta"],
        )

    def test_unsupported_files_are_skipped(self):
        self.create_text(self.alpha, "unsupported.docx", "not a real document")
        self.create_text(self.alpha, "supported.txt", "searchable fictional text")

        results = search_local_materials(self.source, query="searchable")

        self.assertEqual(
            [result.chunk.relative_path for result in results],
            [Path("supported.txt")],
        )

    def test_corrupt_supported_pdf_propagates_extraction_error(self):
        (self.alpha / "corrupt.pdf").write_bytes(b"not a PDF")

        with self.assertLogs("pypdf", level="WARNING"):
            with self.assertRaises(MaterialTextExtractionError):
                search_local_materials(self.source, query="fictional")

    def test_course_name_matching_is_exact(self):
        self.create_text(self.alpha, "notes.txt", "fictional text")

        with self.assertRaisesRegex(ValueError, "unknown course"):
            search_local_materials(
                self.source,
                query="fictional",
                course_name="fictional course alpha",
            )

    def test_unknown_course_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown course"):
            search_local_materials(
                self.source,
                query="fictional",
                course_name="Fictional Missing Course",
            )

    def test_no_match_returns_empty_list(self):
        self.create_text(self.alpha, "notes.txt", "fictional operating systems")

        self.assertEqual(
            search_local_materials(self.source, query="networks"),
            [],
        )

    def test_ranking_is_deterministic_across_courses_and_documents(self):
        self.create_text(self.beta, "zeta.txt", "shared term")
        self.create_text(self.alpha, "zeta.txt", "shared term")
        self.create_text(self.alpha, "alpha.txt", "shared term")

        first = search_local_materials(self.source, query="shared")
        second = search_local_materials(self.source, query="shared")

        self.assertEqual(first, second)
        self.assertEqual(
            [
                (result.chunk.course_name, result.chunk.relative_path.as_posix())
                for result in first
            ],
            [
                ("Fictional Course Alpha", "alpha.txt"),
                ("Fictional Course Alpha", "zeta.txt"),
                ("Fictional Course Beta", "zeta.txt"),
            ],
        )

    def test_limit_is_applied_after_global_ranking(self):
        self.create_text(self.alpha, "once.txt", "term")
        self.create_text(self.beta, "repeated.txt", "term term term")

        results = search_local_materials(
            self.source,
            query="term",
            limit=1,
        )

        self.assertEqual(
            results[0].chunk.relative_path,
            Path("repeated.txt"),
        )

    def test_results_do_not_expose_absolute_paths(self):
        self.create_text(self.alpha, "notes.txt", "fictional text")

        result = search_local_materials(self.source, query="fictional")[0]

        self.assertFalse(hasattr(result.chunk, "path"))
        self.assertFalse(result.chunk.relative_path.is_absolute())

    def test_blank_query_is_rejected_before_filesystem_discovery(self):
        self.root.rename(self.root.with_name("moved-materials"))

        with self.assertRaisesRegex(ValueError, "query"):
            search_local_materials(self.source, query="   ")

    def test_invalid_chunk_limit_is_rejected_before_filesystem_discovery(self):
        self.root.rename(self.root.with_name("moved-materials"))

        with self.assertRaisesRegex(ValueError, "at least 100"):
            search_local_materials(
                self.source,
                query="fictional",
                max_chunk_chars=99,
            )


if __name__ == "__main__":
    unittest.main()
