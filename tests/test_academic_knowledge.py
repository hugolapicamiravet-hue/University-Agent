"""Acceptance tests for the reusable academic knowledge core."""

import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import fields
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from university_agent import (
    AcademicKnowledgeEngine,
    MaterialPassage,
    MaterialReference,
)
from university_agent.academic_operations import MaterialPassage as LegacyPassage


class AcademicKnowledgeEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "materials"
        self.alpha = self.root / "Fictional Course Alpha"
        self.beta = self.root / "Fictional Course Beta"
        self.alpha.mkdir(parents=True)
        self.beta.mkdir()

    def create_text(self, course: Path, relative_path: str, text: str) -> None:
        path = course / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def create_pdf(
        self,
        course: Path,
        relative_path: str,
        page_texts: tuple[str | None, ...],
    ) -> None:
        path = course / relative_path
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

    def test_configures_engine_and_discovers_only_relative_references(self):
        self.create_text(self.alpha, "nested/notes.txt", "fictional material")

        engine = AcademicKnowledgeEngine(self.root)

        self.assertEqual(
            engine.list_courses(),
            ["Fictional Course Alpha", "Fictional Course Beta"],
        )
        references = engine.list_materials("Fictional Course Alpha")
        self.assertEqual(
            references,
            [
                MaterialReference(
                    course_name="Fictional Course Alpha",
                    relative_path="nested/notes.txt",
                    filename="notes.txt",
                    extension=".txt",
                )
            ],
        )
        self.assertEqual(
            {field.name for field in fields(MaterialReference)},
            {"course_name", "relative_path", "filename", "extension"},
        )
        self.assertFalse(Path(references[0].relative_path).is_absolute())
        self.assertFalse(hasattr(references[0], "path"))

    def test_searches_txt_without_exposing_absolute_paths(self):
        self.create_text(
            self.alpha,
            "unit/notes.txt",
            "Fictional notes about mutual exclusion",
        )

        passages = AcademicKnowledgeEngine(self.root).search("mutual exclusion")

        self.assertEqual(len(passages), 1)
        passage = passages[0]
        self.assertEqual(passage.course_name, "Fictional Course Alpha")
        self.assertEqual(passage.relative_path, "unit/notes.txt")
        self.assertIsNone(passage.page_number)
        self.assertIn("mutual exclusion", passage.text)
        self.assertFalse(Path(passage.relative_path).is_absolute())
        self.assertFalse(hasattr(passage, "path"))

    def test_searches_pdf_and_preserves_page_provenance(self):
        self.create_pdf(
            self.alpha,
            "slides.pdf",
            ("Fictional introduction", "Fictional coherence protocol"),
        )

        passages = AcademicKnowledgeEngine(self.root).search("protocol")

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].relative_path, "slides.pdf")
        self.assertEqual(passages[0].page_number, 2)

    def test_course_filter_excludes_unrelated_course(self):
        self.create_text(self.alpha, "alpha.txt", "shared academic term")
        self.create_text(self.beta, "beta.txt", "shared academic term")

        passages = AcademicKnowledgeEngine(self.root).search(
            "shared",
            course="Fictional Course Beta",
        )

        self.assertEqual(
            [(passage.course_name, passage.relative_path) for passage in passages],
            [("Fictional Course Beta", "beta.txt")],
        )

    def test_limit_is_applied_after_existing_global_ranking(self):
        self.create_text(self.alpha, "once.txt", "term")
        self.create_text(self.beta, "repeated.txt", "term term term")

        passages = AcademicKnowledgeEngine(self.root).search("term", limit=1)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].relative_path, "repeated.txt")

    def test_legacy_adapter_reuses_the_public_passage_type(self):
        self.assertIs(LegacyPassage, MaterialPassage)

    def test_public_core_import_does_not_load_application_dependencies(self):
        code = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockApplicationDependencies(importlib.abc.MetaPathFinder):
                blocked = {"google", "ollama", "openai"}

                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".", 1)[0] in self.blocked:
                        raise ImportError(f"blocked application dependency: {fullname}")
                    return None

            sys.meta_path.insert(0, BlockApplicationDependencies())
            from university_agent import AcademicKnowledgeEngine
            assert AcademicKnowledgeEngine is not None
            assert "university_agent.connectors.gmail" not in sys.modules
            assert "university_agent.ollama_agent" not in sys.modules
            assert "university_agent.openai_agent" not in sys.modules
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
