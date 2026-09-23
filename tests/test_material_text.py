"""Tests for deterministic local material text extraction."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from university_agent.local_materials import LocalMaterial, LocalMaterialsSource
from university_agent.material_text import (
    MaterialTextExtractionError,
    UnsupportedMaterialFormatError,
    extract_text,
)


class MaterialTextTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.root = self.base / "materials"
        self.course_path = self.root / "Fictional Course Alpha"
        self.course_path.mkdir(parents=True)
        self.source = LocalMaterialsSource(self.root)

    def create_bytes(self, relative_path: str, content: bytes) -> Path:
        path = self.course_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def discover(self, relative_path: str) -> LocalMaterial:
        expected_path = Path(relative_path)
        return next(
            material
            for material in self.source.list_materials("Fictional Course Alpha")
            if material.relative_path == expected_path
        )

    def create_pdf(
        self,
        relative_path: str,
        page_texts: tuple[str | None, ...],
        *,
        password: str | None = None,
    ) -> LocalMaterial:
        path = self.course_path / relative_path
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

        if password is not None:
            writer.encrypt(password)

        with path.open("wb") as output:
            writer.write(output)

        return self.discover(relative_path)

    def test_extracts_utf8_txt_with_unicode_and_provenance(self):
        path = self.create_bytes(
            "notes/lesson.txt",
            "Lliçó fictícia\nÀlgebra i programació".encode("utf-8"),
        )
        material = self.discover("notes/lesson.txt")

        document = extract_text(self.source, material)

        self.assertEqual(document.course_name, "Fictional Course Alpha")
        self.assertEqual(document.relative_path, Path("notes/lesson.txt"))
        self.assertEqual(document.format, ".txt")
        self.assertEqual(document.text, "Lliçó fictícia\nÀlgebra i programació")
        self.assertIsNone(document.page_texts)
        self.assertIsNone(document.page_count)
        self.assertFalse(hasattr(document, "path"))
        self.assertNotEqual(document.relative_path, path)

    def test_empty_txt_is_valid(self):
        self.create_bytes("empty.txt", b"")

        document = extract_text(self.source, self.discover("empty.txt"))

        self.assertEqual(document.text, "")
        self.assertIsNone(document.page_texts)

    def test_uppercase_txt_extension_is_supported(self):
        self.create_bytes("UPPER.TXT", b"Fictional uppercase text")

        document = extract_text(self.source, self.discover("UPPER.TXT"))

        self.assertEqual(document.format, ".txt")
        self.assertEqual(document.text, "Fictional uppercase text")

    def test_invalid_utf8_preserves_the_original_error_as_cause(self):
        self.create_bytes("invalid.txt", b"\xff\xfe")

        with self.assertRaises(MaterialTextExtractionError) as caught:
            extract_text(self.source, self.discover("invalid.txt"))

        self.assertIsInstance(caught.exception.__cause__, UnicodeDecodeError)

    def test_extracts_fictional_one_page_pdf(self):
        material = self.create_pdf("lesson.pdf", ("Fictional first page",))

        document = extract_text(self.source, material)

        self.assertEqual(document.course_name, "Fictional Course Alpha")
        self.assertEqual(document.relative_path, Path("lesson.pdf"))
        self.assertEqual(document.format, ".pdf")
        self.assertEqual(document.page_texts, ("Fictional first page",))
        self.assertEqual(document.text, "Fictional first page")
        self.assertEqual(document.page_count, 1)

    def test_pdf_preserves_page_order_empty_pages_and_separator(self):
        material = self.create_pdf(
            "multi-page.pdf",
            ("Fictional page one", None, "Fictional page three"),
        )

        document = extract_text(self.source, material)

        self.assertEqual(
            document.page_texts,
            ("Fictional page one", "", "Fictional page three"),
        )
        self.assertEqual(
            document.text,
            "Fictional page one\n\n\n\nFictional page three",
        )
        self.assertEqual(document.page_count, 3)

    def test_uppercase_pdf_extension_is_supported(self):
        material = self.create_pdf("UPPER.PDF", ("Fictional PDF",))

        document = extract_text(self.source, material)

        self.assertEqual(document.format, ".pdf")
        self.assertEqual(document.text, "Fictional PDF")

    def test_corrupt_pdf_preserves_the_original_error_as_cause(self):
        self.create_bytes("corrupt.pdf", b"not a PDF")

        with self.assertLogs("pypdf", level="WARNING"):
            with self.assertRaises(MaterialTextExtractionError) as caught:
                extract_text(self.source, self.discover("corrupt.pdf"))

        self.assertIsNotNone(caught.exception.__cause__)

    def test_zero_page_pdf_is_valid_empty_text(self):
        material = self.create_pdf("zero-pages.pdf", ())

        document = extract_text(self.source, material)

        self.assertEqual(document.page_texts, ())
        self.assertEqual(document.page_count, 0)
        self.assertEqual(document.text, "")

    def test_blank_pdf_page_is_valid_empty_text(self):
        material = self.create_pdf("blank-page.pdf", (None,))

        document = extract_text(self.source, material)

        self.assertEqual(document.page_texts, ("",))
        self.assertEqual(document.page_count, 1)
        self.assertEqual(document.text, "")

    def test_encrypted_pdf_without_password_fails(self):
        material = self.create_pdf(
            "encrypted.pdf",
            ("Fictional protected text",),
            password="fictional-password",
        )

        with self.assertRaises(MaterialTextExtractionError) as caught:
            extract_text(self.source, material)

        self.assertIsNotNone(caught.exception.__cause__)

    def test_unsupported_extension_raises(self):
        self.create_bytes("unsupported.md", b"Fictional markdown")

        with self.assertRaisesRegex(
            UnsupportedMaterialFormatError,
            "unsupported material format",
        ):
            extract_text(self.source, self.discover("unsupported.md"))

    def test_fabricated_external_material_is_rejected_without_opening(self):
        external_path = self.base / "outside.txt"
        external_path.write_text("fictional external data", encoding="utf-8")
        fabricated = LocalMaterial(
            course_name="Fictional Course Alpha",
            relative_path=Path("outside.txt"),
            path=external_path,
            filename="outside.txt",
            extension=".txt",
        )

        with patch.object(
            Path,
            "open",
            side_effect=AssertionError("external material was opened"),
        ):
            with self.assertRaisesRegex(ValueError, "does not belong"):
                extract_text(self.source, fabricated)

    def test_removed_material_is_rejected_before_opening(self):
        path = self.create_bytes("removed.txt", b"Fictional text")
        material = self.discover("removed.txt")
        path.unlink()

        with patch.object(
            Path,
            "open",
            side_effect=AssertionError("removed material was opened"),
        ):
            with self.assertRaisesRegex(ValueError, "does not belong"):
                extract_text(self.source, material)

    def test_material_replaced_by_symlink_is_rejected_before_opening(self):
        path = self.create_bytes("replace.txt", b"Fictional text")
        material = self.discover("replace.txt")
        external_path = self.base / "outside.txt"
        external_path.write_text("fictional external data", encoding="utf-8")
        path.unlink()
        try:
            path.symlink_to(external_path)
        except OSError as error:
            self.skipTest(f"symbolic links unavailable: {error}")

        with patch.object(
            Path,
            "open",
            side_effect=AssertionError("symlinked material was opened"),
        ):
            with self.assertRaisesRegex(ValueError, "does not belong"):
                extract_text(self.source, material)

    def test_material_from_another_source_is_rejected(self):
        self.create_bytes("lesson.txt", b"Fictional text")
        material = self.discover("lesson.txt")
        other_root = self.base / "other-materials"
        (other_root / "Fictional Course Alpha").mkdir(parents=True)
        other_source = LocalMaterialsSource(other_root)

        with self.assertRaisesRegex(ValueError, "does not belong"):
            extract_text(other_source, material)


if __name__ == "__main__":
    unittest.main()
