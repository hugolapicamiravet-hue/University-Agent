"""Tests for deterministic extracted-material chunking."""

import unittest
from pathlib import Path

from university_agent.material_chunks import chunk_document
from university_agent.material_text import ExtractedDocument


def extracted_document(
    text: str,
    *,
    format: str = ".txt",
    page_texts: tuple[str, ...] | None = None,
) -> ExtractedDocument:
    return ExtractedDocument(
        course_name="Fictional Course Alpha",
        relative_path=Path("nested/fictional-material.txt"),
        format=format,
        text=text,
        page_texts=page_texts,
    )


class MaterialChunkTests(unittest.TestCase):
    def test_empty_and_whitespace_txt_produce_no_chunks(self):
        for text in ("", "   \n\t  "):
            with self.subTest(text=text):
                self.assertEqual(chunk_document(extracted_document(text)), [])

    def test_short_txt_produces_one_chunk_with_provenance(self):
        chunks = chunk_document(extracted_document("Fictional short text"))

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.course_name, "Fictional Course Alpha")
        self.assertEqual(
            chunk.relative_path,
            Path("nested/fictional-material.txt"),
        )
        self.assertEqual(chunk.format, ".txt")
        self.assertEqual(chunk.text, "Fictional short text")
        self.assertIsNone(chunk.page_number)
        self.assertEqual(chunk.chunk_index, 0)
        self.assertFalse(hasattr(chunk, "path"))

    def test_paragraph_boundaries_are_preferred(self):
        first = "A" * 60
        second = "B" * 60

        chunks = chunk_document(
            extracted_document(f"{first}\n\n{second}"),
            max_chars=100,
        )

        self.assertEqual([chunk.text for chunk in chunks], [first, second])

    def test_short_paragraphs_are_combined_with_blank_line(self):
        document = extracted_document("First paragraph\n\nSecond paragraph")

        chunks = chunk_document(document, max_chars=100)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["First paragraph\n\nSecond paragraph"],
        )

    def test_long_paragraph_splits_at_whitespace(self):
        words = [f"word{index:02d}" for index in range(30)]
        chunks = chunk_document(
            extracted_document(" ".join(words)),
            max_chars=100,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 100 for chunk in chunks))
        self.assertEqual(
            " ".join(chunk.text for chunk in chunks),
            " ".join(words),
        )

    def test_word_longer_than_limit_is_split_without_empty_chunks(self):
        chunks = chunk_document(
            extracted_document("x" * 205),
            max_chars=100,
        )

        self.assertEqual([len(chunk.text) for chunk in chunks], [100, 100, 5])
        self.assertTrue(all(chunk.text for chunk in chunks))

    def test_chunk_indexes_are_deterministic_and_document_wide(self):
        document = extracted_document(
            "",
            format=".pdf",
            page_texts=("A" * 150, "B" * 150),
        )

        first = chunk_document(document, max_chars=100)
        second = chunk_document(document, max_chars=100)

        self.assertEqual(first, second)
        self.assertEqual(
            [chunk.chunk_index for chunk in first],
            list(range(len(first))),
        )

    def test_unicode_and_accents_are_preserved(self):
        text = "Programació, àlgebra i lingüística fictícia"

        chunks = chunk_document(extracted_document(text))

        self.assertEqual(chunks[0].text, text)

    def test_text_exactly_at_limit_remains_one_chunk(self):
        text = "x" * 100

        chunks = chunk_document(extracted_document(text), max_chars=100)

        self.assertEqual([chunk.text for chunk in chunks], [text])

    def test_max_chars_below_one_hundred_is_rejected(self):
        for max_chars in (99, 1, 0, -1):
            with self.subTest(max_chars=max_chars):
                with self.assertRaisesRegex(ValueError, "at least 100"):
                    chunk_document(
                        extracted_document("Fictional text"),
                        max_chars=max_chars,
                    )

    def test_one_page_pdf_uses_one_based_page_number(self):
        document = extracted_document(
            "Fictional page",
            format=".pdf",
            page_texts=("Fictional page",),
        )

        chunks = chunk_document(document)

        self.assertEqual([chunk.page_number for chunk in chunks], [1])

    def test_multi_page_pdf_never_combines_pages(self):
        document = extracted_document(
            "First page\n\nSecond page",
            format=".pdf",
            page_texts=("First page", "Second page"),
        )

        chunks = chunk_document(document, max_chars=100)

        self.assertEqual(
            [(chunk.page_number, chunk.text) for chunk in chunks],
            [(1, "First page"), (2, "Second page")],
        )

    def test_blank_pdf_pages_create_no_chunks(self):
        document = extracted_document(
            "Visible page",
            format=".pdf",
            page_texts=("", "  \n", "Visible page"),
        )

        chunks = chunk_document(document)

        self.assertEqual(
            [(chunk.page_number, chunk.text) for chunk in chunks],
            [(3, "Visible page")],
        )


if __name__ == "__main__":
    unittest.main()
