"""Tests for narrow agent-facing academic application operations."""

import tempfile
import unittest
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from university_agent.academic_notifications import (
    NotificationCategory,
    ParsedNotification,
)
from university_agent.academic_operations import (
    CourseNoticeResult,
    DeadlineResult,
    MaterialPassage,
    get_course_notices,
    get_deadlines,
    search_materials,
)
from university_agent.connectors.gmail import GmailConnector
from university_agent.course_notices import ParsedCourseNotice
from university_agent.local_materials import LocalMaterialsSource
from university_agent.material_text import MaterialTextExtractionError
from university_agent.recent_course_notices import DetectedCourseNotice
from university_agent.upcoming_deadlines import DetectedDeadline


class GmailAcademicOperationTests(unittest.TestCase):
    def setUp(self):
        self.connector = Mock(spec=GmailConnector)

    @patch("university_agent.academic_operations.find_upcoming_deadlines_until")
    def test_deadlines_are_narrowed_and_order_preserved(self, find_deadlines):
        first_due = datetime(2054, 9, 21, 10, 0)
        second_due = datetime(2054, 9, 22, 12, 0)
        find_deadlines.return_value = [
            DetectedDeadline(
                message_id="fictional-message-1",
                thread_id="fictional-thread-1",
                notification=ParsedNotification(
                    category=NotificationCategory.DEADLINE,
                    raw_subject="Fictional deadline one",
                    title="Fictional task one",
                    event_at=first_due,
                ),
            ),
            DetectedDeadline(
                message_id="fictional-message-2",
                thread_id="fictional-thread-2",
                notification=ParsedNotification(
                    category=NotificationCategory.DEADLINE,
                    raw_subject="Fictional deadline two",
                    title="Fictional task two",
                    event_at=second_due,
                ),
            ),
        ]
        now = datetime(2054, 9, 20, 9, 0)
        until = datetime(2054, 9, 23, 9, 0)

        results = get_deadlines(
            self.connector,
            now=now,
            until=until,
            max_results=37,
        )

        find_deadlines.assert_called_once_with(
            self.connector,
            now=now,
            until=until,
            max_results=37,
        )
        self.assertEqual(
            results,
            [
                DeadlineResult("Fictional task one", first_due),
                DeadlineResult("Fictional task two", second_due),
            ],
        )
        self.assertEqual(
            {field.name for field in fields(DeadlineResult)},
            {"title", "due_at"},
        )

    @patch("university_agent.academic_operations.find_upcoming_deadlines_until")
    def test_deadline_errors_propagate_unchanged(self, find_deadlines):
        error = RuntimeError("fictional Gmail failure")
        find_deadlines.side_effect = error

        with self.assertRaises(RuntimeError) as caught:
            get_deadlines(
                self.connector,
                now=datetime(2054, 9, 20),
                until=datetime(2054, 9, 21),
            )

        self.assertIs(caught.exception, error)

    @patch("university_agent.academic_operations.find_recent_course_notices")
    def test_course_notices_are_narrowed_and_delegated(self, find_notices):
        find_notices.return_value = [
            DetectedCourseNotice(
                message_id="fictional-message",
                thread_id="fictional-thread",
                message_date="Mon, 20 Sep 2054 10:00:00 +0200",
                notice=ParsedCourseNotice(
                    raw_subject="EI0001-MT0001-2054-2055: Fictional notice",
                    course_codes=("EI0001", "MT0001"),
                    academic_year="2054-2055",
                    notice_text="Fictional notice text",
                ),
            )
        ]

        results = get_course_notices(
            self.connector,
            course_code="EI0001",
            academic_year="2054-2055",
            lookback_days=45,
            max_results=23,
        )

        find_notices.assert_called_once_with(
            self.connector,
            course_code="EI0001",
            academic_year="2054-2055",
            lookback_days=45,
            max_results=23,
        )
        self.assertEqual(
            results,
            [
                CourseNoticeResult(
                    course_codes=("EI0001", "MT0001"),
                    academic_year="2054-2055",
                    text="Fictional notice text",
                    message_date="Mon, 20 Sep 2054 10:00:00 +0200",
                )
            ],
        )
        self.assertEqual(
            {field.name for field in fields(CourseNoticeResult)},
            {"course_codes", "academic_year", "text", "message_date"},
        )

    @patch("university_agent.academic_operations.find_recent_course_notices")
    def test_course_notice_errors_propagate_unchanged(self, find_notices):
        error = ValueError("fictional validation failure")
        find_notices.side_effect = error

        with self.assertRaises(ValueError) as caught:
            get_course_notices(
                self.connector,
                course_code="EI0001",
                academic_year="2054-2055",
            )

        self.assertIs(caught.exception, error)


class MaterialAcademicOperationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "materials"
        self.alpha = self.root / "Fictional Course Alpha"
        self.beta = self.root / "Fictional Course Beta"
        self.alpha.mkdir(parents=True)
        self.beta.mkdir()
        self.source = LocalMaterialsSource(self.root)

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

    def test_txt_passage_is_narrowed_without_absolute_path(self):
        self.create_text(
            self.alpha,
            "nested/notes.txt",
            "Fictional cache coherence notes",
        )

        passages = search_materials(self.source, query="coherence")

        self.assertEqual(len(passages), 1)
        passage = passages[0]
        self.assertEqual(passage.course_name, "Fictional Course Alpha")
        self.assertEqual(passage.relative_path, "nested/notes.txt")
        self.assertIsNone(passage.page_number)
        self.assertEqual(passage.text, "Fictional cache coherence notes")
        self.assertGreater(passage.score, 0)
        self.assertFalse(hasattr(passage, "path"))
        self.assertEqual(
            {field.name for field in fields(MaterialPassage)},
            {"course_name", "relative_path", "page_number", "text", "score"},
        )

    def test_pdf_passage_preserves_page_provenance(self):
        self.create_pdf(
            self.alpha,
            "slides.pdf",
            ("Fictional introduction", "Fictional cache protocol"),
        )

        passages = search_materials(self.source, query="protocol")

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].relative_path, "slides.pdf")
        self.assertEqual(passages[0].page_number, 2)

    def test_course_scope_and_all_course_search(self):
        self.create_text(self.alpha, "alpha.txt", "shared fictional term")
        self.create_text(self.beta, "beta.txt", "shared fictional term")

        scoped = search_materials(
            self.source,
            query="shared",
            course_name="Fictional Course Beta",
        )
        all_courses = search_materials(self.source, query="shared")

        self.assertEqual(
            [passage.course_name for passage in scoped],
            ["Fictional Course Beta"],
        )
        self.assertEqual(
            [passage.course_name for passage in all_courses],
            ["Fictional Course Alpha", "Fictional Course Beta"],
        )

    def test_ranking_order_and_scores_are_preserved(self):
        self.create_text(self.alpha, "once.txt", "cache")
        self.create_text(self.beta, "repeated.txt", "cache cache cache")

        passages = search_materials(self.source, query="cache")

        self.assertEqual(
            [passage.relative_path for passage in passages],
            ["repeated.txt", "once.txt"],
        )
        self.assertGreater(passages[0].score, passages[1].score)

    def test_unsupported_formats_are_skipped(self):
        self.create_text(self.alpha, "unsupported.docx", "fictional content")
        self.create_text(self.alpha, "supported.txt", "searchable content")

        passages = search_materials(self.source, query="searchable")

        self.assertEqual(
            [passage.relative_path for passage in passages],
            ["supported.txt"],
        )

    def test_corrupt_supported_document_error_propagates(self):
        (self.alpha / "corrupt.pdf").write_bytes(b"not a PDF")

        with self.assertLogs("pypdf", level="WARNING"):
            with self.assertRaises(MaterialTextExtractionError):
                search_materials(self.source, query="fictional")


if __name__ == "__main__":
    unittest.main()
