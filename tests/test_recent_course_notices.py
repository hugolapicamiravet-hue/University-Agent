"""Isolated tests for the recent course-notices workflow."""

import unittest
from unittest.mock import Mock

from university_agent.connectors.gmail import GmailConnector, GmailMessage
from university_agent.recent_course_notices import find_recent_course_notices


def gmail_message(
    message_id: str,
    subject: str,
    *,
    thread_id: str | None = None,
    message_date: str = "Tue, 22 Sep 2054 10:00:00 +0200",
) -> GmailMessage:
    return {
        "id": message_id,
        "thread_id": thread_id or f"thread-{message_id}",
        "sender": "",
        "recipients": "",
        "subject": subject,
        "date": message_date,
        "snippet": "",
    }


class RecentCourseNoticesTests(unittest.TestCase):
    def setUp(self):
        self.connector = Mock(spec=GmailConnector)

    def test_exact_query_limit_and_provenance_are_preserved(self):
        subject = "EI0004-2054-2055: Aviso: cambio de aula"
        message_date = "Tue, 22 Sep 2054 10:00:00 +0200"
        self.connector.search_messages.return_value = [
            gmail_message(
                "m1",
                subject,
                thread_id="t1",
                message_date=message_date,
            )
        ]

        results = find_recent_course_notices(
            self.connector,
            course_code="EI0004",
            academic_year="2054-2055",
            lookback_days=45,
            max_results=37,
        )

        self.connector.search_messages.assert_called_once_with(
            'newer_than:45d subject:"2054-2055"',
            max_results=37,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].message_id, "m1")
        self.assertEqual(results[0].thread_id, "t1")
        self.assertEqual(results[0].message_date, message_date)
        self.assertEqual(results[0].notice.raw_subject, subject)
        self.assertEqual(results[0].notice.course_codes, ("EI0004",))
        self.assertEqual(results[0].notice.academic_year, "2054-2055")
        self.assertEqual(results[0].notice.notice_text, "Aviso: cambio de aula")

    def test_empty_results_use_default_query_values(self):
        self.connector.search_messages.return_value = []

        self.assertEqual(
            find_recent_course_notices(
                self.connector,
                course_code="EI0004",
                academic_year="2054-2055",
            ),
            [],
        )
        self.connector.search_messages.assert_called_once_with(
            'newer_than:120d subject:"2054-2055"',
            max_results=100,
        )

    def test_single_double_and_triple_code_matches(self):
        self.connector.search_messages.return_value = [
            gmail_message("single", "EI0004-2054-2055: Single"),
            gmail_message("double", "MT0004-EI0004-2054-2055: Double"),
            gmail_message(
                "triple",
                "EI0005-EI0004-MT0005-2054-2055: Triple",
            ),
        ]

        results = find_recent_course_notices(
            self.connector,
            course_code="EI0004",
            academic_year="2054-2055",
        )

        self.assertEqual(
            [result.message_id for result in results],
            ["double", "single", "triple"],
        )

    def test_nonmatching_messages_are_excluded(self):
        self.connector.search_messages.return_value = [
            gmail_message("other-code", "MT0004-2054-2055: Notice"),
            gmail_message("other-year", "EI0004-2053-2054: Notice"),
            gmail_message("malformed", "EI0004: Notice"),
            gmail_message(
                "text-only",
                "EI0002-2054-2055: Discuss EI0004 next week",
            ),
        ]

        self.assertEqual(
            find_recent_course_notices(
                self.connector,
                course_code="EI0004",
                academic_year="2054-2055",
            ),
            [],
        )

    def test_structurally_associated_institutional_notice_is_retained(self):
        subject = (
            "EI0004-2054-2055: Inscripció fictícia oberta. Taller d'Exploració"
        )
        self.connector.search_messages.return_value = [
            gmail_message("institutional", subject)
        ]

        results = find_recent_course_notices(
            self.connector,
            course_code="EI0004",
            academic_year="2054-2055",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].notice.notice_text,
            "Inscripció fictícia oberta. Taller d'Exploració",
        )

    def test_duplicate_id_first_occurrence_wins_before_filtering(self):
        self.connector.search_messages.return_value = [
            gmail_message("duplicate", "MT0004-2054-2055: First"),
            gmail_message("duplicate", "EI0004-2054-2055: Second"),
        ]

        self.assertEqual(
            find_recent_course_notices(
                self.connector,
                course_code="EI0004",
                academic_year="2054-2055",
            ),
            [],
        )

    def test_identical_notices_with_different_ids_remain_separate(self):
        subject = "EI0004-2054-2055: Shared"
        self.connector.search_messages.return_value = [
            gmail_message("m2", subject),
            gmail_message("m1", subject),
        ]

        results = find_recent_course_notices(
            self.connector,
            course_code="EI0004",
            academic_year="2054-2055",
        )

        self.assertEqual(
            [result.message_id for result in results],
            ["m1", "m2"],
        )

    def test_valid_dates_sort_newest_first_and_equal_instants_by_id(self):
        self.connector.search_messages.return_value = [
            gmail_message(
                "older",
                "EI0004-2054-2055: Older",
                message_date="Mon, 21 Sep 2054 09:00:00 +0200",
            ),
            gmail_message(
                "same-b",
                "EI0004-2054-2055: Same B",
                message_date="Tue, 22 Sep 2054 10:00:00 +0200",
            ),
            gmail_message(
                "same-a",
                "EI0004-2054-2055: Same A",
                message_date="Tue, 22 Sep 2054 08:00:00 +0000",
            ),
            gmail_message(
                "newest",
                "EI0004-2054-2055: Newest",
                message_date="Wed, 23 Sep 2054 10:00:00 +0200",
            ),
        ]

        results = find_recent_course_notices(
            self.connector,
            course_code="EI0004",
            academic_year="2054-2055",
        )

        self.assertEqual(
            [result.message_id for result in results],
            ["newest", "same-a", "same-b", "older"],
        )

    def test_invalid_missing_and_naive_dates_sort_last_by_id(self):
        self.connector.search_messages.return_value = [
            gmail_message("missing", "EI0004-2054-2055: Missing", message_date=""),
            gmail_message(
                "valid",
                "EI0004-2054-2055: Valid",
                message_date="Tue, 22 Sep 2054 10:00:00 +0200",
            ),
            gmail_message(
                "malformed",
                "EI0004-2054-2055: Malformed",
                message_date="not a date",
            ),
            gmail_message(
                "naive",
                "EI0004-2054-2055: Naive",
                message_date="Tue, 22 Sep 2054 10:00:00",
            ),
        ]

        results = find_recent_course_notices(
            self.connector,
            course_code="EI0004",
            academic_year="2054-2055",
        )

        self.assertEqual(
            [result.message_id for result in results],
            ["valid", "malformed", "missing", "naive"],
        )

    def test_invalid_inputs_are_rejected_before_gmail_access(self):
        cases = (
            {"course_code": "EI000", "academic_year": "2054-2055"},
            {"course_code": "ei0004", "academic_year": "2054-2055"},
            {"course_code": "COR000001", "academic_year": "2054-2055"},
            {"course_code": "EI٠٠٠١", "academic_year": "2054-2055"},
            {"course_code": "EI0004", "academic_year": "2054-55"},
            {"course_code": "EI0004", "academic_year": "2054/2055"},
            {"course_code": "EI0004", "academic_year": "2054-2056"},
            {"course_code": "EI0004", "academic_year": "٢٠٥٤-٢٠٥٥"},
            {
                "course_code": "EI0004",
                "academic_year": "2054-2055",
                "lookback_days": 0,
            },
        )

        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                find_recent_course_notices(self.connector, **arguments)

        self.connector.search_messages.assert_not_called()

    def test_max_results_is_forwarded_without_local_validation(self):
        self.connector.search_messages.return_value = []

        find_recent_course_notices(
            self.connector,
            course_code="EI0004",
            academic_year="2054-2055",
            max_results=0,
        )

        self.connector.search_messages.assert_called_once_with(
            'newer_than:120d subject:"2054-2055"',
            max_results=0,
        )

    def test_gmail_error_object_propagates_unchanged(self):
        error = RuntimeError("API failure")
        self.connector.search_messages.side_effect = error

        with self.assertRaises(RuntimeError) as caught:
            find_recent_course_notices(
                self.connector,
                course_code="EI0004",
                academic_year="2054-2055",
            )

        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
