"""Isolated tests for Gmail deadline detection composition."""

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from university_agent.academic_notifications import NotificationCategory
from university_agent.connectors.gmail import GmailConnector, GmailMessage
from university_agent.upcoming_deadlines import (
    find_upcoming_deadlines,
    find_upcoming_deadlines_until,
)


def gmail_message(
    message_id: str,
    subject: str,
    *,
    thread_id: str | None = None,
) -> GmailMessage:
    return {
        "id": message_id,
        "thread_id": thread_id or f"thread-{message_id}",
        "sender": "fixture1@example.com",
        "recipients": "fixture3@example.com",
        "subject": subject,
        "date": "",
        "snippet": "",
    }


def deadline_subject(day: int, hour: int, title: str = "Activitat") -> str:
    return (
        f"Venciment el dilluns, {day} de setembre 2054, {hour:02d}:00: "
        f"{title}"
    )


class UpcomingDeadlinesTests(unittest.TestCase):
    def setUp(self):
        self.connector = Mock(spec=GmailConnector)
        self.now = datetime(2054, 9, 20, 12, 0)

    def test_exact_query_limit_and_gmail_identity_are_preserved(self):
        self.connector.search_messages.return_value = [
            gmail_message("m1", deadline_subject(22, 10), thread_id="t1")
        ]

        results = find_upcoming_deadlines(
            self.connector,
            now=self.now,
            max_results=37,
        )

        self.connector.search_messages.assert_called_once_with(
            'subject:"Venciment el"',
            max_results=37,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].message_id, "m1")
        self.assertEqual(results[0].thread_id, "t1")
        self.assertEqual(
            results[0].notification.category,
            NotificationCategory.DEADLINE,
        )
        self.assertEqual(
            results[0].notification.event_at,
            datetime(2054, 9, 22, 10, 0),
        )

    def test_empty_results(self):
        self.connector.search_messages.return_value = []

        self.assertEqual(
            find_upcoming_deadlines(self.connector, now=self.now),
            [],
        )
        self.connector.search_messages.assert_called_once_with(
            'subject:"Venciment el"',
            max_results=100,
        )

    def test_past_deadline_is_excluded(self):
        self.connector.search_messages.return_value = [
            gmail_message("past", deadline_subject(19, 10))
        ]

        self.assertEqual(find_upcoming_deadlines(self.connector, now=self.now), [])

    def test_deadline_equal_to_now_is_excluded(self):
        self.connector.search_messages.return_value = [
            gmail_message("equal", deadline_subject(20, 12))
        ]

        self.assertEqual(find_upcoming_deadlines(self.connector, now=self.now), [])

    def test_unparseable_deadline_is_excluded(self):
        self.connector.search_messages.return_value = [
            gmail_message(
                "invalid",
                "Venciment el dilluns, 31 de febrer 2055, 10:00: Activitat",
            )
        ]

        self.assertEqual(find_upcoming_deadlines(self.connector, now=self.now), [])

    def test_activity_opens_is_excluded(self):
        self.connector.search_messages.return_value = [
            gmail_message(
                "opens",
                "Opens on dilluns, 22 de setembre 2054, 10:00: Activitat",
            )
        ]

        self.assertEqual(find_upcoming_deadlines(self.connector, now=self.now), [])

    def test_other_recognized_categories_and_unknown_are_excluded(self):
        subjects = (
            "Tasca vençuda: Activitat (fecha límite 19/09/2054 a las 10:00)",
            "Heu realitzat la tramesa de la tasca Activitat",
            "Teniu tasques que vencen en 7 dies",
            "Inici de sessió nou del vostre compte Aula Virtual UJI",
            'Confirmació de reserva fictícia',
        )
        self.connector.search_messages.return_value = [
            gmail_message(f"m{index}", subject)
            for index, subject in enumerate(subjects)
        ]

        self.assertEqual(find_upcoming_deadlines(self.connector, now=self.now), [])

    def test_results_are_sorted_by_event_then_message_id(self):
        self.connector.search_messages.return_value = [
            gmail_message("m3", deadline_subject(24, 9, "Later")),
            gmail_message("m2", deadline_subject(22, 10, "Same B")),
            gmail_message("m1", deadline_subject(22, 10, "Same A")),
        ]

        results = find_upcoming_deadlines(self.connector, now=self.now)

        self.assertEqual(
            [result.message_id for result in results],
            ["m1", "m2", "m3"],
        )

    def test_duplicate_message_id_keeps_first_occurrence(self):
        self.connector.search_messages.return_value = [
            gmail_message(
                "duplicate",
                deadline_subject(22, 10, "First"),
                thread_id="first-thread",
            ),
            gmail_message(
                "duplicate",
                deadline_subject(23, 10, "Second"),
                thread_id="second-thread",
            ),
        ]

        results = find_upcoming_deadlines(self.connector, now=self.now)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].thread_id, "first-thread")
        self.assertEqual(results[0].notification.title, "First")

    def test_duplicate_id_first_occurrence_wins_before_filtering(self):
        self.connector.search_messages.return_value = [
            gmail_message("duplicate", deadline_subject(19, 10, "Past")),
            gmail_message("duplicate", deadline_subject(22, 10, "Future")),
        ]

        self.assertEqual(
            find_upcoming_deadlines(self.connector, now=self.now),
            [],
        )

    def test_identical_deadlines_with_different_ids_remain_separate(self):
        subject = deadline_subject(22, 10, "Shared title")
        self.connector.search_messages.return_value = [
            gmail_message("m2", subject),
            gmail_message("m1", subject),
        ]

        results = find_upcoming_deadlines(self.connector, now=self.now)

        self.assertEqual(
            [result.message_id for result in results],
            ["m1", "m2"],
        )

    def test_timezone_aware_now_is_rejected_before_gmail_call(self):
        aware_now = datetime(2054, 9, 20, 12, 0, tzinfo=timezone.utc)

        with self.assertRaisesRegex(ValueError, "timezone-naive"):
            find_upcoming_deadlines(self.connector, now=aware_now)

        self.connector.search_messages.assert_not_called()

    def test_gmail_errors_propagate(self):
        error = RuntimeError("API failure")
        self.connector.search_messages.side_effect = error

        with self.assertRaises(RuntimeError) as caught:
            find_upcoming_deadlines(self.connector, now=self.now)

        self.assertIs(caught.exception, error)


class UpcomingDeadlinesUntilTests(unittest.TestCase):
    def setUp(self):
        self.connector = Mock(spec=GmailConnector)
        self.now = datetime(2054, 9, 20, 12, 0)
        self.until = datetime(2054, 9, 23, 10, 0)

    def test_deadline_before_until_is_included(self):
        self.connector.search_messages.return_value = [
            gmail_message("before-until", deadline_subject(22, 10))
        ]

        results = find_upcoming_deadlines_until(
            self.connector,
            now=self.now,
            until=self.until,
        )

        self.assertEqual([result.message_id for result in results], ["before-until"])

    def test_deadline_equal_to_until_is_included(self):
        self.connector.search_messages.return_value = [
            gmail_message("at-until", deadline_subject(23, 10))
        ]

        results = find_upcoming_deadlines_until(
            self.connector,
            now=self.now,
            until=self.until,
        )

        self.assertEqual([result.message_id for result in results], ["at-until"])

    def test_deadline_after_until_is_excluded(self):
        self.connector.search_messages.return_value = [
            gmail_message("after-until", deadline_subject(24, 10))
        ]

        self.assertEqual(
            find_upcoming_deadlines_until(
                self.connector,
                now=self.now,
                until=self.until,
            ),
            [],
        )

    def test_deadline_equal_to_now_remains_excluded(self):
        self.connector.search_messages.return_value = [
            gmail_message("at-now", deadline_subject(20, 12))
        ]

        self.assertEqual(
            find_upcoming_deadlines_until(
                self.connector,
                now=self.now,
                until=self.until,
            ),
            [],
        )

    def test_existing_order_and_message_id_tie_breaking_are_preserved(self):
        self.connector.search_messages.return_value = [
            gmail_message("later", deadline_subject(23, 9, "Later")),
            gmail_message("same-b", deadline_subject(22, 10, "Same B")),
            gmail_message("earlier", deadline_subject(21, 9, "Earlier")),
            gmail_message("same-a", deadline_subject(22, 10, "Same A")),
        ]

        results = find_upcoming_deadlines_until(
            self.connector,
            now=self.now,
            until=self.until,
        )

        self.assertEqual(
            [result.message_id for result in results],
            ["earlier", "same-a", "same-b", "later"],
        )

    def test_max_results_is_forwarded_unchanged(self):
        self.connector.search_messages.return_value = []

        find_upcoming_deadlines_until(
            self.connector,
            now=self.now,
            until=self.until,
            max_results=37,
        )

        self.connector.search_messages.assert_called_once_with(
            "subject:\"Venciment el\"",
            max_results=37,
        )

    def test_invalid_bounds_are_rejected_before_gmail_access(self):
        aware_now = self.now.replace(tzinfo=timezone.utc)
        aware_until = self.until.replace(tzinfo=timezone.utc)
        cases = (
            (aware_now, self.until, "now must be timezone-naive"),
            (self.now, aware_until, "until must be timezone-naive"),
            (self.now, self.now, "until must be later than now"),
            (self.now, datetime(2054, 9, 19, 12, 0), "until must be later than now"),
        )

        for now, until, message in cases:
            with self.subTest(now=now, until=until):
                self.connector.reset_mock()

                with self.assertRaisesRegex(ValueError, message):
                    find_upcoming_deadlines_until(
                        self.connector,
                        now=now,
                        until=until,
                    )

                self.connector.search_messages.assert_not_called()

    def test_gmail_exception_object_propagates_unchanged(self):
        error = RuntimeError("API failure")
        self.connector.search_messages.side_effect = error

        with self.assertRaises(RuntimeError) as caught:
            find_upcoming_deadlines_until(
                self.connector,
                now=self.now,
                until=self.until,
            )

        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
