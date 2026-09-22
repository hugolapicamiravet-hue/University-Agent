"""Tests for deterministic academic notification subject parsing."""

import unittest
from datetime import datetime

from university_agent.academic_notifications import (
    NotificationCategory,
    ParsedNotification,
    parse_notification_subject,
)


class AcademicNotificationParserTests(unittest.TestCase):
    def test_deadline_with_prefixed_title(self):
        subject = (
            'Venciment el dimarts, 22 de setembre 2054, 19:00:\nZ9X | Diseñando una biblioteca ficticia'
        )

        self.assertEqual(
            parse_notification_subject(subject),
            ParsedNotification(
                category=NotificationCategory.DEADLINE,
                raw_subject=subject,
                title='Z9X | Diseñando una biblioteca ficticia',
                date_text="dimarts, 22 de setembre 2054, 19:00",
                event_at=datetime(2054, 9, 22, 19, 0),
            ),
        )

    def test_deadline_preserves_fx_t02(self):
        subject = (
            'Venciment el dijous, 24 de setembre 2054, 08:00:\nEntrega del ejercicio ficticio fx_t02'
        )
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.DEADLINE)
        self.assertEqual(result.title, 'Entrega del ejercicio ficticio fx_t02')
        self.assertEqual(result.event_at, datetime(2054, 9, 24, 8, 0))

    def test_overdue_task(self):
        subject = (
            'Tasca vençuda:\nVídeo sobre una biblioteca ficticia (fecha límite 20/09/2054 a las 10:00)'
        )
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.TASK_OVERDUE)
        self.assertEqual(result.title, 'Vídeo sobre una biblioteca ficticia')
        self.assertEqual(result.date_text, "20/09/2054 a las 10:00")
        self.assertEqual(result.event_at, datetime(2054, 9, 20, 10, 0))

    def test_overdue_task_preserves_parenthesized_title_text(self):
        subject = (
            "Tasca vençuda: Vídeo (part 1) "
            "(fecha límite 20/09/2054 a las 10:00)"
        )
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.TASK_OVERDUE)
        self.assertEqual(result.title, "Vídeo (part 1)")
        self.assertEqual(result.date_text, "20/09/2054 a las 10:00")
        self.assertEqual(result.event_at, datetime(2054, 9, 20, 10, 0))

    def test_overdue_tasks_without_explicit_deadline_metadata(self):
        cases = (
            (
                'Tasca vençuda:  Entrega ficticia 2. 15 sep. 2054',
                'Entrega ficticia 2. 15 sep. 2054',
            ),
            (
                'Tasca vençuda:  Entrega ficticia 1. 8 sep. 2054',
                'Entrega ficticia 1. 8 sep. 2054',
            ),
            (
                'Tasca vençuda:  Lliurament fictici de la 2ª convocatòria',
                'Lliurament fictici de la 2ª convocatòria',
            ),
            (
                'Tasca vençuda:  Lliurament fictici dels equips',
                'Lliurament fictici dels equips',
            ),
        )

        for subject, expected_title in cases:
            with self.subTest(subject=subject):
                result = parse_notification_subject(subject)
                self.assertEqual(result.category, NotificationCategory.TASK_OVERDUE)
                self.assertEqual(result.title, expected_title)
                self.assertIsNone(result.date_text)
                self.assertIsNone(result.event_at)
                self.assertEqual(result.raw_subject, subject)

    def test_generic_overdue_title_preserves_parentheses_and_colons(self):
        subject = "Tasca vençuda: Vídeo (part 1): preparació"
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.TASK_OVERDUE)
        self.assertEqual(result.title, "Vídeo (part 1): preparació")
        self.assertIsNone(result.date_text)
        self.assertIsNone(result.event_at)

    def test_nonanchored_overdue_variant_is_unknown(self):
        subject = "Avís: Tasca vençuda: Activitat"

        self.assertEqual(
            parse_notification_subject(subject).category,
            NotificationCategory.UNKNOWN,
        )

    def test_activity_opens(self):
        subject = (
            'Opens on dilluns, 21 de setembre 2054, 12:00:\nTest de Práctica Ficticia 1.'
        )
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.ACTIVITY_OPENS)
        self.assertEqual(result.title, 'Test de Práctica Ficticia 1.')
        self.assertEqual(result.event_at, datetime(2054, 9, 21, 12, 0))

    def test_task_submission_confirmation(self):
        subject = (
            'Heu realitzat la tramesa de la tasca\nEntrega del ejercicio ficticio fx_t01'
        )
        result = parse_notification_subject(subject)

        self.assertEqual(
            result.category,
            NotificationCategory.TASK_SUBMISSION_CONFIRMED,
        )
        self.assertEqual(result.title, 'Entrega del ejercicio ficticio fx_t01')
        self.assertIsNone(result.event_at)

    def test_upcoming_tasks_summary(self):
        result = parse_notification_subject("Teniu tasques que vencen en 7 dies")

        self.assertEqual(
            result.category,
            NotificationCategory.UPCOMING_TASKS_SUMMARY,
        )
        self.assertEqual(result.days_until_due, 7)
        self.assertIsNone(result.title)

    def test_account_login_notice(self):
        result = parse_notification_subject(
            "Inici de sessió nou del vostre compte Aula Virtual UJI"
        )

        self.assertEqual(
            result.category,
            NotificationCategory.ACCOUNT_LOGIN_NOTICE,
        )

    def test_raw_subject_is_preserved_while_whitespace_is_normalized(self):
        subject = (
            "  Venciment el   dimarts, 22 de setembre 2054, 19:00:\n\t"
            "Títol   amb espais  "
        )
        result = parse_notification_subject(subject)

        self.assertEqual(result.raw_subject, subject)
        self.assertEqual(result.category, NotificationCategory.DEADLINE)
        self.assertEqual(result.title, "Títol amb espais")

    def test_unrelated_uji_subjects_are_unknown(self):
        subjects = (
            'Confirmació de reserva fictícia',
            'Factura fictícia',
            'Notificació de compra fictícia',
        )

        for subject in subjects:
            with self.subTest(subject=subject):
                result = parse_notification_subject(subject)
                self.assertEqual(result.category, NotificationCategory.UNKNOWN)
                self.assertEqual(result.raw_subject, subject)

    def test_loose_keywords_are_unknown(self):
        for subject in ("examen", "entrega", "Recordatori examen i entrega"):
            with self.subTest(subject=subject):
                self.assertEqual(
                    parse_notification_subject(subject).category,
                    NotificationCategory.UNKNOWN,
                )

    def test_impossible_localized_date_does_not_raise(self):
        subject = "Venciment el dilluns, 31 de febrer 2054, 10:00: Activitat"
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.DEADLINE)
        self.assertEqual(result.date_text, "dilluns, 31 de febrer 2054, 10:00")
        self.assertIsNone(result.event_at)

    def test_impossible_numeric_date_does_not_raise(self):
        subject = "Tasca vençuda: Activitat (fecha límite 31/02/2054 a las 10:00)"
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.TASK_OVERDUE)
        self.assertEqual(result.date_text, "31/02/2054 a las 10:00")
        self.assertIsNone(result.event_at)

    def test_unknown_valencian_month_does_not_raise(self):
        subject = "Opens on dilluns, 21 de desconegut 2054, 12:00: Activitat"
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.ACTIVITY_OPENS)
        self.assertEqual(result.date_text, "dilluns, 21 de desconegut 2054, 12:00")
        self.assertIsNone(result.event_at)

    def test_all_supported_valencian_months(self):
        months = (
            ("gener", 1),
            ("febrer", 2),
            ("març", 3),
            ("abril", 4),
            ("maig", 5),
            ("juny", 6),
            ("juliol", 7),
            ("agost", 8),
            ("setembre", 9),
            ("octubre", 10),
            ("novembre", 11),
            ("desembre", 12),
        )

        for month_name, month_number in months:
            with self.subTest(month=month_name):
                subject = (
                    f"Venciment el dilluns, 1 de {month_name} 2054, 09:30: "
                    "Activitat"
                )
                result = parse_notification_subject(subject)
                self.assertEqual(result.category, NotificationCategory.DEADLINE)
                self.assertEqual(
                    result.event_at,
                    datetime(2054, month_number, 1, 9, 30),
                )

    def test_malformed_times_do_not_raise(self):
        cases = (
            (
                "Venciment el dimarts, 22 de setembre 2054, 25:99: Activitat",
                NotificationCategory.DEADLINE,
                "dimarts, 22 de setembre 2054, 25:99",
            ),
            (
                "Tasca vençuda: Activitat "
                "(fecha límite 20/09/2054 a las hora)",
                NotificationCategory.TASK_OVERDUE,
                "20/09/2054 a las hora",
            ),
        )

        for subject, expected_category, expected_date_text in cases:
            with self.subTest(subject=subject):
                result = parse_notification_subject(subject)
                self.assertEqual(result.category, expected_category)
                self.assertEqual(result.date_text, expected_date_text)
                self.assertIsNone(result.event_at)

    def test_decomposed_unicode_matches_without_changing_raw_subject(self):
        subject = (
            "Tasca venc\u0327uda: Activitat "
            "(fecha límite 20/09/2054 a las 10:00)"
        )
        result = parse_notification_subject(subject)

        self.assertEqual(result.category, NotificationCategory.TASK_OVERDUE)
        self.assertEqual(result.raw_subject, subject)
        self.assertIn("c\u0327", result.raw_subject)

    def test_additional_colons_in_title_are_preserved(self):
        subject = (
            "Venciment el dimarts, 22 de setembre 2054, 19:00: "
            "Pràctica: part 1: preparació"
        )

        self.assertEqual(
            parse_notification_subject(subject).title,
            "Pràctica: part 1: preparació",
        )

    def test_weekday_mismatch_does_not_invalidate_date(self):
        subject = "Venciment el dilluns, 22 de setembre 2054, 19:00: Activitat"

        self.assertEqual(
            parse_notification_subject(subject).event_at,
            datetime(2054, 9, 22, 19, 0),
        )

    def test_empty_and_whitespace_subjects_are_unknown(self):
        for subject in ("", "   ", "\n\t"):
            with self.subTest(subject=subject):
                result = parse_notification_subject(subject)
                self.assertEqual(result.category, NotificationCategory.UNKNOWN)
                self.assertEqual(result.raw_subject, subject)

    def test_partial_templates_are_unknown(self):
        subjects = (
            "Venciment el dimarts, 22 de setembre 2054, 19:00:",
            "Tasca vençuda:",
            "Tasca vençuda:   ",
            "Heu realitzat la tramesa de la tasca",
            "Teniu tasques que vencen en set dies",
        )

        for subject in subjects:
            with self.subTest(subject=subject):
                self.assertEqual(
                    parse_notification_subject(subject).category,
                    NotificationCategory.UNKNOWN,
                )


if __name__ == "__main__":
    unittest.main()
