"""Pure tests for conservative academic announcement body parsing."""

import unittest
from datetime import datetime

from university_agent.academic_body_deadlines import (
    AcademicBodyDeadline,
    parse_academic_body_deadlines,
)


class AcademicBodyDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.message_at = datetime(2054, 9, 25, 9, 0)

    def parse(self, body: str, *, message_at: datetime | None = None):
        return parse_academic_body_deadlines(
            body,
            message_at=self.message_at if message_at is None else message_at,
        )

    def test_strong_spanish_before_pattern(self):
        results = self.parse(
            "Tenéis que subir el entregable fx_t03 antes del próximo jueves "
            "día 1 de octubre a las 8:00."
        )

        self.assertEqual(
            results,
            [
                AcademicBodyDeadline(
                    title="subir el entregable fx_t03",
                    due_at=datetime(2054, 10, 1, 8, 0),
                    date_text="jueves día 1 de octubre a las 8:00",
                )
            ],
        )

    def test_strong_valencian_before_pattern(self):
        results = self.parse(
            "Heu de completar el qüestionari abans del proper dijous dia 1 "
            "d'octubre a les 8:00."
        )

        self.assertEqual(results[0].title, "completar el qüestionari")
        self.assertEqual(results[0].due_at, datetime(2054, 10, 1, 8, 0))

    def test_explicit_year_does_not_need_message_date(self):
        results = parse_academic_body_deadlines(
            "Fecha límite: jueves, 1 de octubre de 2054, 08:00. "
            "Entregar la práctica ficticia.",
            message_at=None,
        )

        self.assertEqual(results[0].due_at, datetime(2054, 10, 1, 8, 0))

    def test_missing_year_uses_message_year(self):
        results = self.parse(
            "Entregar el informe antes del jueves día 1 de octubre a las 08:00."
        )

        self.assertEqual(results[0].due_at.year, 2054)

    def test_missing_year_rolls_across_year_boundary(self):
        results = self.parse(
            "Subir la memoria antes del lunes día 4 de enero a las 09:30.",
            message_at=datetime(2054, 12, 20, 10, 0),
        )

        self.assertEqual(results[0].due_at, datetime(2055, 1, 4, 9, 30))

    def test_missing_year_without_message_date_is_rejected(self):
        self.assertEqual(
            parse_academic_body_deadlines(
                "Entregar la práctica antes del jueves día 1 de octubre a las 8:00.",
                message_at=None,
            ),
            [],
        )

    def test_multiple_actions_with_shared_deadline_are_separate(self):
        results = self.parse(
            "Tenéis que subir el informe y completar el cuestionario antes del "
            "jueves día 1 de octubre a las 8:00."
        )

        self.assertEqual(
            [result.title for result in results],
            ["completar el cuestionario", "subir el informe"],
        )
        self.assertEqual(len({result.due_at for result in results}), 1)

    def test_deadline_heading_can_precede_a_short_action_list(self):
        results = self.parse(
            "Antes del próximo jueves día 1 de octubre a las 8:00.\n"
            "Deberéis realizar estas acciones:\n\n"
            "Subir el informe ficticio.\n"
            "Información complementaria sin otra fecha.\n"
            "Completar el cuestionario ficticio."
        )

        self.assertEqual(
            [result.title for result in results],
            ["Completar el cuestionario ficticio", "Subir el informe ficticio"],
        )

    def test_explicit_questionnaire_deadline(self):
        results = self.parse(
            "Debéis completar el cuestionario de tiempo antes del viernes día "
            "2 de octubre a las 18:15."
        )

        self.assertEqual(results[0].title, "completar el cuestionario de tiempo")

    def test_parentheses_in_explicit_action_are_preserved(self):
        results = self.parse(
            "Subir el entregable ficticio (parte A) antes del jueves día 1 de "
            "octubre a las 08:00."
        )

        self.assertEqual(
            results[0].title,
            "Subir el entregable ficticio (parte A)",
        )

    def test_vague_phrases_are_rejected(self):
        for body in (
            "Tenéis que entregar la actividad esta semana.",
            "Subid la respuesta próximamente.",
            "Completad el formulario cuanto antes.",
            "Entregar el ejercicio para la siguiente clase.",
        ):
            with self.subTest(body=body):
                self.assertEqual(self.parse(body), [])

    def test_general_dated_notice_without_action_is_rejected(self):
        self.assertEqual(
            self.parse(
                "La sesión informativa será el jueves, 1 de octubre de 2054, 08:00."
            ),
            [],
        )

    def test_example_action_and_date_are_rejected(self):
        self.assertEqual(
            self.parse(
                "Ejemplo: entregar el informe antes del jueves día 1 de "
                "octubre a las 08:00."
            ),
            [],
        )

    def test_test_mention_without_submission_action_is_rejected(self):
        self.assertEqual(
            self.parse(
                "El test se realizará el jueves día 1 de octubre a las 08:00."
            ),
            [],
        )

    def test_past_tense_completion_is_rejected(self):
        self.assertEqual(
            self.parse(
                "Habéis completado el cuestionario antes del jueves día 1 de "
                "octubre a las 08:00."
            ),
            [],
        )

    def test_quoted_previous_message_is_ignored(self):
        self.assertEqual(
            self.parse(
                "Recordatorio general.\n"
                "> Tenéis que entregar la práctica antes del jueves día 1 de "
                "octubre a las 08:00."
            ),
            [],
        )

    def test_invalid_dates_do_not_raise(self):
        self.assertEqual(
            self.parse(
                "Entregar la práctica antes del jueves día 31 de febrero a las 28:90."
            ),
            [],
        )

    def test_raw_body_is_not_retained(self):
        result = self.parse(
            "Entregar el informe antes del jueves día 1 de octubre a las 08:00."
        )[0]

        self.assertFalse(hasattr(result, "body"))
        self.assertFalse(hasattr(result, "raw_body"))


if __name__ == "__main__":
    unittest.main()
