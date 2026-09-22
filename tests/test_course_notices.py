"""Tests for deterministic course-notice subject parsing."""

import unittest

from university_agent.course_notices import (
    ParsedCourseNotice,
    parse_course_notice_subject,
)


class CourseNoticeParserTests(unittest.TestCase):
    def test_synthetic_course_notices(self):
        cases = (
            (
                'EI0008-EI0011-2054-2055: Busco equipo para el taller ficticio',
                ("EI0008", "EI0011"),
                'Busco equipo para el taller ficticio',
            ),
            (
                'EI0008-EI0011-2054-2055: Equipos del taller ficticio',
                ("EI0008", "EI0011"),
                'Equipos del taller ficticio',
            ),
            (
                'EI0007-EI0006-2054-2055: !!!80% del simulador ficticio completado!!',
                ("EI0007", "EI0006"),
                '!!!80% del simulador ficticio completado!!',
            ),
            (
                'EI0007-EI0006-2054-2055: Día a día del laboratorio ficticio',
                ("EI0007", "EI0006"),
                'Día a día del laboratorio ficticio',
            ),
            (
                'EI0007-EI0006-2054-2055: Información del aula ficticia y tutorías de prueba',
                ("EI0007", "EI0006"),
                'Información del aula ficticia y tutorías de prueba',
            ),
            (
                'EI0001-MT0001-2054-2055: Abierta la actividad ficticia número 1',
                ("EI0001", "MT0001"),
                'Abierta la actividad ficticia número 1',
            ),
            (
                'EI0009-2054-2055: Demostración ficticia del martes 22/09',
                ("EI0009",),
                'Demostración ficticia del martes 22/09',
            ),
            (
                'MT0004-EI0004-2054-2055: Proyecto ficticio de la segunda semana',
                ("MT0004", "EI0004"),
                'Proyecto ficticio de la segunda semana',
            ),
            (
                'MT0004-EI0004-2054-2055: Proyecto ficticio de la primera semana',
                ("MT0004", "EI0004"),
                'Proyecto ficticio de la primera semana',
            ),
            (
                'EI0002-2054-2055: Evaluación ficticia de la primera práctica de teoría',
                ("EI0002",),
                'Evaluación ficticia de la primera práctica de teoría',
            ),
            (
                'MT0004-EI0004-2054-2055: Publicadas las soluciones de la sesión ficticia',
                ("MT0004", "EI0004"),
                'Publicadas las soluciones de la sesión ficticia',
            ),
            (
                'EI0003-MT0003-2054-2055: Bienvenidos al laboratorio ficticio (EI0003/MT0003)',
                ("EI0003", "MT0003"),
                'Bienvenidos al laboratorio ficticio (EI0003/MT0003)',
            ),
        )

        for subject, course_codes, notice_text in cases:
            with self.subTest(subject=subject):
                self.assertEqual(
                    parse_course_notice_subject(subject),
                    ParsedCourseNotice(
                        raw_subject=subject,
                        course_codes=course_codes,
                        academic_year="2054-2055",
                        notice_text=notice_text,
                    ),
                )

    def test_three_mixed_codes_preserve_order_and_academic_year(self):
        subject = (
            'EI0005-EI0010-MT0005-2053-2054: Sobre el ensayo ficticio de Robótica del martes 16 de junio'
        )

        self.assertEqual(
            parse_course_notice_subject(subject),
            ParsedCourseNotice(
                raw_subject=subject,
                course_codes=("EI0005", "EI0010", "MT0005"),
                academic_year="2053-2054",
                notice_text=(
                    'Sobre el ensayo ficticio de Robótica del martes 16 de junio'
                ),
            ),
        )

    def test_notice_text_preserves_supported_punctuation(self):
        subject = (
            "EI0002-2054-2055: Aviso: ¡cambio de aula! "
            "(grupo EI0002/MT0002)"
        )
        result = parse_course_notice_subject(subject)

        self.assertIsNotNone(result)
        self.assertEqual(
            result.notice_text,
            "Aviso: ¡cambio de aula! (grupo EI0002/MT0002)",
        )

    def test_delimiter_and_folded_whitespace_are_normalized(self):
        subject = "  MT0004-EI0004-2054-2055 \n :\t Trabajo   semanal  "

        self.assertEqual(
            parse_course_notice_subject(subject),
            ParsedCourseNotice(
                raw_subject=subject,
                course_codes=("MT0004", "EI0004"),
                academic_year="2054-2055",
                notice_text="Trabajo semanal",
            ),
        )

    def test_decomposed_unicode_is_normalized_without_changing_raw_subject(self):
        subject = "EI0002-2054-2055: Informacio\u0301n del di\u0301a"
        result = parse_course_notice_subject(subject)

        self.assertIsNotNone(result)
        self.assertEqual(result.raw_subject, subject)
        self.assertIn("o\u0301", result.raw_subject)
        self.assertEqual(result.notice_text, "Información del día")

    def test_invalid_academic_years_return_none(self):
        subjects = (
            "EI0009-2054-55: Notice",
            "EI0009-2054/2055: Notice",
            "EI0009-2054-2056: Notice",
        )

        for subject in subjects:
            with self.subTest(subject=subject):
                self.assertIsNone(parse_course_notice_subject(subject))

    def test_invalid_course_prefixes_return_none(self):
        subjects = (
            "EI000-2054-2055: Notice",
            "EI00000-2054-2055: Notice",
            "ei0009-2054-2055: Notice",
            "COR000001-2054-2055: Notice",
            "EI0009 -MT0009-2054-2055: Notice",
            "EI0009- MT0009-2054-2055: Notice",
        )

        for subject in subjects:
            with self.subTest(subject=subject):
                self.assertIsNone(parse_course_notice_subject(subject))

    def test_non_ascii_digits_return_none(self):
        subjects = (
            "EI٠٠٠١-2054-2055: Notice",
            "EI0009-٢٠٥٤-٢٠٥٥: Notice",
        )

        for subject in subjects:
            with self.subTest(subject=subject):
                self.assertIsNone(parse_course_notice_subject(subject))

    def test_missing_notice_text_returns_none(self):
        for subject in (
            "EI0009-2054-2055:",
            "EI0009-2054-2055:   ",
        ):
            with self.subTest(subject=subject):
                self.assertIsNone(parse_course_notice_subject(subject))

    def test_unanchored_and_unrelated_subjects_return_none(self):
        subjects = (
            "",
            "   ",
            "Avís EI0009-2054-2055: Notice",
            "Re: EI0009-2054-2055: Notice",
            'Confirmació de reserva fictícia',
        )

        for subject in subjects:
            with self.subTest(subject=subject):
                self.assertIsNone(parse_course_notice_subject(subject))


if __name__ == "__main__":
    unittest.main()
