"""Tests for deterministic material routing and source attribution."""

import unittest

from university_agent.agent_tools import (
    MaterialCitation,
    append_material_sources,
    is_explicit_material_query,
    material_citations_from_payload,
    required_material_search_arguments,
)


class ExplicitMaterialQueryTests(unittest.TestCase):
    def test_strong_spanish_material_cues_match(self):
        queries = (
            "¿Dónde hablan mis apuntes de memoria caché?",
            "Explícame exclusión mutua usando mis apuntes.",
            "Según mis apuntes, ¿qué es ACPI?",
            "Busca en mis materiales información sobre semáforos.",
        )

        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(is_explicit_material_query(query))

    def test_supported_valencian_and_english_cues_match(self):
        queries = (
            "Busca-ho als meus apunts",
            "Segons els meus materials, què és ACPI?",
            "Explain this using my notes",
            "Search my course materials for semaphores",
        )

        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(is_explicit_material_query(query))

    def test_matching_is_case_whitespace_and_unicode_robust(self):
        query = "SEGONS\nELS MEUS APUNTS: programacio\u0301 concurrent"

        self.assertTrue(is_explicit_material_query(query))

    def test_vague_or_unrelated_queries_do_not_match(self):
        queries = (
            "¿Qué es memoria caché?",
            "Explícame exclusión mutua.",
            "Hola",
            "¿Qué entregas tengo esta semana?",
            "¿Hay avisos recientes de EI0001?",
            "Busca apuntes públicos sobre redes",
            "Materiales de construcción",
            "permis apuntes",
        )

        for query in queries:
            with self.subTest(query=query):
                self.assertFalse(is_explicit_material_query(query))

    def test_required_arguments_preserve_the_original_query(self):
        query = "  Según MIS apuntes, ¿qué es ACPI?  "

        self.assertEqual(
            required_material_search_arguments(query),
            {"query": query, "course_name": None, "limit": None},
        )
        self.assertIsNone(required_material_search_arguments("¿Qué es ACPI?"))


class MaterialCitationTests(unittest.TestCase):
    def test_pdf_and_txt_results_create_relative_citations(self):
        payload = {
            "ok": True,
            "results": [
                {
                    "course_name": "Fictional Systems",
                    "relative_path": "unit-02/cache.pdf",
                    "page_number": 7,
                },
                {
                    "course_name": "Fictional Systems",
                    "relative_path": "notes.txt",
                    "page_number": None,
                },
            ],
        }

        citations = material_citations_from_payload(payload)

        self.assertEqual(
            citations,
            (
                MaterialCitation("Fictional Systems", "unit-02/cache.pdf", 7),
                MaterialCitation("Fictional Systems", "notes.txt", None),
            ),
        )
        answer = append_material_sources("Model answer", list(citations))
        self.assertEqual(
            answer,
            "Model answer\n\nFuentes consultadas:\n"
            "- [Fictional Systems / unit-02/cache.pdf, p. 7]\n"
            "- [Fictional Systems / notes.txt]",
        )

    def test_duplicate_sources_deduplicate_in_first_seen_order(self):
        first = MaterialCitation("Course B", "b.pdf", 2)
        second = MaterialCitation("Course A", "a.txt", None)

        answer = append_material_sources("Answer", [first, first, second, first])

        self.assertLess(answer.index("Course B"), answer.index("Course A"))
        self.assertEqual(answer.count("Course B / b.pdf, p. 2"), 1)

    def test_answer_is_unchanged_without_retrieved_sources(self):
        self.assertEqual(append_material_sources("Answer  ", []), "Answer  ")
        self.assertEqual(
            material_citations_from_payload({"ok": True, "results": []}),
            (),
        )

    def test_absolute_parent_and_malformed_provenance_are_not_exposed(self):
        payload = {
            "ok": True,
            "results": [
                {
                    "course_name": "Fictional",
                    "relative_path": "/private/notes.txt",
                    "page_number": None,
                },
                {
                    "course_name": "Fictional",
                    "relative_path": "../notes.txt",
                    "page_number": None,
                },
                {
                    "course_name": "Fictional",
                    "relative_path": "C:\\private\\notes.txt",
                    "page_number": None,
                },
                {"course_name": "Fictional"},
            ],
        }

        self.assertEqual(material_citations_from_payload(payload), ())

    def test_multiline_fields_are_rendered_on_one_line(self):
        payload = {
            "ok": True,
            "results": [
                {
                    "course_name": "Fictional\nSystems",
                    "relative_path": "unit-02/\ncache.pdf",
                    "page_number": 3,
                }
            ],
        }

        answer = append_material_sources(
            "Answer",
            list(material_citations_from_payload(payload)),
        )

        self.assertIn("[Fictional Systems / unit-02/ cache.pdf, p. 3]", answer)
        self.assertNotIn("Fictional\nSystems", answer)


if __name__ == "__main__":
    unittest.main()
