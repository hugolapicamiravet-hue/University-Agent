"""Tests for deterministic lexical search over material chunks."""

import unittest
from pathlib import Path

from university_agent.material_chunks import MaterialChunk
from university_agent.material_search import search_chunks


def material_chunk(
    text: str,
    *,
    course_name: str = "Fictional Course Alpha",
    relative_path: str = "notes.txt",
    page_number: int | None = None,
    chunk_index: int = 0,
) -> MaterialChunk:
    return MaterialChunk(
        course_name=course_name,
        relative_path=Path(relative_path),
        format=".txt" if page_number is None else ".pdf",
        text=text,
        page_number=page_number,
        chunk_index=chunk_index,
    )


class MaterialSearchTests(unittest.TestCase):
    def test_one_matching_chunk_is_returned_with_same_provenance(self):
        chunk = material_chunk("Fictional concurrency notes")

        results = search_chunks([chunk], "concurrency")

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].chunk, chunk)
        self.assertGreater(results[0].score, 0)
        self.assertFalse(hasattr(results[0].chunk, "path"))

    def test_no_matches_returns_empty_list(self):
        chunks = [material_chunk("Fictional operating systems notes")]

        self.assertEqual(search_chunks(chunks, "networks"), [])

    def test_blank_or_term_free_query_is_rejected(self):
        for query in ("", "   \n", "!!!"):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "query"):
                    search_chunks([], query)

    def test_non_positive_limit_is_rejected(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "positive"):
                    search_chunks([], "fictional", limit=limit)

    def test_matching_is_case_insensitive(self):
        chunks = [material_chunk("Programación CONCURRENTE")]

        results = search_chunks(chunks, "programación concurrente")

        self.assertEqual(len(results), 1)

    def test_punctuation_separates_terms(self):
        chunks = [material_chunk("datos,estructuras;algoritmos")]

        results = search_chunks(chunks, "datos estructuras algoritmos")

        self.assertEqual(len(results), 1)

    def test_accents_are_preserved(self):
        chunks = [material_chunk("Programació fictícia")]

        self.assertEqual(len(search_chunks(chunks, "programació")), 1)
        self.assertEqual(search_chunks(chunks, "programacio"), [])

    def test_more_distinct_query_terms_outrank_repeated_single_term(self):
        repeated = material_chunk(
            "algoritmo algoritmo algoritmo algoritmo algoritmo",
            relative_path="repeated.txt",
        )
        distinct = material_chunk(
            "algoritmo distribuido",
            relative_path="distinct.txt",
        )

        results = search_chunks([repeated, distinct], "algoritmo distribuido")

        self.assertEqual(
            [result.chunk.relative_path for result in results],
            [Path("distinct.txt"), Path("repeated.txt")],
        )

    def test_more_occurrences_break_ties_in_distinct_terms(self):
        once = material_chunk("red", relative_path="once.txt")
        repeated = material_chunk("red red red", relative_path="repeated.txt")

        results = search_chunks([once, repeated], "red")

        self.assertEqual(
            [result.chunk.relative_path for result in results],
            [Path("repeated.txt"), Path("once.txt")],
        )
        self.assertGreater(results[0].score, results[1].score)

    def test_ties_use_course_path_page_and_chunk_index(self):
        chunks = [
            material_chunk(
                "shared",
                course_name="Course Beta",
                relative_path="a.pdf",
                page_number=1,
            ),
            material_chunk(
                "shared",
                course_name="Course Alpha",
                relative_path="b.pdf",
                page_number=1,
            ),
            material_chunk(
                "shared",
                course_name="Course Alpha",
                relative_path="a.pdf",
                page_number=2,
                chunk_index=2,
            ),
            material_chunk(
                "shared",
                course_name="Course Alpha",
                relative_path="a.pdf",
                page_number=1,
                chunk_index=1,
            ),
            material_chunk(
                "shared",
                course_name="Course Alpha",
                relative_path="a.pdf",
                page_number=1,
                chunk_index=0,
            ),
        ]

        results = search_chunks(reversed(chunks), "shared")

        self.assertEqual(
            [
                (
                    result.chunk.course_name,
                    result.chunk.relative_path.as_posix(),
                    result.chunk.page_number,
                    result.chunk.chunk_index,
                )
                for result in results
            ],
            [
                ("Course Alpha", "a.pdf", 1, 0),
                ("Course Alpha", "a.pdf", 1, 1),
                ("Course Alpha", "a.pdf", 2, 2),
                ("Course Alpha", "b.pdf", 1, 0),
                ("Course Beta", "a.pdf", 1, 0),
            ],
        )

    def test_limit_applies_after_ranking(self):
        chunks = [
            material_chunk("term", relative_path="b.txt"),
            material_chunk("term term", relative_path="a.txt"),
        ]

        results = search_chunks(chunks, "term", limit=1)

        self.assertEqual(
            [result.chunk.relative_path for result in results],
            [Path("a.txt")],
        )

    def test_generator_input_is_supported(self):
        chunks = (
            material_chunk(text, relative_path=f"{index}.txt")
            for index, text in enumerate(("match", "other"))
        )

        results = search_chunks(chunks, "match")

        self.assertEqual(len(results), 1)

    def test_empty_chunk_collection_returns_empty_list(self):
        self.assertEqual(search_chunks([], "fictional"), [])


if __name__ == "__main__":
    unittest.main()
