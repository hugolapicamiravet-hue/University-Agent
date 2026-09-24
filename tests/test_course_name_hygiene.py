"""Tests for normalized course lookup with raw provenance preservation."""

import tempfile
import unittest
import unicodedata
from pathlib import Path

from university_agent.local_material_search import search_local_materials
from university_agent.local_materials import LocalMaterialsSource


class CourseNameHygieneTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "materials"
        self.root.mkdir()

    def create_material(self, course_name: str) -> Path:
        course = self.root / course_name
        course.mkdir()
        material = course / "notes.txt"
        material.write_text("fictional concurrency topic", encoding="utf-8")
        return material

    def test_composed_lookup_matches_decomposed_directory(self):
        raw_name = "Programacio\u0301n Ficticia"
        self.create_material(raw_name)
        source = LocalMaterialsSource(self.root)

        materials = source.list_materials("Programación Ficticia")

        self.assertEqual(materials[0].course_name, raw_name)
        self.assertEqual(source.list_courses()[0].name, raw_name)

    def test_surrounding_whitespace_and_case_are_lookup_only(self):
        raw_name = "Fictional Systems "
        material_path = self.create_material(raw_name)
        source = LocalMaterialsSource(self.root)

        materials = source.list_materials("  FICTIONAL SYSTEMS  ")

        self.assertEqual(materials[0].course_name, raw_name)
        self.assertEqual(materials[0].path, material_path.resolve())
        self.assertEqual(materials[0].relative_path, Path("notes.txt"))
        self.assertEqual((self.root / raw_name).name, raw_name)

    def test_normalized_collision_is_rejected_as_ambiguous(self):
        first = "Fictional Course"
        second = unicodedata.normalize("NFD", "fictional course") + " "
        self.create_material(first)
        self.create_material(second)
        source = LocalMaterialsSource(self.root)

        for lookup in (first, "fictional course", " FICTIONAL COURSE "):
            with self.subTest(lookup=lookup):
                with self.assertRaisesRegex(ValueError, "ambiguous course"):
                    source.list_materials(lookup)

    def test_unknown_course_behavior_is_unchanged(self):
        self.create_material("Fictional Course")

        with self.assertRaisesRegex(ValueError, "unknown course"):
            LocalMaterialsSource(self.root).list_materials("Other Course")

    def test_search_accepts_normalized_name_and_returns_raw_provenance(self):
        raw_name = "Programacio\u0301n Ficticia "
        self.create_material(raw_name)
        source = LocalMaterialsSource(self.root)

        results = search_local_materials(
            source,
            query="concurrency",
            course_name="programación ficticia",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.course_name, raw_name)
        self.assertEqual(results[0].chunk.relative_path, Path("notes.txt"))


if __name__ == "__main__":
    unittest.main()
