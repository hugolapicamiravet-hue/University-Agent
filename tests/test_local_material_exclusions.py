"""Tests for host-controlled local material directory exclusions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from university_agent.local_material_search import search_local_materials
from university_agent.local_materials import LocalMaterialsSource


class LocalMaterialExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "materials"
        self.root.mkdir()

    def create_file(self, relative_path: str, text: str = "fictional") -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def relative_materials(self, source, course="Fictional Graphics"):
        return [
            material.relative_path.as_posix()
            for material in source.list_materials(course)
        ]

    def test_no_exclusions_preserves_recursive_discovery(self):
        self.create_file("Fictional Graphics/Project/Library/generated.txt")
        self.create_file("Fictional Graphics/notes/lesson.txt")

        source = LocalMaterialsSource(self.root)

        self.assertEqual(
            self.relative_materials(source),
            ["Project/Library/generated.txt", "notes/lesson.txt"],
        )

    def test_exact_relative_subtree_is_excluded_but_sibling_remains(self):
        self.create_file("Fictional Graphics/Project/Library/generated.txt")
        self.create_file("Fictional Graphics/Project/Assets/lesson.txt")

        source = LocalMaterialsSource(
            self.root,
            excluded_relative_paths=("Project/Library",),
        )

        self.assertEqual(
            self.relative_materials(source),
            ["Project/Assets/lesson.txt"],
        )

    def test_nested_exclusion_does_not_exclude_same_name_elsewhere(self):
        self.create_file("Fictional Graphics/Project/Library/generated.txt")
        self.create_file("Fictional Graphics/notes/Library/reference.txt")

        source = LocalMaterialsSource(
            self.root,
            excluded_relative_paths=(Path("Project/Library"),),
        )

        self.assertEqual(
            self.relative_materials(source),
            ["notes/Library/reference.txt"],
        )

    def test_excluded_directory_is_not_traversed(self):
        excluded = self.create_file(
            "Fictional Graphics/Project/Generated/private.txt"
        ).parent
        self.create_file("Fictional Graphics/Project/source.txt")
        source = LocalMaterialsSource(
            self.root,
            excluded_relative_paths=("Project/Generated",),
        )
        visited = []
        original_iterdir = Path.iterdir

        def tracking_iterdir(path):
            visited.append(path)
            return original_iterdir(path)

        with patch.object(Path, "iterdir", tracking_iterdir):
            materials = self.relative_materials(source)

        self.assertEqual(materials, ["Project/source.txt"])
        self.assertNotIn(excluded, visited)

    def test_exclusions_apply_to_the_same_exact_path_in_each_course(self):
        self.create_file("Course Alpha/Project/Generated/one.txt")
        self.create_file("Course Beta/Project/Generated/two.txt")
        self.create_file("Course Beta/other/Generated/kept.txt")
        source = LocalMaterialsSource(
            self.root,
            excluded_relative_paths=("Project/Generated",),
        )

        self.assertEqual(source.list_materials("Course Alpha"), [])
        self.assertEqual(
            self.relative_materials(source, "Course Beta"),
            ["other/Generated/kept.txt"],
        )

    def test_parent_traversal_exclusions_are_rejected(self):
        cases = ("../outside", "Project/../../outside")

        for value in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "excluded paths"):
                    LocalMaterialsSource(
                        self.root,
                        excluded_relative_paths=(value,),
                    )

    def test_absolute_and_empty_exclusions_are_rejected(self):
        cases = (self.root / "outside", "", ".")

        for value in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "excluded paths"):
                    LocalMaterialsSource(
                        self.root,
                        excluded_relative_paths=(value,),
                    )

    def test_hidden_and_symlink_policies_remain_unchanged(self):
        self.create_file("Fictional Graphics/.hidden/private.txt")
        self.create_file("Fictional Graphics/visible.txt")
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()
        try:
            (self.root / "Fictional Graphics/link").symlink_to(
                outside,
                target_is_directory=True,
            )
        except OSError as error:
            self.skipTest(f"symbolic links unavailable: {error}")

        source = LocalMaterialsSource(
            self.root,
            excluded_relative_paths=("Project/Generated",),
        )

        self.assertEqual(self.relative_materials(source), ["visible.txt"])

    def test_courses_and_material_ordering_remain_deterministic(self):
        self.create_file("Course Zeta/zeta.txt")
        self.create_file("Course Alpha/zeta.txt")
        self.create_file("Course Alpha/alpha.txt")
        self.create_file("Course Alpha/Generated/ignored.txt")
        source = LocalMaterialsSource(
            self.root,
            excluded_relative_paths=("Generated",),
        )

        self.assertEqual(
            [course.name for course in source.list_courses()],
            ["Course Alpha", "Course Zeta"],
        )
        self.assertEqual(
            self.relative_materials(source, "Course Alpha"),
            ["alpha.txt", "zeta.txt"],
        )

    def test_search_never_receives_excluded_generated_content(self):
        self.create_file(
            "Fictional Graphics/Generated/noise.txt",
            "academic target " * 20,
        )
        self.create_file(
            "Fictional Graphics/notes/lesson.txt",
            "academic target",
        )
        source = LocalMaterialsSource(
            self.root,
            excluded_relative_paths=("Generated",),
        )

        results = search_local_materials(source, query="academic target")

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].chunk.relative_path,
            Path("notes/lesson.txt"),
        )


if __name__ == "__main__":
    unittest.main()
