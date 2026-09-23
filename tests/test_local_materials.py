"""Tests for deterministic local course-material discovery."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from university_agent.local_materials import LocalMaterialsSource


class LocalMaterialsSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.root = self.base / "materials"
        self.root.mkdir()

    def create_file(self, relative_path: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fictional material", encoding="utf-8")
        return path

    def test_empty_root_returns_no_courses(self):
        source = LocalMaterialsSource(self.root)

        self.assertEqual(source.list_courses(), [])

    def test_immediate_directories_are_courses_and_names_are_preserved(self):
        course_path = self.root / "Curso Ficticio Ámbar"
        course_path.mkdir()

        courses = LocalMaterialsSource(self.root).list_courses()

        self.assertEqual(len(courses), 1)
        self.assertEqual(courses[0].name, "Curso Ficticio Ámbar")
        self.assertEqual(courses[0].path, course_path.resolve())

    def test_root_files_are_not_courses(self):
        self.create_file("readme.txt")

        self.assertEqual(LocalMaterialsSource(self.root).list_courses(), [])

    def test_courses_are_sorted_by_name(self):
        for name in ("Curso Zeta", "Curso Alfa", "Curso Medio"):
            (self.root / name).mkdir()

        courses = LocalMaterialsSource(self.root).list_courses()

        self.assertEqual(
            [course.name for course in courses],
            ["Curso Alfa", "Curso Medio", "Curso Zeta"],
        )

    def test_exact_course_lists_material_metadata(self):
        material_path = self.create_file("Curso Alfa/tema-01.pdf")

        materials = LocalMaterialsSource(self.root).list_materials("Curso Alfa")

        self.assertEqual(len(materials), 1)
        material = materials[0]
        self.assertEqual(material.course_name, "Curso Alfa")
        self.assertEqual(material.relative_path, Path("tema-01.pdf"))
        self.assertEqual(material.path, material_path.resolve())
        self.assertEqual(material.filename, "tema-01.pdf")
        self.assertEqual(material.extension, ".pdf")

    def test_nested_directories_are_traversed_recursively(self):
        self.create_file("Curso Alfa/practicas/bloque-1/practica-01.txt")

        materials = LocalMaterialsSource(self.root).list_materials("Curso Alfa")

        self.assertEqual(
            [material.relative_path for material in materials],
            [Path("practicas/bloque-1/practica-01.txt")],
        )

    def test_nested_relative_paths_are_preserved(self):
        self.create_file("Curso Alfa/teoria/unidad-01/apuntes.md")

        material = LocalMaterialsSource(self.root).list_materials("Curso Alfa")[0]

        self.assertEqual(
            material.relative_path,
            Path("teoria/unidad-01/apuntes.md"),
        )

    def test_materials_are_sorted_by_relative_path(self):
        for relative_path in (
            "Curso Alfa/zeta.txt",
            "Curso Alfa/practicas/beta.txt",
            "Curso Alfa/alfa.txt",
        ):
            self.create_file(relative_path)

        materials = LocalMaterialsSource(self.root).list_materials("Curso Alfa")

        self.assertEqual(
            [material.relative_path.as_posix() for material in materials],
            ["alfa.txt", "practicas/beta.txt", "zeta.txt"],
        )

    def test_hidden_courses_files_and_directories_are_ignored(self):
        (self.root / ".Curso Oculto").mkdir()
        self.create_file("Curso Alfa/.DS_Store")
        self.create_file("Curso Alfa/.privado/secreto.txt")
        self.create_file("Curso Alfa/visible.txt")

        source = LocalMaterialsSource(self.root)

        self.assertEqual(
            [course.name for course in source.list_courses()],
            ["Curso Alfa"],
        )
        self.assertEqual(
            [
                material.relative_path
                for material in source.list_materials("Curso Alfa")
            ],
            [Path("visible.txt")],
        )

    def test_nonexistent_root_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            LocalMaterialsSource(self.base / "missing")

    def test_file_root_is_rejected(self):
        file_root = self.base / "not-a-directory.txt"
        file_root.write_text("fictional", encoding="utf-8")

        with self.assertRaisesRegex(NotADirectoryError, "not a directory"):
            LocalMaterialsSource(file_root)

    def test_unknown_course_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown course"):
            LocalMaterialsSource(self.root).list_materials("Curso Inexistente")

    def test_parent_path_traversal_is_rejected(self):
        (self.base / "outside").mkdir()

        with self.assertRaisesRegex(ValueError, "unknown course"):
            LocalMaterialsSource(self.root).list_materials("../outside")

    def test_absolute_course_path_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()

        with self.assertRaisesRegex(ValueError, "unknown course"):
            LocalMaterialsSource(self.root).list_materials(str(outside))

    def test_symlinks_cannot_escape_the_materials_root(self):
        external = self.base / "external"
        external.mkdir()
        external_file = external / "private.txt"
        external_file.write_text("fictional", encoding="utf-8")
        safe_course = self.root / "Curso Alfa"
        safe_course.mkdir()
        self.create_file("Curso Alfa/local.txt")

        try:
            (self.root / "Curso Enlazado").symlink_to(
                external,
                target_is_directory=True,
            )
            (safe_course / "external-directory").symlink_to(
                external,
                target_is_directory=True,
            )
            (safe_course / "external-file.txt").symlink_to(external_file)
        except OSError as error:
            self.skipTest(f"symbolic links unavailable: {error}")

        source = LocalMaterialsSource(self.root)

        self.assertEqual(
            [course.name for course in source.list_courses()],
            ["Curso Alfa"],
        )
        self.assertEqual(
            [
                material.relative_path
                for material in source.list_materials("Curso Alfa")
            ],
            [Path("local.txt")],
        )

    def test_discovery_does_not_open_material_files(self):
        self.create_file("Curso Alfa/tema-01.pdf")
        source = LocalMaterialsSource(self.root)

        with patch.object(
            Path,
            "open",
            side_effect=AssertionError("material content was opened"),
        ):
            materials = source.list_materials("Curso Alfa")

        self.assertEqual(
            [material.relative_path for material in materials],
            [Path("tema-01.pdf")],
        )


if __name__ == "__main__":
    unittest.main()
