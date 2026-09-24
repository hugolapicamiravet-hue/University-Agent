"""Regression coverage for material-discovery work in batch search."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from university_agent.local_material_search import search_local_materials
from university_agent.local_materials import LocalMaterialsSource


class LocalMaterialSearchDiscoveryTests(unittest.TestCase):
    def test_each_course_is_discovered_once_for_multiple_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "materials"
            course = root / "Fictional Course"
            course.mkdir(parents=True)
            for index in range(3):
                (course / f"notes-{index}.txt").write_text(
                    "fictional shared term",
                    encoding="utf-8",
                )
            source = LocalMaterialsSource(root)

            with patch.object(
                source,
                "list_materials",
                wraps=source.list_materials,
            ) as list_materials:
                results = search_local_materials(source, query="shared")

            self.assertEqual(len(results), 3)
            list_materials.assert_called_once_with("Fictional Course")


if __name__ == "__main__":
    unittest.main()
