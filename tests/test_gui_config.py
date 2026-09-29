"""Tests for the minimal local GUI configuration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from university_agent.gui_config import (
    GuiConfigurationError,
    default_config_path,
    load_gui_configuration,
)


class GuiConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.materials = self.root / "materials"
        self.materials.mkdir()
        self.config = self.root / "config.toml"

    def write_config(self, text: str) -> None:
        self.config.write_text(text, encoding="utf-8")

    def test_loads_relative_materials_and_optional_settings(self):
        self.write_config(
            "[university_agent]\n"
            'materials_root = "materials"\n'
            'timezone = "Europe/Madrid"\n'
            'model = "fictional-local-model"\n'
            'exclude = ["private"]\n'
        )

        configuration = load_gui_configuration(self.config)

        self.assertEqual(configuration.materials_root, self.materials.resolve())
        self.assertEqual(configuration.timezone_name, "Europe/Madrid")
        self.assertEqual(configuration.model_override, "fictional-local-model")
        self.assertEqual(configuration.excluded_relative_paths, ("private",))

    def test_missing_configuration_has_safe_actionable_error(self):
        with self.assertRaisesRegex(
            GuiConfigurationError,
            "config.example.toml",
        ) as raised:
            load_gui_configuration(self.root / "missing.toml")

        self.assertNotIn(str(self.root), str(raised.exception))

    def test_missing_or_non_directory_materials_root_is_rejected(self):
        cases = ("missing", "file.txt")
        (self.root / "file.txt").write_text("not a directory", encoding="utf-8")
        for value in cases:
            with self.subTest(value=value):
                self.write_config(
                    "[university_agent]\n"
                    f'materials_root = "{value}"\n'
                    'timezone = "Europe/Madrid"\n'
                )
                with self.assertRaisesRegex(
                    GuiConfigurationError,
                    "missing or is not a directory",
                ):
                    load_gui_configuration(self.config)

    def test_invalid_timezone_is_rejected(self):
        self.write_config(
            "[university_agent]\n"
            'materials_root = "materials"\n'
            'timezone = "Fictional/Nowhere"\n'
        )

        with self.assertRaisesRegex(GuiConfigurationError, "valid IANA"):
            load_gui_configuration(self.config)

    def test_default_path_supports_environment_override(self):
        path = default_config_path(
            environment={"UNIVERSITY_AGENT_CONFIG": "~/fictional-config.toml"}
        )

        self.assertEqual(path.name, "fictional-config.toml")
        self.assertTrue(path.is_absolute())

    def test_default_path_is_per_user_config_without_repository_assumption(self):
        with patch("university_agent.gui_config.Path.home", return_value=self.root):
            path = default_config_path(environment={})

        self.assertEqual(
            path,
            self.root / ".config" / "university-agent" / "config.toml",
        )


if __name__ == "__main__":
    unittest.main()
