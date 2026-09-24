"""Isolated tests for provider selection in the command-line demo."""

import unittest
from unittest.mock import Mock, patch

from scripts import agent_demo


class AgentDemoTests(unittest.TestCase):
    @patch("scripts.agent_demo.LocalMaterialsSource")
    @patch("scripts.agent_demo.GmailConnector")
    @patch("scripts.agent_demo.OllamaClient")
    @patch("scripts.agent_demo.OllamaUniversityAgent")
    def test_ollama_is_the_default_provider(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.return_value = "Local answer"

        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional",
                "query",
            ],
        ), patch("builtins.print") as output:
            result = agent_demo.main()

        self.assertEqual(result, 0)
        agent_type.assert_called_once_with(
            client=client_type.return_value,
            connector=connector_type.return_value,
            materials_source=source_type.return_value,
            timezone_name="Europe/Madrid",
            model="qwen3:14b",
        )
        output.assert_called_once_with("Local answer")

    @patch("scripts.agent_demo.LocalMaterialsSource")
    @patch("scripts.agent_demo.GmailConnector")
    @patch("scripts.agent_demo.OpenAI")
    @patch("scripts.agent_demo.UniversityAgent")
    def test_openai_remains_an_explicit_option(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.return_value = "Optional answer"

        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--provider",
                "openai",
                "--model",
                "fictional-openai-model",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional query",
            ],
        ), patch("builtins.print") as output:
            result = agent_demo.main()

        self.assertEqual(result, 0)
        agent_type.assert_called_once_with(
            client=client_type.return_value,
            connector=connector_type.return_value,
            materials_source=source_type.return_value,
            timezone_name="Europe/Madrid",
            model="fictional-openai-model",
        )
        output.assert_called_once_with("Optional answer")


if __name__ == "__main__":
    unittest.main()
