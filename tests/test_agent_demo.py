"""Isolated tests for provider selection in the command-line demo."""

import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import Mock, patch

from scripts import agent_demo


class AgentDemoTests(unittest.TestCase):
    def setUp(self):
        status_patcher = patch("scripts.agent_demo._status")
        status_patcher.start()
        self.addCleanup(status_patcher.stop)

    def _selected_ollama_model(
        self,
        query: str,
        *,
        override: str | None = None,
    ) -> str:
        arguments = [
            "agent_demo.py",
            "--materials-root",
            "/fictional/materials",
            "--timezone",
            "Europe/Madrid",
        ]
        if override is not None:
            arguments.extend(("--model", override))
        arguments.append(query)

        with (
            patch("scripts.agent_demo.LocalMaterialsSource"),
            patch("scripts.agent_demo.GmailConnector"),
            patch("scripts.agent_demo.OllamaClient"),
            patch("scripts.agent_demo.OllamaUniversityAgent") as agent_type,
            patch("sys.argv", arguments),
            patch("builtins.print"),
        ):
            agent_type.return_value.run.return_value = "Local answer"
            self.assertEqual(agent_demo.main(), 0)

        return agent_type.call_args.kwargs["model"]

    def test_explicit_material_queries_select_small_model(self):
        queries = (
            "¿Dónde hablan mis apuntes de memoria compartida?",
            "Busca en mis materiales información sobre semáforos.",
            "Busca-ho als meus apunts.",
            "Explain this using my notes.",
        )

        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(
                    self._selected_ollama_model(query),
                    "llama3.2:3b",
                )

    def test_other_queries_select_large_model(self):
        queries = (
            "¿Qué es una sección crítica?",
            "¿Qué entregas tengo esta semana?",
        )

        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(
                    self._selected_ollama_model(query),
                    "qwen3:14b",
                )

    def test_explicit_model_overrides_automatic_routing(self):
        cases = (
            (
                "¿Dónde hablan mis apuntes de memoria compartida?",
                "qwen3:14b",
            ),
            ("¿Qué es una sección crítica?", "llama3.2:3b"),
        )

        for query, override in cases:
            with self.subTest(query=query, override=override):
                self.assertEqual(
                    self._selected_ollama_model(query, override=override),
                    override,
                )

    @patch("scripts.agent_demo.LocalMaterialsSource")
    @patch("scripts.agent_demo.GmailConnector")
    @patch("scripts.agent_demo.OllamaClient")
    @patch("scripts.agent_demo.OllamaUniversityAgent")
    def test_num_predict_is_forwarded_only_to_ollama_agent(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.return_value = "Bounded answer"

        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "--num-predict",
                "256",
                "Fictional query",
            ],
        ), patch("builtins.print"):
            result = agent_demo.main()

        self.assertEqual(result, 0)
        agent_type.assert_called_once_with(
            client=client_type.return_value,
            connector=connector_type.return_value,
            materials_source=source_type.return_value,
            timezone_name="Europe/Madrid",
            model="qwen3:14b",
            num_predict=256,
        )

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
        source_type.assert_called_once_with(
            agent_demo.Path("/fictional/materials"),
            excluded_relative_paths=[],
        )
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
        source_type.assert_called_once_with(
            agent_demo.Path("/fictional/materials"),
            excluded_relative_paths=[],
        )
        agent_type.assert_called_once_with(
            client=client_type.return_value,
            connector=connector_type.return_value,
            materials_source=source_type.return_value,
            timezone_name="Europe/Madrid",
            model="fictional-openai-model",
        )
        output.assert_called_once_with("Optional answer")

    @patch("scripts.agent_demo.LocalMaterialsSource")
    @patch("scripts.agent_demo.GmailConnector")
    @patch("scripts.agent_demo.OllamaClient")
    @patch("scripts.agent_demo.OllamaUniversityAgent")
    def test_repeatable_exclusions_are_host_configuration(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.return_value = "Filtered answer"

        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--materials-root",
                "/fictional/materials",
                "--exclude",
                "Project/Library",
                "--exclude",
                "Project/Temp",
                "--timezone",
                "Europe/Madrid",
                "Fictional query",
            ],
        ), patch("builtins.print"):
            result = agent_demo.main()

        self.assertEqual(result, 0)
        source_type.assert_called_once_with(
            agent_demo.Path("/fictional/materials"),
            excluded_relative_paths=["Project/Library", "Project/Temp"],
        )
        agent_type.assert_called_once_with(
            client=client_type.return_value,
            connector=connector_type.return_value,
            materials_source=source_type.return_value,
            timezone_name="Europe/Madrid",
            model="qwen3:14b",
        )

    def test_help_explains_local_defaults_and_host_options(self):
        help_text = " ".join(
            agent_demo._build_parser().format_help().split()
        )

        for expected in (
            "local-first University Agent",
            "default: ollama",
            "llama3.2:3b",
            "qwen3:14b",
            "one subdirectory per course",
            "exact directory path relative to each course",
            "explicit model override",
            "output-token cap",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, help_text)

    @patch("scripts.agent_demo.LocalMaterialsSource")
    @patch("scripts.agent_demo.GmailConnector")
    @patch("scripts.agent_demo.OllamaClient")
    @patch("scripts.agent_demo.OllamaUniversityAgent")
    def test_status_identifies_model_and_explicit_material_search(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.return_value = "Grounded answer"

        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "¿Dónde hablan mis apuntes de semáforos?",
            ],
        ), patch("scripts.agent_demo._status") as status, patch(
            "builtins.print"
        ) as output:
            result = agent_demo.main()

        self.assertEqual(result, 0)
        self.assertEqual(
            [call.args[0] for call in status.call_args_list],
            [
                "Using local Ollama model llama3.2:3b.",
                "Searching local materials before generating the answer...",
            ],
        )
        output.assert_called_once_with("Grounded answer")

    @patch("scripts.agent_demo.LocalMaterialsSource")
    def test_missing_or_unusable_material_root_has_safe_error(self, source_type):
        arguments = [
            "agent_demo.py",
            "--materials-root",
            "/fictional/private/path",
            "--timezone",
            "Europe/Madrid",
            "Fictional query",
        ]
        for error in (
            FileNotFoundError("/fictional/private/path"),
            NotADirectoryError("/fictional/private/path"),
        ):
            with self.subTest(error=type(error).__name__), patch(
                "sys.argv", arguments
            ), patch("scripts.agent_demo._status") as status:
                source_type.side_effect = error

                self.assertEqual(agent_demo.main(), 2)
                status.assert_called_once_with(
                    "Materials root is missing or is not a usable directory."
                )
                self.assertNotIn("private", status.call_args.args[0])

    @patch("scripts.agent_demo.LocalMaterialsSource")
    def test_unknown_timezone_has_actionable_error(self, source_type):
        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Fictional/Nowhere",
                "Fictional query",
            ],
        ), patch("scripts.agent_demo._status") as status:
            result = agent_demo.main()

        self.assertEqual(result, 2)
        status.assert_called_once_with(
            "Unknown timezone. Use an IANA name such as Europe/Madrid."
        )

    @patch("scripts.agent_demo.LocalMaterialsSource")
    @patch("scripts.agent_demo.GmailConnector")
    @patch("scripts.agent_demo.OllamaClient")
    @patch("scripts.agent_demo.OllamaUniversityAgent")
    def test_ollama_provider_error_is_actionable_and_does_not_reach_stdout(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.side_effect = (
            agent_demo.OllamaUniversityAgentProviderError("sanitized")
        )

        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional query",
            ],
        ), patch("scripts.agent_demo._status") as status, patch(
            "builtins.print"
        ) as output:
            result = agent_demo.main()

        self.assertEqual(result, 1)
        self.assertIn("qwen3:14b", status.call_args.args[0])
        self.assertIn("ollama list", status.call_args.args[0])
        output.assert_not_called()

    @patch("scripts.agent_demo.LocalMaterialsSource")
    @patch("scripts.agent_demo.GmailConnector")
    @patch("scripts.agent_demo.OllamaClient")
    @patch("scripts.agent_demo.OllamaUniversityAgent")
    def test_tool_failure_has_actionable_safe_error(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.side_effect = (
            agent_demo.OllamaUniversityAgentToolError("private detail")
        )

        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional query",
            ],
        ), patch("scripts.agent_demo._status") as status:
            result = agent_demo.main()

        self.assertEqual(result, 1)
        self.assertIn("academic data operation failed", status.call_args.args[0])
        self.assertIn("Gmail OAuth", status.call_args.args[0])
        self.assertNotIn("private detail", status.call_args.args[0])

    def test_nonpositive_num_predict_is_rejected_by_cli(self):
        parser = agent_demo._build_parser()

        for value in ("0", "-1"):
            with (
                self.subTest(value=value),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(
                    [
                        "--materials-root",
                        "/fictional/materials",
                        "--timezone",
                        "Europe/Madrid",
                        "--num-predict",
                        value,
                        "Fictional query",
                    ]
                )

    @patch("scripts.agent_demo.LocalMaterialsSource")
    @patch("scripts.agent_demo.GmailConnector")
    @patch("scripts.agent_demo.OpenAI")
    def test_missing_openai_configuration_has_actionable_error(
        self,
        client_type,
        connector_type,
        source_type,
    ):
        client_type.side_effect = agent_demo.OpenAIError("private detail")

        with patch(
            "sys.argv",
            [
                "agent_demo.py",
                "--provider",
                "openai",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional query",
            ],
        ), patch("scripts.agent_demo._status") as status:
            result = agent_demo.main()

        self.assertEqual(result, 2)
        status.assert_called_once_with(
            "OpenAI is not configured. Set OPENAI_API_KEY locally or use the "
            "default Ollama provider."
        )
        self.assertNotIn("private detail", status.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
