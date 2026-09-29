"""Isolated tests for the packaged command-line interface."""

import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from university_agent import cli
from university_agent.connectors.gmail import GmailConnector


class CliTests(unittest.TestCase):
    def setUp(self):
        self._status_patcher = patch("university_agent.cli._status")
        self._status_patcher.start()
        self.addCleanup(self._status_patcher.stop)

    def test_package_cli_import_does_not_execute_main(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import university_agent.cli"],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_help_exits_successfully_without_creating_dependencies(self):
        with (
            patch("sys.argv", ["university-agent", "--help"]),
            patch("university_agent.cli.LocalMaterialsSource") as source_type,
            redirect_stdout(StringIO()) as output,
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("local-first University Agent", output.getvalue())
        source_type.assert_not_called()

    def test_packaging_declares_console_entry_point(self):
        project_root = Path(__file__).resolve().parents[1]
        configuration = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            configuration["project"]["scripts"]["university-agent"],
            "university_agent.cli:main",
        )

    def test_compatibility_wrapper_delegates_to_package_cli(self):
        from scripts import agent_demo

        self.assertIs(agent_demo.main, cli.main)

    def test_gui_subcommand_loads_configuration_and_launches_local_ui(self):
        configuration = Mock()
        with (
            patch(
                "sys.argv",
                [
                    "university-agent",
                    "gui",
                    "--config",
                    "/fictional/config.toml",
                    "--port",
                    "9876",
                    "--no-browser",
                ],
            ),
            patch(
                "university_agent.cli.load_gui_configuration",
                return_value=configuration,
            ) as load_configuration,
            patch("university_agent.cli.run_gui") as run_gui,
        ):
            result = cli.main()

        self.assertEqual(result, 0)
        load_configuration.assert_called_once_with(
            cli.Path("/fictional/config.toml")
        )
        run_gui.assert_called_once_with(
            configuration,
            port=9876,
            open_browser=False,
        )

    def test_packaged_cli_week_query_uses_body_deadline_workflow(self):
        connector = Mock(spec=GmailConnector)
        connector.search_messages.return_value = [
            {
                "id": "fictional-message",
                "thread_id": "fictional-thread",
                "sender": "fixture1@example.com",
                "recipients": "fixture2@example.com",
                "subject": "EI9001-2026-2027: Actividad semanal ficticia",
                "date": "Tue, 29 Sep 2026 09:00:00 +0200",
                "snippet": "",
            }
        ]
        connector.get_plain_text_body.return_value = (
            "Tenéis que completar el cuestionario ficticio antes del jueves "
            "día 1 de octubre a las 08:00."
        )
        client = Mock()
        client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "Tienes un cuestionario ficticio.",
            }
        }

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "sys.argv",
                [
                    "university-agent",
                    "--materials-root",
                    directory,
                    "--timezone",
                    "Europe/Madrid",
                    "¿Qué tengo que hacer esta semana?",
                ],
            ),
            patch("university_agent.cli.GmailConnector", return_value=connector),
            patch("university_agent.cli.OllamaClient", return_value=client),
            patch("university_agent.cli.datetime") as host_datetime,
            patch("builtins.print") as output,
        ):
            host_datetime.now.return_value = datetime(
                2026,
                9,
                29,
                12,
                0,
                tzinfo=ZoneInfo("Europe/Madrid"),
            )
            result = cli.main()

        self.assertEqual(result, 0)
        connector.get_plain_text_body.assert_called_once_with("fictional-message")
        output.assert_called_once_with("Tienes un cuestionario ficticio.")

    def test_gui_help_requires_no_configuration_or_external_service(self):
        with (
            patch("sys.argv", ["university-agent", "gui", "--help"]),
            patch("university_agent.cli.load_gui_configuration") as load_configuration,
            patch("university_agent.cli.run_gui") as run_gui,
            redirect_stdout(StringIO()) as output,
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("loopback-only", output.getvalue())
        load_configuration.assert_not_called()
        run_gui.assert_not_called()

    def test_missing_gui_configuration_is_safe(self):
        with (
            patch("sys.argv", ["university-agent", "gui"]),
            patch(
                "university_agent.cli.load_gui_configuration",
                side_effect=cli.GuiConfigurationError("safe configuration error"),
            ),
            patch("university_agent.cli.run_gui") as run_gui,
            patch("university_agent.cli._status") as status,
        ):
            result = cli.main()

        self.assertEqual(result, 2)
        status.assert_called_once_with("safe configuration error")
        run_gui.assert_not_called()

    def _selected_ollama_model(
        self,
        query: str,
        *,
        override: str | None = None,
    ) -> str:
        arguments = [
            "cli.py",
            "--materials-root",
            "/fictional/materials",
            "--timezone",
            "Europe/Madrid",
        ]
        if override is not None:
            arguments.extend(("--model", override))
        arguments.append(query)

        with (
            patch("university_agent.cli.LocalMaterialsSource"),
            patch("university_agent.cli.GmailConnector"),
            patch("university_agent.cli.OllamaClient"),
            patch("university_agent.cli.OllamaUniversityAgent") as agent_type,
            patch("sys.argv", arguments),
            patch("builtins.print"),
        ):
            agent_type.return_value.run.return_value = "Local answer"
            self.assertEqual(cli.main(), 0)

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

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OllamaClient")
    @patch("university_agent.cli.OllamaUniversityAgent")
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
                "cli.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "--num-predict",
                "256",
                "Fictional query",
            ],
        ), patch("builtins.print"):
            result = cli.main()

        self.assertEqual(result, 0)
        agent_type.assert_called_once_with(
            client=client_type.return_value,
            connector=connector_type.return_value,
            materials_source=source_type.return_value,
            timezone_name="Europe/Madrid",
            model="qwen3:14b",
            num_predict=256,
        )

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OllamaClient")
    @patch("university_agent.cli.OllamaUniversityAgent")
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
                "cli.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional",
                "query",
            ],
        ), patch("builtins.print") as output:
            result = cli.main()

        self.assertEqual(result, 0)
        source_type.assert_called_once_with(
            cli.Path("/fictional/materials"),
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

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OpenAI")
    @patch("university_agent.cli.UniversityAgent")
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
                "cli.py",
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
            result = cli.main()

        self.assertEqual(result, 0)
        source_type.assert_called_once_with(
            cli.Path("/fictional/materials"),
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

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OllamaClient")
    @patch("university_agent.cli.OllamaUniversityAgent")
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
                "cli.py",
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
            result = cli.main()

        self.assertEqual(result, 0)
        source_type.assert_called_once_with(
            cli.Path("/fictional/materials"),
            excluded_relative_paths=["Project/Library", "Project/Temp"],
        )
        agent_type.assert_called_once_with(
            client=client_type.return_value,
            connector=connector_type.return_value,
            materials_source=source_type.return_value,
            timezone_name="Europe/Madrid",
            model="qwen3:14b",
        )

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OllamaClient")
    @patch("university_agent.cli.OllamaUniversityAgent")
    def test_status_uses_stderr_and_answer_uses_stdout(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.return_value = "Fictional answer"
        self._status_patcher.stop()
        try:
            with (
                patch(
                    "sys.argv",
                    [
                        "university-agent",
                        "--materials-root",
                        "/fictional/materials",
                        "--timezone",
                        "Europe/Madrid",
                        "Fictional query",
                    ],
                ),
                redirect_stdout(StringIO()) as output,
                redirect_stderr(StringIO()) as status,
            ):
                result = cli.main()
        finally:
            self._status_patcher.start()

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "Fictional answer\n")
        self.assertEqual(
            status.getvalue(),
            "Using local Ollama model qwen3:14b.\n"
            "Generating the answer...\n",
        )

    def test_help_explains_local_defaults_and_host_options(self):
        help_text = " ".join(
            cli._build_parser().format_help().split()
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

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OllamaClient")
    @patch("university_agent.cli.OllamaUniversityAgent")
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
                "cli.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "¿Dónde hablan mis apuntes de semáforos?",
            ],
        ), patch("university_agent.cli._status") as status, patch(
            "builtins.print"
        ) as output:
            result = cli.main()

        self.assertEqual(result, 0)
        self.assertEqual(
            [call.args[0] for call in status.call_args_list],
            [
                "Using local Ollama model llama3.2:3b.",
                "Searching local materials before generating the answer...",
            ],
        )
        output.assert_called_once_with("Grounded answer")

    @patch("university_agent.cli.LocalMaterialsSource")
    def test_missing_or_unusable_material_root_has_safe_error(self, source_type):
        arguments = [
            "cli.py",
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
            ), patch("university_agent.cli._status") as status:
                source_type.side_effect = error

                self.assertEqual(cli.main(), 2)
                status.assert_called_once_with(
                    "Materials root is missing or is not a usable directory."
                )
                self.assertNotIn("private", status.call_args.args[0])

    @patch("university_agent.cli.LocalMaterialsSource")
    def test_unknown_timezone_has_actionable_error(self, source_type):
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Fictional/Nowhere",
                "Fictional query",
            ],
        ), patch("university_agent.cli._status") as status:
            result = cli.main()

        self.assertEqual(result, 2)
        status.assert_called_once_with(
            "Unknown timezone. Use an IANA name such as Europe/Madrid."
        )

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OllamaClient")
    @patch("university_agent.cli.OllamaUniversityAgent")
    def test_ollama_provider_error_is_actionable_and_does_not_reach_stdout(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.side_effect = (
            cli.OllamaUniversityAgentProviderError("sanitized")
        )

        with patch(
            "sys.argv",
            [
                "cli.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional query",
            ],
        ), patch("university_agent.cli._status") as status, patch(
            "builtins.print"
        ) as output:
            result = cli.main()

        self.assertEqual(result, 1)
        self.assertIn("qwen3:14b", status.call_args.args[0])
        self.assertIn("ollama list", status.call_args.args[0])
        output.assert_not_called()

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OllamaClient")
    @patch("university_agent.cli.OllamaUniversityAgent")
    def test_tool_failure_has_actionable_safe_error(
        self,
        agent_type,
        client_type,
        connector_type,
        source_type,
    ):
        agent_type.return_value.run.side_effect = (
            cli.OllamaUniversityAgentToolError("private detail")
        )

        with patch(
            "sys.argv",
            [
                "cli.py",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional query",
            ],
        ), patch("university_agent.cli._status") as status:
            result = cli.main()

        self.assertEqual(result, 1)
        self.assertIn("academic data operation failed", status.call_args.args[0])
        self.assertIn("Gmail OAuth", status.call_args.args[0])
        self.assertNotIn("private detail", status.call_args.args[0])

    def test_nonpositive_num_predict_is_rejected_by_cli(self):
        parser = cli._build_parser()

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

    @patch("university_agent.cli.LocalMaterialsSource")
    @patch("university_agent.cli.GmailConnector")
    @patch("university_agent.cli.OpenAI")
    def test_missing_openai_configuration_has_actionable_error(
        self,
        client_type,
        connector_type,
        source_type,
    ):
        client_type.side_effect = cli.OpenAIError("private detail")

        with patch(
            "sys.argv",
            [
                "cli.py",
                "--provider",
                "openai",
                "--materials-root",
                "/fictional/materials",
                "--timezone",
                "Europe/Madrid",
                "Fictional query",
            ],
        ), patch("university_agent.cli._status") as status:
            result = cli.main()

        self.assertEqual(result, 2)
        status.assert_called_once_with(
            "OpenAI is not configured. Set OPENAI_API_KEY locally or use the "
            "default Ollama provider."
        )
        self.assertNotIn("private detail", status.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
