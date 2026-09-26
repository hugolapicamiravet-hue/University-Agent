"""Privacy-safe acceptance test for the packaged fictional-material demo."""

import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from university_agent import cli
from university_agent.connectors.gmail import GmailConnector


class FakeOllamaClient:
    """Capture one local-model request and return a fictional final answer."""

    def __init__(self) -> None:
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "message": {
                "role": "assistant",
                "content": (
                    "La exclusión mutua protege una sección crítica frente "
                    "a accesos concurrentes."
                ),
            }
        }


class FictionalDemoAcceptanceTests(unittest.TestCase):
    def test_packaged_host_flow_searches_samples_without_external_access(self):
        project_root = Path(__file__).resolve().parents[1]
        materials_root = project_root / "examples" / "materials"
        fake_client = FakeOllamaClient()
        connector = Mock(spec=GmailConnector)
        arguments = [
            "university-agent",
            "--materials-root",
            str(materials_root),
            "--timezone",
            "Europe/Madrid",
            "Explícame exclusión mutua usando mis apuntes",
        ]

        with (
            patch("sys.argv", arguments),
            patch("university_agent.cli.OllamaClient", return_value=fake_client),
            patch("university_agent.cli.GmailConnector", return_value=connector),
            patch("university_agent.cli.OpenAI") as openai_client_type,
            redirect_stdout(StringIO()) as output,
            redirect_stderr(StringIO()) as status,
        ):
            result = cli.main()

        self.assertEqual(result, 0)
        self.assertEqual(len(fake_client.calls), 1)
        request = fake_client.calls[0]
        self.assertEqual(request["model"], "llama3.2:3b")
        self.assertIn("Searching local materials", status.getvalue())
        self.assertNotIn("OpenAI", status.getvalue())
        openai_client_type.assert_not_called()
        self.assertEqual(connector.method_calls, [])

        tool_messages = [
            message
            for message in request["messages"]
            if message.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        payload = json.loads(tool_messages[0]["content"])
        self.assertIs(payload["ok"], True)
        relative_paths = {
            result["relative_path"] for result in payload["results"]
        }
        self.assertIn("mutual_exclusion.txt", relative_paths)
        self.assertIn("scheduling.txt", relative_paths)

        answer = output.getvalue()
        self.assertIn("Fuentes consultadas:", answer)
        self.assertIn(
            "[Operating Systems / mutual_exclusion.txt]",
            answer,
        )
        self.assertIn("[Operating Systems / scheduling.txt]", answer)
        self.assertNotIn(str(project_root), answer)
        self.assertNotIn(
            str(project_root),
            json.dumps(request, ensure_ascii=False),
        )


if __name__ == "__main__":
    unittest.main()
