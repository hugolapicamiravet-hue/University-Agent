"""Isolated tests for the local browser chat presentation layer."""

import http.client
import json
import tempfile
import threading
import unittest
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from university_agent.gui import (
    ChatResult,
    GuiChatService,
    _HTML_PAGE,
    create_gui_server,
)
from university_agent.gui_config import GuiConfiguration
from university_agent.ollama_agent import OllamaUniversityAgentProviderError


class FakeAgent:
    def __init__(self, answer: str = "Fictional answer") -> None:
        self.answer = answer
        self.calls = []
        self.error: Exception | None = None

    def run(self, message: str, *, now: datetime) -> str:
        self.calls.append((message, now))
        if self.error is not None:
            raise self.error
        return self.answer


class GuiChatServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.materials = Path(self.temporary_directory.name) / "materials"
        self.materials.mkdir()
        self.configuration = GuiConfiguration(
            materials_root=self.materials,
            timezone_name="Europe/Madrid",
        )
        self.agent = FakeAgent()
        self.factory = Mock(return_value=self.agent)
        self.now = datetime(2054, 9, 20, 10, 0, tzinfo=ZoneInfo("Europe/Madrid"))
        self.service = GuiChatService(
            self.configuration,
            client=Mock(),
            connector=Mock(),
            agent_factory=self.factory,
            now_provider=lambda timezone: self.now,
        )

    def test_empty_message_is_rejected_without_backend_invocation(self):
        for message in ("", "   "):
            with self.subTest(message=message):
                result = self.service.submit(message)
                self.assertFalse(result.ok)
                self.assertIn("Escribe una pregunta", result.error)
        self.factory.assert_not_called()

    def test_successful_answer_and_sources_are_preserved_exactly(self):
        answer = (
            "Respuesta ficticia.\n\n"
            "Fuentes consultadas:\n- [Fictional Course / notes.txt]"
        )
        self.agent.answer = answer

        result = self.service.submit("  Busca en mis apuntes un tema ficticio  ")

        self.assertTrue(result.ok)
        self.assertEqual(result.answer, answer)
        self.assertEqual(self.agent.calls, [("Busca en mis apuntes un tema ficticio", self.now)])
        self.assertEqual(
            {field.name for field in fields(ChatResult)},
            {"ok", "answer", "error", "status"},
        )

    def test_automatic_routing_and_override_are_preserved(self):
        self.service.submit("Busca en mis apuntes información ficticia")
        self.assertEqual(self.factory.call_args.kwargs["model"], "llama3.2:3b")

        override_factory = Mock(return_value=FakeAgent())
        override_service = GuiChatService(
            GuiConfiguration(
                materials_root=self.materials,
                timezone_name="Europe/Madrid",
                model_override="fictional-model",
            ),
            client=Mock(),
            connector=Mock(),
            agent_factory=override_factory,
            now_provider=lambda timezone: self.now,
        )
        override_service.submit("Busca en mis apuntes información ficticia")

        self.assertEqual(
            override_factory.call_args.kwargs["model"],
            "fictional-model",
        )

    def test_material_cache_is_reused_across_independent_messages(self):
        self.service.submit("Busca en mis apuntes información ficticia")
        first_cache = self.factory.call_args.kwargs["material_cache"]

        self.service.submit("Busca en mis apuntes otro tema ficticio")
        second_cache = self.factory.call_args.kwargs["material_cache"]

        self.assertIs(first_cache, second_cache)
        self.assertIs(
            self.factory.call_args.kwargs["timing_callback"],
            self.factory.call_args_list[0].kwargs["timing_callback"],
        )

    def test_backend_error_is_safe_and_has_no_traceback(self):
        self.agent.error = RuntimeError("private backend detail")

        result = self.service.submit("Fictional question")

        self.assertFalse(result.ok)
        self.assertNotIn("private backend detail", result.error)
        self.assertNotIn("Traceback", result.error)

    def test_missing_model_error_is_actionable_without_private_details(self):
        provider_error = OllamaUniversityAgentProviderError("safe")
        provider_error.__cause__ = RuntimeError("model not found: private host detail")
        self.agent.error = provider_error

        result = self.service.submit("Fictional question")

        self.assertFalse(result.ok)
        self.assertIn("ollama pull qwen3:14b", result.error)
        self.assertNotIn("private host detail", result.error)

    def test_concurrent_duplicate_submission_is_rejected(self):
        self.service._request_lock.acquire()
        self.addCleanup(self.service._request_lock.release)

        result = self.service.submit("Fictional question")

        self.assertFalse(result.ok)
        self.assertIn("consulta en curso", result.error)
        self.factory.assert_not_called()

    def test_gui_module_has_no_openai_specific_surface(self):
        self.assertNotIn("OpenAI", _HTML_PAGE)
        self.assertIn("Chat local con Ollama", _HTML_PAGE)
        self.assertIn("Cada pregunta es independiente", _HTML_PAGE)
        self.assertIn("Procesando…", _HTML_PAGE)


class GuiHttpTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.submit.return_value = ChatResult(
            ok=True,
            answer="Fictional multiline answer\nFuentes consultadas:\n- [notes.txt]",
        )
        self.server = create_gui_server(self.service, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method: str, path: str, body: bytes | None = None):
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.server.server_address[1],
            timeout=2,
        )
        headers = {"Content-Type": "application/json"} if body else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        content = response.read()
        connection.close()
        return response, content

    def test_page_and_chat_endpoint(self):
        page_response, page = self.request("GET", "/")
        self.assertEqual(page_response.status, 200)
        self.assertIn(b"University Agent", page)
        self.assertEqual(page_response.getheader("Cache-Control"), "no-store")

        request = json.dumps({"message": "Fictional question"}).encode()
        chat_response, body = self.request("POST", "/api/chat", request)
        payload = json.loads(body)

        self.assertEqual(chat_response.status, 200)
        self.assertEqual(payload["answer"], self.service.submit.return_value.answer)
        self.service.submit.assert_called_once_with("Fictional question")

    def test_invalid_json_has_safe_error_and_no_traceback(self):
        response, body = self.request("POST", "/api/chat", b"not-json")

        self.assertEqual(response.status, 400)
        self.assertNotIn(b"Traceback", body)
        self.service.submit.assert_not_called()

    def test_non_json_post_is_rejected_before_service_invocation(self):
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.server.server_address[1],
            timeout=2,
        )
        connection.request(
            "POST",
            "/api/chat",
            body=b'{"message": "Fictional question"}',
            headers={"Content-Type": "text/plain"},
        )
        response = connection.getresponse()
        body = response.read()
        connection.close()

        self.assertEqual(response.status, 415)
        self.assertIn("JSON", json.loads(body)["error"])
        self.service.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
