"""Small loopback-only browser chat for the existing local agent backend."""

from __future__ import annotations

import json
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from zoneinfo import ZoneInfo

from ollama import Client as OllamaClient

from university_agent.connectors.gmail import GmailConnector
from university_agent.gui_config import GuiConfiguration
from university_agent.local_agent import (
    create_ollama_agent,
    select_ollama_model,
    write_ollama_timing,
)
from university_agent.local_material_cache import LocalMaterialCache
from university_agent.local_materials import LocalMaterialsSource
from university_agent.ollama_agent import (
    OllamaUniversityAgent,
    OllamaUniversityAgentProviderError,
    OllamaUniversityAgentRoundLimitError,
    OllamaUniversityAgentToolError,
)


_MAX_REQUEST_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class ChatResult:
    """One safe response suitable for rendering in the local interface."""

    ok: bool
    answer: str | None = None
    error: str | None = None
    status: int = HTTPStatus.OK


class GuiChatService:
    """Invoke independent local-agent queries behind a narrow UI boundary."""

    def __init__(
        self,
        configuration: GuiConfiguration,
        *,
        client: Any | None = None,
        connector: GmailConnector | None = None,
        agent_factory: Callable[..., OllamaUniversityAgent] = OllamaUniversityAgent,
        now_provider: Callable[[ZoneInfo], datetime] | None = None,
    ) -> None:
        self._configuration = configuration
        self._materials_source = LocalMaterialsSource(
            configuration.materials_root,
            excluded_relative_paths=configuration.excluded_relative_paths,
        )
        self._material_cache = LocalMaterialCache()
        self._client = OllamaClient() if client is None else client
        self._connector = GmailConnector() if connector is None else connector
        self._agent_factory = agent_factory
        self._now_provider = now_provider or datetime.now
        self._timezone = ZoneInfo(configuration.timezone_name)
        self._request_lock = threading.Lock()

    def submit(self, message: str) -> ChatResult:
        """Process one stateless message and return only safe display data."""
        if not isinstance(message, str) or not message.strip():
            return ChatResult(
                ok=False,
                error="Escribe una pregunta antes de enviarla.",
                status=HTTPStatus.BAD_REQUEST,
            )
        if not self._request_lock.acquire(blocking=False):
            return ChatResult(
                ok=False,
                error="Ya hay una consulta en curso. Espera a que termine.",
                status=HTTPStatus.CONFLICT,
            )

        query = message.strip()
        model = select_ollama_model(query, self._configuration.model_override)
        try:
            agent, _ = create_ollama_agent(
                query=query,
                client=self._client,
                connector=self._connector,
                materials_source=self._materials_source,
                timezone_name=self._configuration.timezone_name,
                model_override=self._configuration.model_override,
                material_cache=self._material_cache,
                timing_callback=write_ollama_timing,
                agent_factory=self._agent_factory,
            )
            answer = agent.run(query, now=self._now_provider(self._timezone))
            return ChatResult(ok=True, answer=answer)
        except OllamaUniversityAgentProviderError as error:
            return ChatResult(
                ok=False,
                error=_provider_error_message(error, model=model),
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except OllamaUniversityAgentToolError:
            return ChatResult(
                ok=False,
                error=(
                    "No se pudo consultar la información académica. Revisa "
                    "la configuración local y, si corresponde, Gmail OAuth."
                ),
                status=HTTPStatus.BAD_GATEWAY,
            )
        except OllamaUniversityAgentRoundLimitError:
            return ChatResult(
                ok=False,
                error="La consulta necesitó demasiados pasos. Prueba a concretarla.",
                status=HTTPStatus.BAD_GATEWAY,
            )
        except Exception:
            return ChatResult(
                ok=False,
                error="University-Agent no pudo completar la consulta de forma segura.",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        finally:
            self._request_lock.release()


def create_gui_server(
    service: GuiChatService,
    *,
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a loopback-only HTTP server without starting it."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")

    handler = _handler_for(service)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def run_gui(
    configuration: GuiConfiguration,
    *,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Run the local chat until interrupted by the user."""
    service = GuiChatService(configuration)
    server = create_gui_server(service, port=port)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"University-Agent GUI available at {url}")
    print("Press Ctrl+C to stop it. Queries are independent; history is not saved.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _provider_error_message(
    error: OllamaUniversityAgentProviderError,
    *,
    model: str,
) -> str:
    cause = error.__cause__
    details = str(cause).casefold() if cause is not None else ""
    if "not found" in details or "does not exist" in details:
        return (
            f"El modelo local {model} no está instalado. "
            f"Instálalo manualmente con `ollama pull {model}`."
        )
    if any(
        marker in details
        for marker in ("connection refused", "failed to connect", "connection error")
    ):
        return "Ollama no está en ejecución. Inícialo y vuelve a intentarlo."
    return (
        f"Ollama no pudo usar el modelo {model}. Comprueba que el servicio "
        "está activo y que el modelo está instalado."
    )


def _handler_for(service: GuiChatService) -> type[BaseHTTPRequestHandler]:
    class GuiRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required HTTP handler name
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = _HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._security_headers("text/html; charset=utf-8", len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - required HTTP handler name
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = self.headers.get_content_type()
            if content_type != "application/json":
                self._write_json(
                    ChatResult(
                        False,
                        error="La solicitud debe usar JSON.",
                        status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    )
                )
                return
            length = self.headers.get("Content-Length")
            try:
                content_length = int(length or "0")
            except ValueError:
                self._write_json(
                    ChatResult(False, error="Solicitud no válida.", status=400)
                )
                return
            if not 0 < content_length <= _MAX_REQUEST_BYTES:
                self._write_json(
                    ChatResult(False, error="Solicitud vacía o demasiado grande.", status=400)
                )
                return
            try:
                payload = json.loads(self.rfile.read(content_length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._write_json(
                    ChatResult(False, error="Solicitud JSON no válida.", status=400)
                )
                return
            message = payload.get("message") if isinstance(payload, dict) else None
            self._write_json(service.submit(message))

        def _write_json(self, result: ChatResult) -> None:
            payload = {"ok": result.ok}
            if result.ok:
                payload["answer"] = result.answer
            else:
                payload["error"] = result.error
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(result.status)
            self._security_headers("application/json; charset=utf-8", len(body))
            self.end_headers()
            self.wfile.write(body)

        def _security_headers(self, content_type: str, length: int) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'; "
                "img-src 'none'; frame-src 'none'",
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return GuiRequestHandler


_HTML_PAGE = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>University Agent</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #f5f6f8; color: #202124; }
    main { max-width: 800px; margin: 0 auto; padding: 32px 20px; }
    h1 { margin: 0 0 4px; font-size: 1.6rem; }
    .note { color: #5f6368; margin: 0 0 20px; font-size: .92rem; }
    #conversation { min-height: 360px; background: white; border: 1px solid #ddd;
      border-radius: 12px; padding: 18px; overflow-y: auto; }
    .message { max-width: 88%; margin: 0 0 14px; padding: 11px 13px;
      border-radius: 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
    .user { margin-left: auto; background: #e8f0fe; }
    .assistant { background: #f1f3f4; }
    .error { background: #fce8e6; color: #8c1d18; }
    form { display: flex; gap: 10px; margin-top: 14px; align-items: flex-end; }
    textarea { flex: 1; min-height: 48px; max-height: 160px; resize: vertical;
      border: 1px solid #bbb; border-radius: 9px; padding: 11px; font: inherit; }
    button { min-height: 48px; border: 0; border-radius: 9px; padding: 0 20px;
      background: #3157a4; color: white; font-weight: 600; cursor: pointer; }
    button:disabled, textarea:disabled { opacity: .6; cursor: not-allowed; }
  </style>
</head>
<body>
<main>
  <h1>University Agent</h1>
  <p class="note">Chat local con Ollama. Cada pregunta es independiente y el historial no se guarda.</p>
  <section id="conversation" aria-live="polite" aria-label="Conversación"></section>
  <form id="chat-form">
    <textarea id="message" aria-label="Mensaje" placeholder="Escribe tu pregunta…" required></textarea>
    <button id="send" type="submit">Enviar</button>
  </form>
</main>
<script>
  const form = document.getElementById('chat-form');
  const input = document.getElementById('message');
  const send = document.getElementById('send');
  const conversation = document.getElementById('conversation');
  let busy = false;

  function addMessage(text, role) {
    const element = document.createElement('div');
    element.className = `message ${role}`;
    element.textContent = text;
    conversation.appendChild(element);
    conversation.scrollTop = conversation.scrollHeight;
    return element;
  }

  async function submitMessage() {
    const message = input.value.trim();
    if (!message || busy) return;
    busy = true;
    addMessage(message, 'user');
    input.value = '';
    input.disabled = true;
    send.disabled = true;
    const thinking = addMessage('Procesando…', 'assistant');
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message})
      });
      const payload = await response.json();
      thinking.remove();
      addMessage(payload.ok ? payload.answer : payload.error, payload.ok ? 'assistant' : 'error');
    } catch (_) {
      thinking.remove();
      addMessage('No se pudo conectar con University-Agent.', 'error');
    } finally {
      busy = false;
      input.disabled = false;
      send.disabled = false;
      input.focus();
    }
  }

  form.addEventListener('submit', event => { event.preventDefault(); submitMessage(); });
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submitMessage();
    }
  });
</script>
</body>
</html>
"""
