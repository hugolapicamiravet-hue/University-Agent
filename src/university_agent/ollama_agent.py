"""Host-controlled Ollama chat adapter for academic operations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

from university_agent.agent_tools import (
    SYSTEM_INSTRUCTIONS,
    AgentToolRuntime,
    AgentToolRuntimeError,
    ToolArgumentError,
    append_material_sources,
    invalid_arguments_payload,
    material_citations_from_payload,
    required_material_search_arguments,
    required_remaining_week_deadline_arguments,
)
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_material_cache import LocalMaterialCache
from university_agent.local_materials import LocalMaterialsSource


@dataclass(frozen=True, slots=True)
class OllamaRunTiming:
    """Privacy-safe aggregate timing for one local-agent request."""

    model: str
    total_seconds: float
    tool_seconds: float
    gmail_seconds: float
    material_seconds: float
    model_seconds: float
    model_rounds: int
    response_chars: int


class OllamaUniversityAgentError(RuntimeError):
    """Base class for safe host-level Ollama agent failures."""


class OllamaUniversityAgentProviderError(OllamaUniversityAgentError):
    """The local Ollama client failed or returned no final text."""


class OllamaUniversityAgentToolError(OllamaUniversityAgentError):
    """A deterministic application operation failed internally."""


class OllamaUniversityAgentRoundLimitError(OllamaUniversityAgentError):
    """The model exceeded the host-controlled tool-round limit."""


class OllamaUniversityAgent:
    """Run a bounded local Ollama tool loop over academic operations."""

    def __init__(
        self,
        *,
        client: Any,
        connector: GmailConnector,
        materials_source: LocalMaterialsSource,
        timezone_name: str,
        model: str = "qwen3:14b",
        max_gmail_results: int = 100,
        max_lookback_days: int = 120,
        max_material_results: int = 10,
        max_chunk_chars: int = 2000,
        max_tool_rounds: int = 4,
        num_predict: int | None = None,
        material_cache: LocalMaterialCache | None = None,
        timing_callback: Callable[[OllamaRunTiming], None] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be positive")
        if num_predict is not None and (
            isinstance(num_predict, bool)
            or not isinstance(num_predict, int)
            or num_predict < 1
        ):
            raise ValueError("num_predict must be a positive integer or None")

        self._client = client
        self._model = model
        self._max_tool_rounds = max_tool_rounds
        self._num_predict = num_predict
        self._timing_callback = timing_callback
        self._runtime = AgentToolRuntime(
            connector=connector,
            materials_source=materials_source,
            timezone_name=timezone_name,
            max_gmail_results=max_gmail_results,
            max_lookback_days=max_lookback_days,
            max_material_results=max_material_results,
            max_chunk_chars=max_chunk_chars,
            material_cache=material_cache,
        )
        self._tools = [
            {"type": "function", "function": definition}
            for definition in self._runtime.definitions
        ]

    def run(self, user_input: str, *, now: datetime) -> str:
        """Return one final local-model answer after a bounded tool loop."""
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must not be blank")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")

        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_input},
        ]
        citations = []
        tool_rounds = 0
        model_rounds = 0
        model_seconds = 0.0
        tool_times = {"gmail": 0.0, "materials": 0.0, "other": 0.0}
        started = perf_counter()

        def execute_runtime(
            name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            operation_started = perf_counter()
            try:
                return self._execute_runtime(name, arguments, now=now)
            finally:
                elapsed = perf_counter() - operation_started
                if name in {
                    "get_remaining_week_deadlines",
                    "get_recent_course_notices",
                }:
                    tool_times["gmail"] += elapsed
                elif name == "search_local_materials":
                    tool_times["materials"] += elapsed
                else:
                    tool_times["other"] += elapsed

        def finish(answer: str) -> str:
            if self._timing_callback is not None:
                self._timing_callback(
                    OllamaRunTiming(
                        model=self._model,
                        total_seconds=perf_counter() - started,
                        tool_seconds=sum(tool_times.values()),
                        gmail_seconds=tool_times["gmail"],
                        material_seconds=tool_times["materials"],
                        model_seconds=model_seconds,
                        model_rounds=model_rounds,
                        response_chars=len(answer),
                    )
                )
            return answer

        deadline_arguments = required_remaining_week_deadline_arguments(user_input)
        if deadline_arguments is not None:
            deadline_payload = execute_runtime(
                "get_remaining_week_deadlines",
                deadline_arguments,
            )
            return finish(_format_remaining_week_deadlines(deadline_payload))

        required_arguments = required_material_search_arguments(user_input)
        if required_arguments is not None:
            payload = execute_runtime(
                "search_local_materials",
                required_arguments,
            )
            citations.extend(material_citations_from_payload(payload))
            messages.extend(
                (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_local_materials",
                                    "arguments": required_arguments,
                                }
                            }
                        ],
                    },
                    _tool_message("search_local_materials", payload),
                )
            )

        while True:
            model_started = perf_counter()
            response = self._chat(messages)
            model_seconds += perf_counter() - model_started
            model_rounds += 1
            message = _value(response, "message")
            if message is None:
                raise OllamaUniversityAgentProviderError(
                    "Ollama response contained invalid message"
                )

            tool_calls = _value(message, "tool_calls") or []
            if not isinstance(tool_calls, list):
                raise OllamaUniversityAgentProviderError(
                    "Ollama response contained invalid tool calls"
                )
            if not tool_calls:
                content = _value(message, "content")
                if not isinstance(content, str) or not content.strip():
                    raise OllamaUniversityAgentProviderError(
                        "Ollama response did not contain final text"
                    )
                return finish(append_material_sources(content, citations))

            if tool_rounds >= self._max_tool_rounds:
                raise OllamaUniversityAgentRoundLimitError(
                    "Ollama tool-round limit exceeded"
                )
            tool_rounds += 1
            messages.append(message)

            for tool_call in tool_calls:
                function = _value(tool_call, "function")
                name = _value(function, "name")
                arguments = _value(function, "arguments")
                if not isinstance(name, str):
                    payload = invalid_arguments_payload(
                        ToolArgumentError("tool name must be a string")
                    )
                    tool_name = "invalid_tool"
                elif not isinstance(arguments, dict):
                    payload = invalid_arguments_payload(
                        ToolArgumentError("tool arguments must be a JSON object")
                    )
                    tool_name = name
                else:
                    tool_name = name
                    payload = execute_runtime(name, arguments)

                if tool_name == "search_local_materials":
                    citations.extend(material_citations_from_payload(payload))
                messages.append(_tool_message(tool_name, payload))

    def _execute_runtime(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        now: datetime,
    ) -> dict[str, Any]:
        try:
            return self._runtime.execute(name, arguments, now=now)
        except AgentToolRuntimeError as error:
            raise OllamaUniversityAgentToolError(
                "Academic operation failed"
            ) from error

    def _chat(self, messages: list[Any]) -> Any:
        request = {
            "model": self._model,
            "messages": messages,
            "tools": self._tools,
            "think": False,
        }
        if self._num_predict is not None:
            request["options"] = {"num_predict": self._num_predict}
        try:
            return self._client.chat(**request)
        except Exception as error:
            raise OllamaUniversityAgentProviderError(
                "Ollama chat request failed"
            ) from error


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _tool_message(
    tool_name: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_name": tool_name,
        "content": json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ),
    }


def _format_remaining_week_deadlines(payload: dict[str, Any]) -> str:
    """Render the verified Spanish week query without another model pass."""
    results = payload.get("results") if payload.get("ok") is True else None
    if not isinstance(results, list) or not results:
        return (
            "No he detectado plazos académicos pendientes para lo que queda "
            "de esta semana."
        )

    lines = ["Plazos académicos detectados para esta semana:"]
    for result in results:
        title = result["title"]
        due_at = datetime.fromisoformat(result["due_at"])
        lines.append(f"- {title} — {due_at:%d/%m/%Y %H:%M}")
    return "\n".join(lines)
