"""Host-controlled Ollama chat adapter for academic operations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from university_agent.agent_tools import (
    SYSTEM_INSTRUCTIONS,
    AgentToolRuntime,
    AgentToolRuntimeError,
    ToolArgumentError,
    invalid_arguments_payload,
)
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_materials import LocalMaterialsSource


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
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be positive")

        self._client = client
        self._model = model
        self._max_tool_rounds = max_tool_rounds
        self._runtime = AgentToolRuntime(
            connector=connector,
            materials_source=materials_source,
            timezone_name=timezone_name,
            max_gmail_results=max_gmail_results,
            max_lookback_days=max_lookback_days,
            max_material_results=max_material_results,
            max_chunk_chars=max_chunk_chars,
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
        tool_rounds = 0

        while True:
            response = self._chat(messages)
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
                return content

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
                    try:
                        payload = self._runtime.execute(
                            name,
                            arguments,
                            now=now,
                        )
                    except AgentToolRuntimeError as error:
                        raise OllamaUniversityAgentToolError(
                            "Academic operation failed"
                        ) from error

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )

    def _chat(self, messages: list[Any]) -> Any:
        try:
            return self._client.chat(
                model=self._model,
                messages=messages,
                tools=self._tools,
                think=False,
            )
        except Exception as error:
            raise OllamaUniversityAgentProviderError(
                "Ollama chat request failed"
            ) from error


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
