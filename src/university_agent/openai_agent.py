"""Host-controlled OpenAI Responses API adapter for academic operations."""

from __future__ import annotations

import json
from datetime import datetime
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
)
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_materials import LocalMaterialsSource


class UniversityAgentError(RuntimeError):
    """Base class for safe host-level agent failures."""


class UniversityAgentProviderError(UniversityAgentError):
    """The configured OpenAI client failed or returned no final text."""


class UniversityAgentToolError(UniversityAgentError):
    """A deterministic application operation failed internally."""


class UniversityAgentRoundLimitError(UniversityAgentError):
    """The model exceeded the host-controlled tool-round limit."""


class UniversityAgent:
    """Run a narrow OpenAI tool loop over established academic operations."""

    def __init__(
        self,
        *,
        client: Any,
        connector: GmailConnector,
        materials_source: LocalMaterialsSource,
        timezone_name: str,
        model: str = "gpt-5.6-luna",
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
            {
                "type": "function",
                **definition,
                "strict": True,
            }
            for definition in self._runtime.definitions
        ]

    def run(self, user_input: str, *, now: datetime) -> str:
        """Return one final model answer after a bounded local tool loop."""
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must not be blank")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")

        input_items: list[Any] = [{"role": "user", "content": user_input}]
        citations = []
        tool_rounds = 0

        required_arguments = required_material_search_arguments(user_input)
        if required_arguments is not None:
            payload = self._execute_runtime(
                "search_local_materials",
                required_arguments,
                now=now,
            )
            citations.extend(material_citations_from_payload(payload))
            input_items.extend(
                (
                    {
                        "type": "function_call",
                        "call_id": "host-required-material-search",
                        "name": "search_local_materials",
                        "arguments": json.dumps(required_arguments),
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "host-required-material-search",
                        "output": _encode_payload(payload),
                    },
                )
            )

        while True:
            response = self._create_response(input_items)
            output = getattr(response, "output", None)
            if not isinstance(output, list):
                raise UniversityAgentProviderError(
                    "OpenAI response contained invalid output"
                )

            tool_calls = [
                item
                for item in output
                if getattr(item, "type", None) == "function_call"
            ]
            if not tool_calls:
                output_text = getattr(response, "output_text", None)
                if not isinstance(output_text, str) or not output_text.strip():
                    raise UniversityAgentProviderError(
                        "OpenAI response did not contain final text"
                    )
                return append_material_sources(output_text, citations)

            if tool_rounds >= self._max_tool_rounds:
                raise UniversityAgentRoundLimitError(
                    "OpenAI tool-round limit exceeded"
                )
            tool_rounds += 1

            input_items.extend(output)
            for tool_call in tool_calls:
                output_payload = self._execute_tool_call(tool_call, now=now)
                if tool_call.name == "search_local_materials":
                    citations.extend(material_citations_from_payload(output_payload))
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": _encode_payload(output_payload),
                    }
                )

    def _create_response(self, input_items: list[Any]) -> Any:
        try:
            return self._client.responses.create(
                model=self._model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=input_items,
                tools=self._tools,
                parallel_tool_calls=True,
                store=False,
            )
        except Exception as error:
            raise UniversityAgentProviderError(
                "OpenAI response request failed"
            ) from error

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
            raise UniversityAgentToolError(
                "Academic operation failed"
            ) from error

    def _execute_tool_call(
        self,
        tool_call: Any,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        try:
            arguments = _decode_arguments(tool_call.arguments)
        except ToolArgumentError as error:
            return invalid_arguments_payload(error)

        return self._execute_runtime(tool_call.name, arguments, now=now)


def _encode_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _decode_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ToolArgumentError("tool arguments must be JSON")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ToolArgumentError("tool arguments must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ToolArgumentError("tool arguments must be a JSON object")
    return decoded
