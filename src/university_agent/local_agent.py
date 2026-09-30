"""Shared host construction for the local Ollama University Agent."""

from __future__ import annotations

from collections.abc import Callable
import sys
from typing import Any

from university_agent.agent_tools import is_explicit_material_query
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_materials import LocalMaterialsSource
from university_agent.local_material_cache import LocalMaterialCache
from university_agent.ollama_agent import OllamaRunTiming, OllamaUniversityAgent


OLLAMA_MATERIAL_MODEL = "llama3.2:3b"
OLLAMA_DEFAULT_MODEL = "qwen3:14b"


def select_ollama_model(query: str, override: str | None = None) -> str:
    """Select the local model while preserving an explicit host override."""
    if override is not None:
        return override
    if is_explicit_material_query(query):
        return OLLAMA_MATERIAL_MODEL
    return OLLAMA_DEFAULT_MODEL


def create_ollama_agent(
    *,
    query: str,
    client: Any,
    connector: GmailConnector,
    materials_source: LocalMaterialsSource,
    timezone_name: str,
    model_override: str | None = None,
    num_predict: int | None = None,
    material_cache: LocalMaterialCache | None = None,
    timing_callback: Callable[[OllamaRunTiming], None] | None = None,
    agent_factory: Callable[..., OllamaUniversityAgent] = OllamaUniversityAgent,
) -> tuple[OllamaUniversityAgent, str]:
    """Build one local agent using the shared deterministic model policy."""
    model = select_ollama_model(query, model_override)
    options: dict[str, int] = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    agent_arguments = {
        "client": client,
        "connector": connector,
        "materials_source": materials_source,
        "timezone_name": timezone_name,
        "model": model,
        **options,
    }
    if material_cache is not None:
        agent_arguments["material_cache"] = material_cache
    if timing_callback is not None:
        agent_arguments["timing_callback"] = timing_callback
    agent = agent_factory(
        **agent_arguments,
    )
    return agent, model


def write_ollama_timing(timing: OllamaRunTiming) -> None:
    """Write one privacy-safe local timing summary to stderr."""
    sys.stderr.write(
        "University-Agent timing: "
        f"model={timing.model} "
        f"total={timing.total_seconds:.3f}s "
        f"tools={timing.tool_seconds:.3f}s "
        f"gmail={timing.gmail_seconds:.3f}s "
        f"materials={timing.material_seconds:.3f}s "
        f"model_time={timing.model_seconds:.3f}s "
        f"model_rounds={timing.model_rounds} "
        f"response_chars={timing.response_chars}\n"
    )
