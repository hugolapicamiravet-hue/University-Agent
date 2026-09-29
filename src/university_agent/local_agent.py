"""Shared host construction for the local Ollama University Agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from university_agent.agent_tools import is_explicit_material_query
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_materials import LocalMaterialsSource
from university_agent.ollama_agent import OllamaUniversityAgent


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
    agent_factory: Callable[..., OllamaUniversityAgent] = OllamaUniversityAgent,
) -> tuple[OllamaUniversityAgent, str]:
    """Build one local agent using the shared deterministic model policy."""
    model = select_ollama_model(query, model_override)
    options: dict[str, int] = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    agent = agent_factory(
        client=client,
        connector=connector,
        materials_source=materials_source,
        timezone_name=timezone_name,
        model=model,
        **options,
    )
    return agent, model
