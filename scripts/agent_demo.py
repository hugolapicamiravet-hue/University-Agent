"""Run one University Agent query with host-controlled local configuration."""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ollama import Client as OllamaClient
from openai import OpenAI

from university_agent.agent_tools import is_explicit_material_query
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_materials import LocalMaterialsSource
from university_agent.ollama_agent import (
    OllamaUniversityAgent,
    OllamaUniversityAgentError,
)
from university_agent.openai_agent import UniversityAgent, UniversityAgentError


_OLLAMA_MATERIAL_MODEL = "llama3.2:3b"
_OLLAMA_DEFAULT_MODEL = "qwen3:14b"


def _select_ollama_model(query: str, override: str | None) -> str:
    """Select the local model while preserving an explicit host override."""
    if override is not None:
        return override
    if is_explicit_material_query(query):
        return _OLLAMA_MATERIAL_MODEL
    return _OLLAMA_DEFAULT_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one University Agent query")
    parser.add_argument("--materials-root", required=True, type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="exclude an exact directory path relative to each course",
    )
    parser.add_argument("--timezone", required=True)
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai"),
        default="ollama",
    )
    parser.add_argument("--model")
    parser.add_argument("query", nargs="*")
    arguments = parser.parse_args()

    query = " ".join(arguments.query).strip()
    if not query:
        query = input("Question: ").strip()
    if not query:
        parser.error("a non-empty query is required")

    common = {
        "connector": GmailConnector(),
        "materials_source": LocalMaterialsSource(
            arguments.materials_root,
            excluded_relative_paths=arguments.exclude,
        ),
        "timezone_name": arguments.timezone,
    }
    if arguments.provider == "ollama":
        agent = OllamaUniversityAgent(
            client=OllamaClient(),
            model=_select_ollama_model(query, arguments.model),
            **common,
        )
    else:
        agent = UniversityAgent(
            client=OpenAI(),
            model=arguments.model or "gpt-5.6-luna",
            **common,
        )
    now = datetime.now(ZoneInfo(arguments.timezone))

    try:
        answer = agent.run(query, now=now)
    except (UniversityAgentError, OllamaUniversityAgentError):
        print("The agent could not complete the request safely.")
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
