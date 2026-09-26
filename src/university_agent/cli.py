"""Run one University Agent query with host-controlled local configuration."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ollama import Client as OllamaClient
from openai import OpenAI, OpenAIError

from university_agent.agent_tools import is_explicit_material_query
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_materials import LocalMaterialsSource
from university_agent.ollama_agent import (
    OllamaUniversityAgent,
    OllamaUniversityAgentError,
    OllamaUniversityAgentProviderError,
    OllamaUniversityAgentRoundLimitError,
    OllamaUniversityAgentToolError,
)
from university_agent.openai_agent import (
    UniversityAgent,
    UniversityAgentError,
    UniversityAgentProviderError,
    UniversityAgentRoundLimitError,
    UniversityAgentToolError,
)


_OLLAMA_MATERIAL_MODEL = "llama3.2:3b"
_OLLAMA_DEFAULT_MODEL = "qwen3:14b"


def _select_ollama_model(query: str, override: str | None) -> str:
    """Select the local model while preserving an explicit host override."""
    if override is not None:
        return override
    if is_explicit_material_query(query):
        return _OLLAMA_MATERIAL_MODEL
    return _OLLAMA_DEFAULT_MODEL


def _status(message: str) -> None:
    """Write host status separately from the final answer."""
    sys.stderr.write(f"{message}\n")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask one question using the local-first University Agent.",
        epilog=(
            "Ollama is the default. Explicit personal-material queries use "
            "llama3.2:3b; other Ollama queries use qwen3:14b unless --model "
            "is supplied."
        ),
    )
    parser.add_argument(
        "--materials-root",
        required=True,
        type=Path,
        help="directory containing one subdirectory per course",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="exclude an exact directory path relative to each course",
    )
    parser.add_argument(
        "--timezone",
        required=True,
        help="IANA timezone used for trusted dates, for example Europe/Madrid",
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai"),
        default="ollama",
        help="inference provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        help="explicit model override; disables automatic Ollama routing",
    )
    parser.add_argument(
        "--num-predict",
        type=_positive_integer,
        help="optional Ollama output-token cap; unset by default",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="question; omit to enter it without shell-history exposure",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    arguments = parser.parse_args()

    query = " ".join(arguments.query).strip()
    if not query:
        query = input("Question: ").strip()
    if not query:
        parser.error("a non-empty query is required")

    try:
        materials_source = LocalMaterialsSource(
            arguments.materials_root,
            excluded_relative_paths=arguments.exclude,
        )
    except (FileNotFoundError, NotADirectoryError):
        _status("Materials root is missing or is not a usable directory.")
        return 2
    except ValueError as error:
        _status(f"Invalid materials configuration: {error}")
        return 2

    try:
        timezone = ZoneInfo(arguments.timezone)
    except ZoneInfoNotFoundError:
        _status("Unknown timezone. Use an IANA name such as Europe/Madrid.")
        return 2

    common = {
        "connector": GmailConnector(),
        "materials_source": materials_source,
        "timezone_name": arguments.timezone,
    }
    if arguments.provider == "ollama":
        model = _select_ollama_model(query, arguments.model)
        generation_options = {}
        if arguments.num_predict is not None:
            generation_options["num_predict"] = arguments.num_predict
        agent = OllamaUniversityAgent(
            client=OllamaClient(),
            model=model,
            **common,
            **generation_options,
        )
        _status(f"Using local Ollama model {model}.")
    else:
        model = arguments.model or "gpt-5.6-luna"
        try:
            client = OpenAI()
        except OpenAIError:
            _status(
                "OpenAI is not configured. Set OPENAI_API_KEY locally "
                "or use the default Ollama provider."
            )
            return 2
        agent = UniversityAgent(
            client=client,
            model=model,
            **common,
        )
        _status(f"Using OpenAI model {model}.")
    if is_explicit_material_query(query):
        _status("Searching local materials before generating the answer...")
    else:
        _status("Generating the answer...")
    now = datetime.now(timezone)

    try:
        answer = agent.run(query, now=now)
    except OllamaUniversityAgentProviderError:
        _status(
            f"Ollama could not use model {model}. Ensure Ollama is running "
            "and inspect installed models with `ollama list`."
        )
        return 1
    except UniversityAgentProviderError:
        _status(
            "OpenAI could not complete the request. Check local API-key, "
            "network, and model configuration."
        )
        return 1
    except (OllamaUniversityAgentToolError, UniversityAgentToolError):
        _status(
            "An academic data operation failed. Check the configured local "
            "materials and Gmail OAuth setup required by the question."
        )
        return 1
    except (OllamaUniversityAgentRoundLimitError, UniversityAgentRoundLimitError):
        _status(
            "The agent reached its tool-call limit. Try a more specific question."
        )
        return 1
    except (UniversityAgentError, OllamaUniversityAgentError):
        _status("The agent could not complete the request safely.")
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
