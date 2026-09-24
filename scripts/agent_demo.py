"""Run one University Agent query with host-controlled local configuration."""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from openai import OpenAI

from university_agent.connectors.gmail import GmailConnector
from university_agent.local_materials import LocalMaterialsSource
from university_agent.openai_agent import UniversityAgent, UniversityAgentError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one University Agent query")
    parser.add_argument("--materials-root", required=True, type=Path)
    parser.add_argument("--timezone", required=True)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("query", nargs="*")
    arguments = parser.parse_args()

    query = " ".join(arguments.query).strip()
    if not query:
        query = input("Question: ").strip()
    if not query:
        parser.error("a non-empty query is required")

    agent = UniversityAgent(
        client=OpenAI(),
        connector=GmailConnector(),
        materials_source=LocalMaterialsSource(arguments.materials_root),
        timezone_name=arguments.timezone,
        model=arguments.model,
    )
    now = datetime.now(ZoneInfo(arguments.timezone))

    try:
        answer = agent.run(query, now=now)
    except UniversityAgentError:
        print("The agent could not complete the request safely.")
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
