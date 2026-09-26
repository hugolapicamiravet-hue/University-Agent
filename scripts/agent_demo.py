"""Compatibility wrapper for the installed University Agent CLI."""

from university_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
