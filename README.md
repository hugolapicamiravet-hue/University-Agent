# University AI Agent

A minimal Python project foundation for a future personal university assistant.

The long-term goal is to integrate:

- UJI university Gmail, using OAuth authentication
- UJI Moodle Aula Virtual
- Personal university notes

## Current scope

This repository currently provides read-only Gmail access for a manual OAuth
test. It does not implement an AI agent, Moodle access, persistence, email
modification or sending, automatic classification, or a web interface.

## Requirements

- Python 3.12 or newer

## Local setup

Create and activate a virtual environment using Python 3.12 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

## Gmail OAuth setup

1. Enable the Gmail API in the Google Cloud project.
2. Configure the OAuth consent screen and create an OAuth client with the
   **Desktop app** application type.
3. Save the downloaded client file as `secrets/credentials.json`.
4. Run the manual demo from the repository root:

```bash
python scripts/gmail_demo.py
```

The first run opens a browser for authorization. The application requests only
the `https://www.googleapis.com/auth/gmail.readonly` scope. After authorization,
the token is stored at `secrets/token.json` and the 10 most recent messages are
printed with their sender, subject, and date.

The entire `secrets/` directory is ignored by Git. Never commit credentials,
OAuth client secrets, access tokens, refresh tokens, or other sensitive values.

## Structure

```text
.
├── pyproject.toml
├── README.md
├── scripts/
│   └── gmail_demo.py
└── src/
    └── university_agent/
        ├── __init__.py
        └── connectors/
            ├── __init__.py
            └── gmail.py
```
