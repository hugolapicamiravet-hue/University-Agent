# University AI Agent

A minimal Python project foundation for a future personal university assistant.

The long-term goal is to integrate:

- UJI university Gmail, using OAuth authentication
- UJI Moodle Aula Virtual
- Personal university notes

## Current scope

This repository currently provides read-only Gmail search and message metadata
retrieval, with a manual OAuth demo. It does not implement an AI agent,
Moodle access, persistence, email modification or sending, automatic
classification, or a web interface.

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

## Gmail connector

`GmailConnector.search_messages(query, max_results=10)` forwards Gmail's native
search syntax unchanged and returns one page of message metadata. For example,
`connector.search_messages("from:fixture2@example.com", max_results=5)`.
`max_results` must be between 1 and 500; pagination is not implemented.

`GmailConnector.get_message(message_id)` retrieves metadata for one nonempty
Gmail message ID. Both methods return dictionaries with `id`, `thread_id`,
`sender`, `recipients`, `subject`, `date`, and `snippet` (search returns a list).
Headers remain unparsed strings; missing headers and snippets become empty
strings. Bodies and attachments are not retrieved or parsed.

`fetch_recent_messages()` preserves its original list of sender, subject, and
date dictionaries for up to 10 messages. API errors propagate to the caller.
An optional `service=` constructor argument allows an injected Gmail service.

## Unit tests

Run the complete isolated suite from the repository root after installation:

```bash
python -m unittest discover -s tests -v
```

Tests use mocked services and require no Gmail account or OAuth credentials.

## Structure

```text
.
├── pyproject.toml
├── README.md
├── scripts/
│   └── gmail_demo.py
├── tests/
│   └── test_gmail.py
└── src/
    └── university_agent/
        ├── __init__.py
        └── connectors/
            ├── __init__.py
            └── gmail.py
```
