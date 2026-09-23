# University Agent

[![CI](https://github.com/hugolapicamiravet-hue/University-Agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hugolapicamiravet-hue/University-Agent/actions/workflows/ci.yml)

## Overview

University Agent is a Python project for building an AI-assisted university
workflow. The current implementation deliberately establishes reliable,
read-only data access and deterministic processing before adding an AI layer.

Today, the project reads Gmail metadata, parses supported academic email
subjects, detects upcoming deadlines, and retrieves recent notices explicitly
associated with a course code. It does not currently implement an AI agent or
connect to Moodle.

## Implemented functionality

### Read-only Gmail integration

`GmailConnector` authenticates through OAuth 2.0 and requests only the
`gmail.readonly` scope. It provides metadata-oriented access without sending or
modifying email:

- `search_messages()` executes Gmail-native search queries and returns message
  metadata.
- `get_message()` retrieves one message's metadata by Gmail message ID.
- `fetch_recent_messages()` returns the sender, subject, and date of up to ten
  recent messages.

Returned metadata includes message and thread identifiers, selected headers,
and Gmail's snippet preview. Decoded message bodies and attachments are not
retrieved or parsed.

### Academic notification parsing

`academic_notifications.py` deterministically parses observed academic
notification subject formats. It recognizes supported deadline, overdue-task,
activity-opening, submission-confirmation, task-summary, and account-login
templates while safely classifying unsupported subjects as unknown.

Where a supported subject contains a valid date and time, the parser produces a
normalized `datetime`. Invalid or unsupported date text does not interrupt
batch processing.

### Course-prefixed notice parsing

`course_notices.py` parses supported subjects whose prefix explicitly contains
one or more `EI####` or `MT####` course codes and a consecutive `YYYY-YYYY`
academic year. It extracts the codes, academic year, and notice text without
inferring a course name or interpreting the meaning of the notice.

### Upcoming deadlines

`find_upcoming_deadlines()` searches Gmail for candidate deadline messages,
parses their subjects, and returns only recognized deadlines whose parsed time
is strictly later than the caller-supplied `now`. Results retain Gmail message
and thread identifiers and are ordered chronologically.

### Recent course notices

`find_recent_course_notices()` searches a configurable recent Gmail window and
returns parsed notices containing an exact requested course code and academic
year. It preserves Gmail provenance and the original Date header. Valid Date
headers are ordered newest first; missing or unusable dates are placed last.

## Architecture

```text
Gmail API
    |
    v
GmailConnector
    |
    +-----------------------------+
    |                             |
    v                             v
academic_notifications.py    course_notices.py
    |                             |
    v                             v
upcoming_deadlines.py        recent_course_notices.py
```

The connector owns communication with the external source, parsers perform
deterministic extraction from subject text, and workflows compose those pieces
into application operations. This keeps OAuth and Gmail access out of parsing
logic and allows application behavior to be tested without a live mailbox.

## Requirements

- Python 3.12 or newer

Runtime dependencies are declared in `pyproject.toml` and are limited to the
Google libraries required for Gmail OAuth and API access.

## Installation

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

1. Enable the Gmail API in a Google Cloud project.
2. Configure the OAuth consent screen and create an OAuth client with the
   **Desktop app** application type.
3. Save the downloaded client file locally as `secrets/credentials.json`.
4. Run the manual demo from the repository root:

```bash
python scripts/gmail_demo.py
```

The first run opens a browser for authorization. The application requests only
the `https://www.googleapis.com/auth/gmail.readonly` scope. The resulting token
remains local at `secrets/token.json`. The entire `secrets/` directory is
Git-ignored; credentials and tokens must never be committed.

## Usage

### Gmail search

```python
from university_agent.connectors.gmail import GmailConnector

connector = GmailConnector()
messages = connector.search_messages("is:unread newer_than:7d", max_results=10)
```

### Upcoming deadlines

```python
from datetime import datetime

from university_agent.connectors.gmail import GmailConnector
from university_agent.upcoming_deadlines import find_upcoming_deadlines

deadlines = find_upcoming_deadlines(
    GmailConnector(),
    now=datetime(2026, 9, 23, 12, 0),
    max_results=100,
)
```

`now` must be timezone-naive because parsed notification timestamps currently
contain no timezone information.

### Recent course notices

```python
from university_agent.connectors.gmail import GmailConnector
from university_agent.recent_course_notices import find_recent_course_notices

notices = find_recent_course_notices(
    GmailConnector(),
    course_code="EI0001",
    academic_year="2026-2027",
)
```

The course code above is illustrative. The operation performs exact code and
academic-year matching; it does not resolve course names.

### Manual OAuth demo

Run:

```bash
python scripts/gmail_demo.py
```

The demo authenticates with `GmailConnector` and prints sender, subject, and
date metadata for up to ten recent messages. This is real mailbox data: keep
the terminal output private and do not publish or commit it.

## Testing

Run the isolated unit-test suite from the repository root after installation:

```bash
python -m unittest discover -s tests
```

The current suite contains 69 tests. Pure parsers are tested directly, while
Gmail-dependent behavior uses injected or mocked services and connectors. The
tests do not require live Gmail access, OAuth credentials, tokens, or network
access.

## Project structure

```text
.
├── .gitignore
├── README.md
├── pyproject.toml
├── scripts/
│   └── gmail_demo.py
├── src/
│   └── university_agent/
│       ├── __init__.py
│       ├── academic_notifications.py
│       ├── course_notices.py
│       ├── recent_course_notices.py
│       ├── upcoming_deadlines.py
│       └── connectors/
│           ├── __init__.py
│           └── gmail.py
└── tests/
    ├── test_academic_notifications.py
    ├── test_course_notices.py
    ├── test_gmail.py
    ├── test_recent_course_notices.py
    └── test_upcoming_deadlines.py
```

## Security and privacy

- Gmail access uses only the read-only scope; the project cannot send or
  modify email.
- OAuth credentials and tokens remain in the local, Git-ignored `secrets/`
  directory.
- Mailbox metadata, snippets, diagnostic output, and demo output may contain
  private information and must not be committed or published.
- Credentials, client secrets, access tokens, and refresh tokens must never be
  included in source code, documentation, issues, or logs.

## Current limitations

- Gmail is the only implemented external data source; Moodle is not integrated.
- No AI agent, course-name resolution, database, or persistence is implemented.
- Message bodies, MIME parts, and attachments are not processed.
- Gmail result pagination is not implemented.
- Deterministic parsers support only observed subject formats.
- Course-notice text is not semantically classified.
- Parsed deadline timestamps currently contain no timezone information.

## Future roadmap

The following items are future work and are not currently implemented:

1. Obtain institutional authorization for supported Moodle access.
2. Add a read-only Moodle connector using the officially supported
   authentication and API mechanism.
3. Use Moodle as the authoritative source for enrolled courses, course names,
   sections, deadlines, announcements, and materials where the authorized API
   actually exposes them.
4. Add document processing and search for course materials when supported by
   the Moodle integration.
5. Compose higher-level deterministic academic operations.
6. Add an AI layer that selects and combines those tested operations.
7. Later consider state, automation, and a user-facing interface.
