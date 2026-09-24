# University Agent

[![CI](https://github.com/hugolapicamiravet-hue/University-Agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hugolapicamiravet-hue/University-Agent/actions/workflows/ci.yml)

## Overview

University Agent is a Python project for an AI-assisted university workflow.
The current implementation combines reliable read-only data access and
deterministic processing with one host-controlled OpenAI Responses API agent.

Today, the project reads Gmail metadata, parses supported academic email
subjects, detects upcoming deadlines, and retrieves recent notices explicitly
associated with a course code. It also discovers user-managed course directories
and material files from an explicitly configured local filesystem root, with
deterministic text extraction, chunking, and lexical retrieval for TXT and PDF
files. The agent can select three narrow academic operations and generate a
grounded response; it does not connect to Moodle.

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

### Local course materials

`LocalMaterialsSource` discovers course directories and material files beneath
an explicitly supplied filesystem root. The root is user-managed and must live
outside this repository. Immediate visible directories are treated as course
names, and visible, non-symlinked files are discovered recursively within an
exact course directory.

Discovery itself returns filesystem metadata only. Document contents remain
local and are read only when explicitly passed to the text-extraction operation.

### Material text extraction

`extract_text()` validates that a material still belongs to its supplied
`LocalMaterialsSource`, then extracts strict UTF-8 text from TXT files or
embedded text from PDF files. PDF page boundaries are preserved alongside the
combined text.

Scanned or image-only PDFs require OCR and are not supported. DOCX and PPTX are
also not supported. Extraction does not index, classify, summarize, or
semantically search document contents.

### Deterministic chunking and lexical search

`chunk_document()` splits extracted text at paragraph and whitespace boundaries
using a configurable character limit. PDF chunks preserve 1-based page
provenance and never combine text from different pages.

`search_chunks()` performs transparent Unicode-aware lexical matching. It ranks
chunks by the number of distinct query terms matched, then by total term
occurrences, with deterministic provenance-based tie breaking. Accents are
preserved; no stemming, translation, embeddings, or semantic matching occurs.

`search_local_materials()` composes discovery, extraction, chunking, and search
for one exact local course name or all discovered courses. Unsupported formats
are skipped in this batch workflow, while corrupt supported TXT or PDF files
still raise an extraction error.

### Agent-facing application operations

`academic_operations.py` exposes three narrow deterministic operations used by
the OpenAI runtime:

- `get_deadlines()` returns titles and due times within explicit `now` and
  `until` bounds.
- `get_course_notices()` returns notice text, explicit course codes, academic
  year, and the original Gmail Date header.
- `search_materials()` returns ranked local passages with a course name,
  relative document path, optional PDF page number, text, and lexical score.

The Gmail connector and local-material source are supplied by the application
host. Results intentionally omit Gmail message/thread identifiers, raw email
subjects, and absolute filesystem paths.

### OpenAI Responses agent

`UniversityAgent` implements a bounded OpenAI Responses API function-calling
loop. It exposes exactly three strict tools: remaining-week deadlines, recent
course notices, and lexical local-material search. The host injects Gmail and
local-material dependencies, trusted time, timezone, model, and resource caps.

The default model is `gpt-5.6-luna` and can be replaced at construction time.
Responses use `store=False`; tool calls are limited to four rounds by default.
Provider and internal operation failures cross the public boundary only as
sanitized host exceptions.

## Architecture

```text
                               User
                                 |
                                 v
                UniversityAgent / OpenAI Responses API
                    (strict tools, bounded loop)
                                 |
                  host time, timezone, limits
                                 |
                         academic_operations.py
                           /        |         \
                          /         |          \
                 deadlines       notices      materials
                     |               |             |
               Gmail workflows  Gmail workflow  local search workflow
                     |               |             |
                 Gmail API       Gmail API     user-managed files
                                                  |
                                      discover -> extract -> chunk
                                                  |
                                           lexical search
```

The connector owns communication with the external source, parsers perform
deterministic extraction from subject text, and workflows compose those pieces
into application operations. This keeps OAuth and Gmail access out of parsing
logic and allows application behavior to be tested without a live mailbox.
`LocalMaterialsSource` independently owns filesystem discovery and does not
depend on Gmail.

## Query capability matrix

| Example question | Status | Current boundary |
| --- | --- | --- |
| “¿Qué entregas tengo esta semana?” | Supported within limits | The host resolves the remaining ISO week; results include only deadlines detected from supported Gmail subjects and are not authoritative. |
| “¿Hay avisos recientes de EI0001?” | Partially supported | Requires an exact supported course code and academic year; the agent must clarify a missing year rather than guess. |
| “¿Dónde hablan mis apuntes de memoria caché?” | Supported within limits | Lexically searches supported local TXT and embedded-text PDF materials and returns relative source provenance. |
| “Explícame memoria caché usando mis apuntes.” | Supported within limits | The agent can generate an explanation grounded in lexically retrieved passages and should cite their relative provenance. |
| “¿Qué asignaturas tengo?” | Partially supported | User-managed local directory names can be listed; authoritative enrollment is unavailable. |
| “¿Qué ha cambiado hoy en Moodle?” | Not yet supported | Moodle is not integrated. |
| “¿He entregado esta práctica?” | Not yet supported | Gmail subject evidence is not an authoritative submission-state source. |

Natural-language interpretation, tool selection, and presentation belong to the
agent. Parsing, trusted time, bounds validation, filesystem safety, extraction,
chunking, and lexical ranking remain deterministic code.

## LLM integration boundary

The implemented runtime uses the official OpenAI Python SDK and Responses API
with three strict function schemas:

- `get_remaining_week_deadlines` accepts no model-controlled arguments.
- `get_recent_course_notices` accepts `course_code`, `academic_year`, and an
  optional bounded `lookback_days` value.
- `search_local_materials` accepts `query`, an optional exact `course_name`, and
  an optional bounded result `limit`.

The model may select an operation and supply semantic inputs such as an
exact course code/year, a material query, or an optional exact local course
name. The application host must inject and control:

- `GmailConnector` and `LocalMaterialsSource`
- credentials, local roots, and trusted current time
- timezone and calendar policy
- Gmail/material result limits and chunk-size limits
- validation, safe error translation, and tool execution

`resolve_remaining_week_window()` converts a trusted timezone-aware `now` and
an explicit IANA timezone name into timezone-naïve local bounds from that moment
through Sunday 23:59:59.999999. It uses an ISO Monday–Sunday week, never reads
the system clock, and produces bounds compatible with the current deadline
parser. The host—not Gmail or the model—chooses the timezone.

For a query such as “¿Qué tengo pendiente esta semana?”, the model selects
the deadline operation, the host resolves and validates the week bounds,
and `get_deadlines()` should return only narrowed deadline fields. The final
answer must describe these as deadlines detected from supported Gmail subject
formats, not as a complete authoritative task list.

For recent course notices, the model may provide an exact supported course code
and academic year. For material questions, it may provide search terms and an
optional exact local course name; it receives only ranked `MaterialPassage`
values, never arbitrary filesystem access.

Input-validation errors may be returned as concise correction guidance. Empty
results are normal. Authentication, provider, filesystem, and unexpected
internal errors must be translated to safe application errors; raw exception
details, credentials, paths, and connector state must not be sent to the model.

## Requirements

- Python 3.12 or newer

Runtime dependencies are declared in `pyproject.toml`: the Google libraries
required for Gmail OAuth/API access, `pypdf` for embedded PDF text extraction,
and the official `openai` Python SDK.

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

## OpenAI setup

Create an API key in the OpenAI dashboard and expose it only through the local
environment. The official SDK reads `OPENAI_API_KEY` automatically:

```bash
export OPENAI_API_KEY="replace-with-your-local-key"
```

Do not place the key in source code, the repository, command arguments, or
committed environment files. OpenAI API usage may incur provider charges.

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

### Local course materials

Keep the materials root outside the repository and organize it with one
user-managed directory per course:

```python
from pathlib import Path

from university_agent.local_materials import LocalMaterialsSource

source = LocalMaterialsSource(Path("/path/to/university-materials"))
courses = source.list_courses()
materials = source.list_materials("Curso Ficticio")
```

Discovery includes all visible regular files recursively, regardless of file
extension. Hidden entries and symbolic links are ignored. File contents are not
read.

To extract supported text explicitly:

```python
from university_agent.material_text import extract_text

material = materials[0]
document = extract_text(source, material)
```

TXT files are decoded strictly as UTF-8. PDF extraction preserves one text
string per page and does not perform OCR. The extracted result retains the
course name and relative material path without exposing an absolute path.

To search supported materials directly:

```python
from pathlib import Path

from university_agent.local_material_search import search_local_materials
from university_agent.local_materials import LocalMaterialsSource

source = LocalMaterialsSource(Path("/path/to/university-materials"))
results = search_local_materials(
    source,
    query="distributed systems",
    course_name="Fictional Course Alpha",
    limit=5,
)
```

Omit `course_name` to search every discovered course. Course matching is exact.
This is local lexical search, not semantic search or RAG.

### Agent-facing operations

Application hosts can compose the narrow result models without exposing broad
connector capabilities to the model:

```python
from datetime import datetime
from pathlib import Path

from university_agent.academic_operations import get_deadlines, search_materials
from university_agent.connectors.gmail import GmailConnector
from university_agent.local_materials import LocalMaterialsSource

deadlines = get_deadlines(
    GmailConnector(),
    now=datetime(2026, 9, 21, 9, 0),
    until=datetime(2026, 9, 28, 9, 0),
)

passages = search_materials(
    LocalMaterialsSource(Path("/path/to/university-materials")),
    query="cache coherence",
    course_name="Fictional Course Alpha",
)
```

The caller is responsible for injecting authenticated/local dependencies and
explicit time bounds. These deterministic functions return structured values
and do not themselves print, persist, summarize, or invoke a model.

### OpenAI agent demo

Run one natural-language query with an explicit user-managed materials root and
IANA timezone:

```bash
python scripts/agent_demo.py \
  --materials-root /path/to/university-materials \
  --timezone Europe/Madrid \
  "¿Dónde hablan mis apuntes de memoria caché?"
```

Omit the final query to enter it interactively, which avoids placing it in shell
history. The script prints only the final answer. Gmail authentication remains
lazy and is used only if the model selects a Gmail-backed operation.

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

The current suite contains 186 tests. Pure parsers are tested directly, while
Gmail-dependent behavior uses injected or mocked services and connectors. The
OpenAI adapter uses a fake injected client. Tests do not require live Gmail,
OpenAI access, OAuth credentials, API keys, tokens, or network access.

## Project structure

```text
.
├── .gitignore
├── README.md
├── pyproject.toml
├── scripts/
│   ├── agent_demo.py
│   └── gmail_demo.py
├── src/
│   └── university_agent/
│       ├── __init__.py
│       ├── academic_operations.py
│       ├── academic_notifications.py
│       ├── course_notices.py
│       ├── local_material_search.py
│       ├── local_materials.py
│       ├── material_chunks.py
│       ├── material_search.py
│       ├── material_text.py
│       ├── openai_agent.py
│       ├── recent_course_notices.py
│       ├── time_windows.py
│       ├── upcoming_deadlines.py
│       └── connectors/
│           ├── __init__.py
│           └── gmail.py
└── tests/
    ├── test_academic_operations.py
    ├── test_academic_notifications.py
    ├── test_course_notices.py
    ├── test_gmail.py
    ├── test_local_material_search.py
    ├── test_local_materials.py
    ├── test_material_chunks.py
    ├── test_material_search.py
    ├── test_material_text.py
    ├── test_openai_agent.py
    ├── test_recent_course_notices.py
    ├── test_time_windows.py
    └── test_upcoming_deadlines.py
```

## Security and privacy

- Gmail access uses only the read-only scope; the project cannot send or
  modify email.
- OAuth credentials and tokens remain in the local, Git-ignored `secrets/`
  directory.
- Mailbox metadata, snippets, diagnostic output, and demo output may contain
  private information and must not be committed or published.
- Local material contents remain private until selected passages are used by
  the agent. Those retrieved passages and their relative provenance are sent to
  OpenAI when needed for a generated answer; do not use the agent with material
  that must never leave the local machine.
- Retrieved document text is treated as untrusted data, not as instructions for
  the model or permission to change tool policy.
- The agent uses `store=False`, bounded strict tools, host-controlled limits,
  and sanitized public errors. Provider-side retention and data controls still
  depend on the configured OpenAI account and current provider policy.
- Credentials, client secrets, access tokens, and refresh tokens must never be
  included in source code, documentation, issues, or logs.

## Current limitations

- Gmail and OpenAI are the implemented network services; Moodle is not
  integrated.
- Material text extraction supports only UTF-8 TXT and embedded PDF text.
- Scanned or image-only PDFs, OCR, DOCX, and PPTX are not supported.
- Large or unusually complex PDFs may require substantial memory.
- The agent supports one provider only: OpenAI Responses API. No conversation
  persistence, provider abstraction, or authoritative course-name resolution
  is implemented.
- Search is lexical only: it does not infer synonyms or semantic similarity.
- No embeddings, vector database, semantic retrieval, or persistent RAG index
  is implemented. Generated explanations use only selected lexical passages.
- Message bodies, MIME parts, and attachments are not processed.
- Gmail result pagination is not implemented.
- Deterministic parsers support only observed subject formats.
- Course-notice text is not semantically classified.
- Parsed deadline timestamps currently contain no timezone information.

## Future roadmap

The following items are future work and are not currently implemented:

1. Evaluate the three natural-language flows with fictional data and a locally
   configured API key, then add regression cases for observed tool-selection
   failures.
2. Add DOCX or PPTX text extraction only if real local materials justify it.
3. Consider Moodle as an optional authoritative source only if an officially
   supported integration is authorized and verified.
4. Evaluate semantic retrieval as a complement to the existing lexical search
   only when real queries demonstrate the need.
5. Later consider state, automation, and a user-facing interface.
