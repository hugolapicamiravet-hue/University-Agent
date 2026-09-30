# Academic knowledge core

University-Agent exposes a small document-focused API intended to become a
reusable academic knowledge engine. The public package boundary is:

```python
from university_agent import AcademicKnowledgeEngine

engine = AcademicKnowledgeEngine("/path/to/materials")
passages = engine.search(
    "exclusión mutua",
    course="Operating Systems",
    limit=5,
)
```

`AcademicKnowledgeEngine` owns the existing local workflow:

```text
document discovery -> text extraction -> chunking -> lexical retrieval
                   -> evidence with relative provenance
```

It can list course names and relative material references, and its search
results contain the course, relative source path, PDF page when available,
passage text, and the established lexical score. Absolute filesystem paths are
not part of this public result boundary.

The engine deliberately preserves the current TXT/PDF extraction, chunking,
ranking, limits, error behavior, and process-local extraction cache. It does
not implement a persistent index, semantic search, embeddings, new formats,
stable passage identifiers, or claim-level citation alignment.

## Application adapters

Gmail workflows, the Ollama and OpenAI tool loops, the CLI, and the local GUI
remain implemented application adapters. They are not exported by the package's
document-core API. Existing agent material tools delegate to the knowledge
engine while Gmail deadlines and notices retain their current behavior.

## Future Hugo OS boundary

Hugo OS is expected to own conversation, Gmail, structured tasks and exams,
calendar behavior, authentication, and general model orchestration. No Hugo OS
integration exists in this phase. A future integration can consume the engine's
document evidence without constructing a Gmail connector or an LLM client.
