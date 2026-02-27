# tests/ — Unit Tests

> **Maintenance**: Update this file when test patterns, file layout, or import paths change.

## Running

```bash
# Activate venv first
source .venv/Scripts/activate

# Run all tests
pytest tests/ -v

# Run a specific test class
pytest tests/test_generator.py::TestValidateCitationsGate1 -v
```

## Test Patterns

- **Pure unit tests only** — no external calls, no MongoDB, no Gemini API calls
- All network/API dependencies must be mocked
- Tests are fast and deterministic; they test logic, not integration

## File Layout

```
tests/
├── test_generator.py    — primacy-recency, temperatures, Gate 1 citation check
└── conftest.py          — shared pytest fixtures (create if needed)
```

## What NOT to Test Here

End-to-end pipeline evaluation lives in `evals/` (requires live MongoDB + API keys).
These tests cover only pure-Python logic that can run offline.

## Import Locations After Module Split

After `rag/` was split into focused modules, imports come from their new homes:

```python
from rag.context_builder import format_context_xml
from rag.gates import validate_citations_gate1, validate_citations_with_trace
from rag.prompts import _FUNCTION_TEMPERATURES, _FUNCTION_PROMPTS
from rag.generator import generate_stream, generate_comparison   # high-level API
```
