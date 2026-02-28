# tests/

Pure unit tests only — no external calls, no MongoDB, no API keys. End-to-end evals live in `evals/`.

## Import Paths (after module split)

```python
from rag.context_builder import format_context_xml
from rag.gates import validate_citations_gate1, validate_citations_with_trace
from rag.prompts import _FUNCTION_TEMPERATURES, _FUNCTION_PROMPTS
from rag.generator import generate_stream, generate_comparison
```
