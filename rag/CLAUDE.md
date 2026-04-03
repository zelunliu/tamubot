# rag/ — RAG Pipeline

## Public API

```python
from rag import run_pipeline, run_pipeline_with_memory, get_current_state
from rag import ChunkDoc, CourseDoc, PolicyDoc, VALID_CATEGORIES
from rag import RouterResult, PipelineState, ConversationState
from rag import get_langfuse, run_ragas_background, compute_ragas_metrics
```

## Module Locations

- LLM client: `rag/tools/llm.py` — `call_llm()`, `stream_llm()`
- Observability: `rag/tools/langfuse.py` — `get_langfuse()`, RAGAS
- MongoDB search: `rag/tools/mongo.py` — `hybrid_search()`, `semantic_search()`, etc.
- Voyage AI: `rag/tools/voyage.py` — `embed_query()`, `rerank()`
- Schedule logic: `rag/tools/schedule.py` — `parse_meeting_times()`, `filter_conflicting_courses()`
- mem0: `rag/tools/mem0.py` — `Mem0Manager`, `get_mem0_manager()`
- State contracts: `rag/state/pipeline_state.py` — `PipelineState`, `RouterResult`, `ConversationState`
- Graph entry: `rag/graph/pipeline.py` — `run_pipeline()`, `run_pipeline_with_memory()`
- Graph builder: `rag/graph/builder.py` — `build_graph()`, `build_graph_with_memory()`
- Middleware: `rag/graph/middleware.py` — `@tracing_middleware`, `@timing_middleware`, `@error_guard_middleware`
- Nodes: `rag/nodes/*.py` — one file per node, no DI
- Edges: `rag/edges/routing.py` — `route_after_router()`, `route_after_retrieval()`

## LLM Client

All LLM calls go through `rag/tools/llm.py` — do NOT call `config.get_tamu_client()` / `config.get_genai_client()` directly in nodes:

```python
result = call_llm(messages, temperature=0, max_tokens=4096, json_mode=True, thinking_budget=512)
for token in stream_llm(messages, temperature=0.2, max_tokens=4096, thinking_budget=1024):
```

- `result.input_tokens / output_tokens` — None on TAMU path (SSE, no token counts exposed)
- Langfuse model name: `config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL`


## Gotchas

- **TAMU gateway `max_tokens`**: min 4096 on ALL `call_llm()` calls — smaller values return empty response
- **`generate_eval_search_string`**: uses `max_tokens=4096` (not 256) for TAMU compat
- **Gemini JSON mode**: free-form Markdown fields silently return empty → always render Markdown in Python (`_render_comparison_markdown()`)
- **Primacy-recency** (`format_context_xml`): rank 1 → context start, rank 2 → context end, ranks 3–N → middle
- **Gate 1** (sync, regex): `validate_citations_with_trace()` — checks `[Source N]` presence after generation
- **Gate 2** (async, LLM): `run_groundedness_scoring_background()` — RAGAS groundedness in background thread, score → Langfuse
