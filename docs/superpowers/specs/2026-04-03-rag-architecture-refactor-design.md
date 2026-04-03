# Design Spec: rag/ Architecture Refactor — State / Nodes / Edges / Graph / Tools

**Date:** 2026-04-03  
**Status:** Approved

---

## Problem

The `rag/` folder has two parallel representations of the same logic:

1. **Legacy top-level modules** (`search.py`, `generator.py`, `router.py`, etc.) — original pipeline code
2. **`v4/` sub-package** — LangGraph pipeline that wraps the legacy modules via a DI layer (`ComponentRegistry`, `interfaces.py`, `registry_factory.py`, `components/`, `providers/`)

This results in:
- ~200 lines of duplicated search logic (`search.py` vs `search_v3.py`)
- Two files for every concept (e.g., `reranker.py` + `v4/components/rerankers.py`)
- Implicit data contracts (`list[dict]` with undocumented keys throughout)
- A DI layer that adds indirection without enabling any actual swapping in practice

---

## Design Principle

**Contract-based deep module development:**
- **Deep modules**: each module has a simple interface (few exported functions/types) hiding complex internal implementation
- **Contracts**: the state TypedDicts are the explicit data contracts — every field that flows between nodes is typed and documented
- **No service/adapter layer**: nodes call tools directly; tools wrap external APIs; state flows between nodes

---

## Architecture

### Mental Model

```
State (TypedDicts)   ← data contracts: what flows between nodes
Nodes                ← deep modules: (state) → state updates
Edges                ← routing contracts: state field → next node
Graph                ← LangGraph builder + pipeline entry point
Tools                ← external API wrappers (LLM, MongoDB, Voyage, Langfuse, mem0)
```

### Target Folder Structure

```
rag/
├── state/
│   ├── __init__.py
│   └── pipeline_state.py        ← PipelineState, ConversationState, ConversationMessage
├── nodes/                        ← one file per graph node
│   ├── router_node.py
│   ├── retrieval_node.py
│   ├── anchor_node.py
│   ├── eval_search_node.py
│   ├── generator_node.py
│   ├── history_inject_node.py
│   ├── history_update_node.py
│   ├── merge_node.py
│   ├── out_of_scope_node.py
│   └── schedule_filter_node.py
├── edges/
│   └── routing.py               ← _route_after_router, _route_after_retrieval
├── graph/
│   ├── builder.py               ← build_graph(), build_graph_with_memory()
│   ├── pipeline.py              ← run_pipeline(), run_pipeline_with_memory(), get_current_state()
│   ├── checkpointer.py
│   ├── session.py
│   └── trace_registry.py
├── tools/
│   ├── llm.py                   ← TAMU/Gemini client (from llm_client.py)
│   ├── mongo.py                 ← MongoDB search + fetch (from search_v3.py + components/)
│   ├── voyage.py                ← embedding + reranking (from search_v3.py + reranker.py + components/)
│   ├── langfuse.py              ← Langfuse REST client + RAGAS (from observability.py)
│   ├── context.py               ← context formatting (from context_builder.py)
│   ├── schedule.py              ← meeting time parsing (from schedule.py)
│   └── mem0.py                  ← Mem0Manager + registry (from v4/mem0_*.py)
├── models.py                    ← Pydantic data models (keep at root, shared with ingestion_pipeline)
├── prompts.py                   ← prompt strings (pure, no pipeline coupling)
├── comparison_schemas.py        ← Pydantic comparison schemas
└── __init__.py                  ← thin: run_pipeline, observability, data models only
```

---

## Contracts

### Data contracts (state TypedDicts)

`state/pipeline_state.py` defines all data that flows through the graph. Fields that were previously typed as `Any` are tightened:

- `router_result: RouterResult` — `RouterResult` dataclass moves to `state/pipeline_state.py` (since it appears in PipelineState, placing it there avoids circular imports with nodes/)
- `trace: LFTrace | None` — typed with `TYPE_CHECKING` guard
- All chunk lists typed as `list[dict]` (acceptable; full TypedDict schema for chunks is a future step)

### Behavioral contracts (node signature)

Every node has the same contract:

```python
def some_node(state: PipelineState) -> dict:
    """One-line description of what this node does."""
    ...
    return {"field_name": value}
```

No `registry` parameter. No `functools.partial` binding. Nodes import tools directly.

### Routing contracts (edges)

Edge functions read one state field and return a string node name:

```python
def route_after_router(state: PipelineState) -> str:
    function = state.get("function", "out_of_scope")
    if function == "out_of_scope": return "out_of_scope"
    elif function == "recurrent": return "anchor"
    else: return "retrieval"
```

---

## What Gets Removed

**Deleted entirely (not moved):**
- `rag/search.py` — v1 legacy collections, not used by v4
- `rag/v4/interfaces.py` — Protocol/DI layer
- `rag/v4/registry_factory.py` — no registry needed
- `rag/v4/components/` (6 files) — logic split into tools/
- `rag/v4/providers/v3_adapters.py` — no adapters needed
- `rag/v4/middleware.py` — timing absorbed into nodes

**Merged/moved:**
- `search_v3.py` → `tools/mongo.py` + `tools/voyage.py`
- `generator.py` → `nodes/generator_node.py` + `nodes/eval_search_node.py`
- `router.py` → `nodes/router_node.py`
- `reranker.py` → `tools/voyage.py`
- `gates.py` → `nodes/generator_node.py`
- `models_v3.py` → merged into `models.py`
- `llm_client.py` → `tools/llm.py`
- `observability.py` → `tools/langfuse.py`
- `context_builder.py` → `tools/context.py`
- `schedule.py` → `tools/schedule.py`
- `v4/nodes/*.py` → `nodes/` (with DI removed)
- `v4/graph.py` → `graph/builder.py` + `edges/routing.py`
- `v4/pipeline_v4.py` → `graph/pipeline.py`

---

## Public API (rag/__init__.py after)

```python
from rag.graph.pipeline import run_pipeline, run_pipeline_with_memory, get_current_state
from rag.tools.langfuse import get_langfuse, run_ragas_background, compute_ragas_metrics
from rag.models import ChunkDoc, CourseDoc, PolicyDoc, VALID_CATEGORIES
from rag.state.pipeline_state import PipelineState, ConversationState
```

Removed from public API: `classify_query`, `generate`, `generate_stream`, `hybrid_search`, `search_semantic`, `rerank`, `router_order`, `generator_order`, `route_retrieve_rerank`, etc. — these are internal pipeline details, not a public API.

---

## Verification

```bash
make typecheck    # mypy passes on new structure
make lint         # ruff passes
make test         # existing test suite passes
make probe        # end-to-end RAG query returns correct answer
```
