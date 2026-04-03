# rag/ Architecture Refactor — State / Nodes / Edges / Graph / Tools

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flatten `rag/` into a single clean structure — `state/`, `nodes/`, `edges/`, `graph/`, `tools/` — deleting the `v4/` DI layer (ComponentRegistry, interfaces.py, components/, providers/) and all legacy top-level modules (search.py, generator.py, router.py, etc.) that v4 previously wrapped.

**Architecture:** State TypedDicts are the data contracts between nodes. Nodes call tools directly (no registry injection). Tools are thin wrappers over external APIs (MongoDB, Voyage AI, LLM, Langfuse, mem0). The LangGraph graph structure (builder, edges, pipeline entry) lives in `graph/`.

**Tech Stack:** Python 3.11+, LangGraph, MongoDB Atlas, Voyage AI, TAMU AI Gateway (OpenAI-compatible SSE), Langfuse REST, mem0, pytest

**Spec:** `docs/superpowers/specs/2026-04-03-rag-architecture-refactor-design.md`

---

## File Map

**Created:**
- `rag/state/__init__.py`, `rag/state/pipeline_state.py`
- `rag/nodes/__init__.py` + 10 node files
- `rag/edges/__init__.py`, `rag/edges/routing.py`
- `rag/graph/__init__.py`, `rag/graph/builder.py`, `rag/graph/pipeline.py`
- `rag/graph/middleware.py`, `rag/graph/trace_registry.py`, `rag/graph/cache_utils.py`
- `rag/graph/checkpointer.py`, `rag/graph/session.py`, `rag/graph/exceptions.py`
- `rag/tools/__init__.py`, `rag/tools/llm.py`, `rag/tools/langfuse.py`
- `rag/tools/context.py`, `rag/tools/schedule.py`
- `rag/tools/voyage.py`, `rag/tools/mongo.py`, `rag/tools/mem0.py`

**Modified:**
- `rag/__init__.py` (shrinks to graph.pipeline + tools.langfuse + models)
- `app.py` (update imports from rag.v4.* → rag.graph.* / rag.tools.*)
- `tests/test_v4_middleware.py`, `tests/test_v4_router_node.py`, `tests/test_v4_graph.py`,
  `tests/test_v4_state.py`, `tests/test_v4_session.py`, `tests/test_v4_history_node.py`,
  `tests/test_v4_recurrent_path.py`, `tests/test_v4_routing_matrix.py`,
  `tests/test_v4_pipeline_timing.py`, `tests/test_v4_mem0_cache.py`,
  `tests/test_v4_observability.py`, `tests/test_v4_integration.py`,
  `tests/test_v4_components.py`, `tests/test_v3_v4_parity.py`, `tests/test_generator.py`
- `tests/CLAUDE.md` (update import paths)

**Deleted:**
- `rag/search.py`, `rag/search_v3.py`, `rag/generator.py`, `rag/router.py`
- `rag/reranker.py`, `rag/gates.py`, `rag/models_v3.py`
- `rag/llm_client.py`, `rag/observability.py`, `rag/context_builder.py`, `rag/schedule.py`
- `rag/v4/` (entire directory — 20+ files)
- `tests/test_v4_interfaces.py`, `tests/test_v4_swappability.py` (test DI layer being removed)

---

## Task 1: Create Directory Structure

**Files:**
- Create: `rag/state/__init__.py`, `rag/nodes/__init__.py`, `rag/edges/__init__.py`
- Create: `rag/graph/__init__.py`, `rag/tools/__init__.py`

- [ ] **Step 1: Create all package `__init__.py` files**

```bash
mkdir -p rag/state rag/nodes rag/edges rag/graph rag/tools
touch rag/state/__init__.py rag/nodes/__init__.py rag/edges/__init__.py
touch rag/graph/__init__.py rag/tools/__init__.py
```

- [ ] **Step 2: Verify structure**

```bash
find rag/state rag/nodes rag/edges rag/graph rag/tools -name "__init__.py"
```
Expected: 5 files listed

- [ ] **Step 3: Commit**

```bash
git add rag/state rag/nodes rag/edges rag/graph rag/tools
git commit -m "chore: scaffold state/nodes/edges/graph/tools directories"
```

---

## Task 2: Migrate tools/llm.py

Move `rag/llm_client.py` → `rag/tools/llm.py`. Only the module path changes; internal logic is identical.

**Files:**
- Create: `rag/tools/llm.py`
- Keep old: `rag/llm_client.py` (delete in Task 18 only after all imports updated)

- [ ] **Step 1: Copy file to new location**

```bash
cp rag/llm_client.py rag/tools/llm.py
```

- [ ] **Step 2: Update module docstring in tools/llm.py**

In `rag/tools/llm.py`, change the first line:
```python
# Before (top of file, line 1):
"""Unified LLM client — routes calls to TAMU AI gateway or Gemini based on config."""

# After:
"""Unified LLM client — routes calls to TAMU AI gateway or Gemini based on config.

Canonical location: rag/tools/llm.py
"""
```

- [ ] **Step 3: Add backward-compat re-export to old location**

Overwrite `rag/llm_client.py` with a thin re-export (keeps old importers working until Task 18):
```python
"""Backward-compat shim — import from rag.tools.llm instead."""
from rag.tools.llm import *  # noqa: F401, F403
from rag.tools.llm import call_llm, stream_llm, LLMResult, _count_tokens_approx, _count_messages_tokens
```

- [ ] **Step 4: Run existing tests to verify nothing broken**

```bash
make test
```
Expected: same pass/fail count as before (no new failures from llm_client changes)

- [ ] **Step 5: Commit**

```bash
git add rag/tools/llm.py rag/llm_client.py
git commit -m "refactor(tools): migrate llm_client.py → tools/llm.py with shim"
```

---

## Task 3: Migrate tools/langfuse.py

Move `rag/observability.py` → `rag/tools/langfuse.py`.

**Files:**
- Create: `rag/tools/langfuse.py`
- Update: `rag/observability.py` (shim)

- [ ] **Step 1: Copy and update**

```bash
cp rag/observability.py rag/tools/langfuse.py
```

Update top docstring in `rag/tools/langfuse.py`:
```python
"""Langfuse REST client + RAGAS scoring.

Canonical location: rag/tools/langfuse.py
"""
```

- [ ] **Step 2: Create shim at old location**

```python
# rag/observability.py
"""Backward-compat shim — import from rag.tools.langfuse instead."""
from rag.tools.langfuse import *  # noqa: F401, F403
from rag.tools.langfuse import (
    get_langfuse, compute_ragas_metrics, run_ragas_background,
    MinimalLangfuseClient, LFGeneration, LFSpan,
)
```

- [ ] **Step 3: Run tests**

```bash
make test
```

- [ ] **Step 4: Commit**

```bash
git add rag/tools/langfuse.py rag/observability.py
git commit -m "refactor(tools): migrate observability.py → tools/langfuse.py with shim"
```

---

## Task 4: Migrate tools/context.py and tools/schedule.py

Two small pure-utility files, no external dependencies.

**Files:**
- Create: `rag/tools/context.py`, `rag/tools/schedule.py`
- Update: `rag/context_builder.py`, `rag/schedule.py` (shims)

- [ ] **Step 1: Copy both files**

```bash
cp rag/context_builder.py rag/tools/context.py
cp rag/schedule.py rag/tools/schedule.py
```

- [ ] **Step 2: Replace rag/context_builder.py with shim**

```python
# rag/context_builder.py
"""Backward-compat shim — import from rag.tools.context instead."""
from rag.tools.context import *  # noqa: F401, F403
from rag.tools.context import format_context_xml, collapse_whitespace, strip_thinking_blocks
```

- [ ] **Step 3: Replace rag/schedule.py with shim**

```python
# rag/schedule.py
"""Backward-compat shim — import from rag.tools.schedule instead."""
from rag.tools.schedule import *  # noqa: F401, F403
from rag.tools.schedule import (
    MeetingInterval, parse_meeting_times, schedules_conflict, filter_conflicting_courses
)
```

- [ ] **Step 4: Run tests**

```bash
make test
```

- [ ] **Step 5: Commit**

```bash
git add rag/tools/context.py rag/tools/schedule.py rag/context_builder.py rag/schedule.py
git commit -m "refactor(tools): migrate context_builder + schedule → tools/"
```

---

## Task 5: Create tools/voyage.py

Merge embedding + reranking from `rag/search_v3.py`, `rag/reranker.py`, `rag/v4/components/embedders.py`, `rag/v4/components/rerankers.py` into one clean tool.

**Files:**
- Create: `rag/tools/voyage.py`

- [ ] **Step 1: Write rag/tools/voyage.py**

```python
"""Voyage AI tool — embedding and reranking.

Exposes:
  embed_query(text) -> list[float]
  rerank(query, chunks, top_k) -> list[dict]
  stratified_select(chunks, k) -> list[dict]
"""
from __future__ import annotations

from typing import Optional

import voyageai

import config

EMBEDDING_MODEL = "voyage-3"
RERANK_MODEL = "rerank-2"

_voyage: Optional[voyageai.Client] = None


def _get_client() -> voyageai.Client:
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _voyage


def embed_query(text: str) -> list[float]:
    """Embed a query string using Voyage AI voyage-3."""
    client = _get_client()
    result = client.embed([text], model=EMBEDDING_MODEL, input_type="query")
    return result.embeddings[0]


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Cross-encoder rerank chunks by relevance to query, return top_k.

    Preserves all original chunk fields. Adds/updates 'score' field with
    rerank score. Returns chunks sorted descending by score.
    Falls back to original order on any Voyage error.
    """
    if not chunks:
        return []
    top_k = min(top_k, len(chunks))

    from rag.graph.trace_registry import child_span
    client = _get_client()

    texts = [c.get("content", "") for c in chunks]
    with child_span("rerank.voyage", {"query": query[:100], "n_chunks": len(chunks)}) as ctx:
        response = client.rerank(query=query, documents=texts, model=RERANK_MODEL, top_k=top_k)
        results = []
        for item in response.results:
            chunk = dict(chunks[item.index])
            chunk["score"] = item.relevance_score
            results.append(chunk)
        ctx["output"] = {"n_reranked": len(results)}

    return results


def stratified_select(chunks: list[dict], k: int) -> list[dict]:
    """Select up to k chunks with at most ceil(k/n_courses) per course.

    Used as a fallback when Voyage reranking is unavailable.
    """
    if not chunks or k <= 0:
        return []
    from collections import defaultdict
    import math
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        buckets[c.get("course_id", "_")].append(c)
    n = len(buckets)
    per_course = math.ceil(k / n) if n else k
    selected = []
    for course_chunks in buckets.values():
        selected.extend(course_chunks[:per_course])
    return selected[:k]
```

- [ ] **Step 2: Verify it's importable**

```bash
python -c "from rag.tools.voyage import embed_query, rerank, stratified_select; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Create shim in reranker.py**

```python
# rag/reranker.py
"""Backward-compat shim — import from rag.tools.voyage instead."""
from rag.tools.voyage import rerank, stratified_select

def rerank_multi_course(query, chunks, top_k):
    """Multi-course rerank — delegates to rerank()."""
    return rerank(query, chunks, top_k)
```

- [ ] **Step 4: Run tests**

```bash
make test
```

- [ ] **Step 5: Commit**

```bash
git add rag/tools/voyage.py rag/reranker.py
git commit -m "refactor(tools): create tools/voyage.py from reranker + search_v3 embedding"
```

---

## Task 6: Create tools/mongo.py

Merge `rag/search_v3.py` (all search/fetch functions) and `rag/v4/components/document_stores.py` (MongoDocumentStore). The unified tool exposes clean functions; no class wrapper needed.

**Files:**
- Create: `rag/tools/mongo.py`

- [ ] **Step 1: Write rag/tools/mongo.py**

```python
"""MongoDB tool — all search and fetch operations against chunks_v3 / courses_v3.

Exposes:
  hybrid_search(query, course_id, k) -> list[dict]
  semantic_search(query, k) -> list[dict]
  fetch_anchor_chunks(course_ids) -> (list[dict], list[tuple[str,str]], bool)
  get_meeting_times(course_ids) -> dict[str, Any]
  get_syllabus_urls(course_ids) -> dict[str, str]
  get_missing_sections(course_id) -> list[str]
"""
from __future__ import annotations

from typing import Any, Optional

from pymongo import MongoClient

import config

CHUNKS_COLLECTION = "chunks_v3"
COURSES_COLLECTION = "courses_v3"
VECTOR_INDEX = "vector_index_v3"
TEXT_INDEX = "text_index_v3"

_client: Optional[MongoClient] = None


def _get_db():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI)
    return _client[config.MONGODB_DB]


def _projection() -> dict:
    return {"$project": {
        "course_id": 1, "chunk_index": 1, "content": 1,
        "header_text": 1, "anchor": 1, "section": 1, "term": 1, "score": 1,
        "category": 1,
    }}


def _atlas_filter(course_id: str | None, term: str | None) -> dict | None:
    f: dict = {}
    if course_id:
        f["course_id"] = course_id
    if term:
        f["term"] = term
    return f if f else None


def _mongo_filter(course_id: str | None, term: str | None) -> dict:
    f: dict = {}
    if course_id:
        f["course_id"] = course_id
    if term:
        f["term"] = term
    return f


def _build_vector_stage(embedding: list[float], k: int, filters: dict | None) -> dict:
    stage: dict = {
        "$vectorSearch": {
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": k * 10,
            "limit": k,
        }
    }
    if filters:
        stage["$vectorSearch"]["filter"] = filters
    return stage


def _build_text_stage(query: str, k: int, course_id: str | None) -> list[dict]:
    compound: dict = {"must": [{"text": {"query": query, "path": "content"}}]}
    if course_id:
        compound["filter"] = [{"equals": {"path": "course_id", "value": course_id}}]
    return [
        {"$search": {"index": TEXT_INDEX, "compound": compound}},
        {"$limit": k},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
        _projection(),
    ]


def _rrf_fuse(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked lists of dicts with '_id' key."""
    scores: dict = {}
    docs: dict = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = str(doc.get("_id", id(doc)))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            docs[doc_id] = doc
    return [docs[did] for did in sorted(scores, key=scores.__getitem__, reverse=True)]


def hybrid_search(query: str, course_id: str, k: int) -> list[dict]:
    """RRF hybrid search (vector + BM25) filtered to one course."""
    from rag.tools.voyage import embed_query
    from rag.graph.trace_registry import child_span

    db = _get_db()
    emb = embed_query(query)

    atlas_f = _atlas_filter(course_id, None)
    vector_pipeline = [
        _build_vector_stage(emb, k, atlas_f),
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        _projection(),
    ]
    text_pipeline = _build_text_stage(query, k, course_id)

    with child_span("search.mongo_hybrid", {"course_id": course_id, "k": k}) as ctx:
        try:
            vector_results = list(db[CHUNKS_COLLECTION].aggregate(vector_pipeline))
        except Exception:
            vector_results = []
        try:
            text_results = list(db[CHUNKS_COLLECTION].aggregate(text_pipeline))
        except Exception:
            text_results = []

        results = _rrf_fuse([vector_results, text_results])[:k]
        for r in results:
            r.pop("_id", None)
        ctx["output"] = {"n_fused": len(results)}

    return results


def semantic_search(query: str, k: int) -> list[dict]:
    """Corpus-wide semantic vector search."""
    from rag.tools.voyage import embed_query
    from rag.graph.trace_registry import child_span

    db = _get_db()
    emb = embed_query(query)
    pipeline = [
        _build_vector_stage(emb, k, filters=None),
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        _projection(),
    ]
    with child_span("search.mongo_semantic", {"k": k}) as ctx:
        results = list(db[CHUNKS_COLLECTION].aggregate(pipeline))
        for r in results:
            r.pop("_id", None)
        ctx["output"] = {"n_results": len(results)}
    return results


def fetch_anchor_chunks(
    course_ids: list[str],
) -> tuple[list[dict], list[tuple[str, str]], bool]:
    """Fetch anchor chunks for all courses in the recurrent pass.

    Returns (chunks, data_gaps, data_integrity).
    data_gaps = [(course_id, category), ...] for course+category pairs with 0 results.
    data_integrity = False if any gaps found.
    """
    from rag.models import VALID_CATEGORIES

    db = _get_db()
    all_chunks: list[dict] = []
    data_gaps: list[tuple[str, str]] = []

    for course_id in course_ids:
        for category in VALID_CATEGORIES:
            pipeline = [
                {"$match": {"course_id": course_id, "anchor": True}},
                _projection(),
            ]
            results = list(db[CHUNKS_COLLECTION].aggregate(pipeline))
            for r in results:
                r.pop("_id", None)
            if not results:
                data_gaps.append((course_id, category))
            else:
                all_chunks.extend(results)

    data_integrity = len(data_gaps) == 0
    return all_chunks, data_gaps, data_integrity


def get_meeting_times(course_ids: list[str]) -> dict[str, Any]:
    """Return {course_id: meeting_time_string} for the given course IDs."""
    db = _get_db()
    result = {}
    if not course_ids:
        return result
    docs = db[COURSES_COLLECTION].find(
        {"course_id": {"$in": course_ids}},
        {"course_id": 1, "meeting_time": 1, "_id": 0},
    )
    for doc in docs:
        result[doc["course_id"]] = doc.get("meeting_time")
    return result


def get_syllabus_urls(course_ids: list[str]) -> dict[str, str]:
    """Return {course_id: syllabus_url} for the given course IDs."""
    db = _get_db()
    result = {}
    if not course_ids:
        return result
    docs = db[COURSES_COLLECTION].find(
        {"course_id": {"$in": course_ids}},
        {"course_id": 1, "syllabus_url": 1, "_id": 0},
    )
    for doc in docs:
        url = doc.get("syllabus_url")
        if url:
            result[doc["course_id"]] = url
    return result


def get_missing_sections(course_id: str) -> list[str]:
    """Return list of section names missing from course_id's chunks."""
    db = _get_db()
    present = set(
        db[CHUNKS_COLLECTION].distinct("section", {"course_id": course_id})
    )
    from rag.models import VALID_CATEGORIES
    return [s for s in VALID_CATEGORIES if s not in present]
```

- [ ] **Step 2: Verify it's importable**

```bash
python -c "from rag.tools.mongo import hybrid_search, semantic_search, fetch_anchor_chunks; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag/tools/mongo.py
git commit -m "refactor(tools): create tools/mongo.py from search_v3 + document_stores"
```

---

## Task 7: Create tools/mem0.py

Consolidate `rag/v4/mem0_manager.py`, `rag/v4/mem0_registry.py`, `rag/v4/tamu_mem0_llm.py`, `rag/v4/voyage_mem0_embedder.py` into one tool.

**Files:**
- Create: `rag/tools/mem0.py`

- [ ] **Step 1: Read existing source files**

```bash
cat rag/v4/mem0_manager.py
cat rag/v4/mem0_registry.py
cat rag/v4/tamu_mem0_llm.py
cat rag/v4/voyage_mem0_embedder.py
```

- [ ] **Step 2: Write rag/tools/mem0.py**

Merge all four files. The class registration strings (provider names) must be updated to point to `rag.tools.mem0`:

```python
"""Mem0 tool — per-session semantic memory using mem0.

Exposes:
  Mem0Manager         — per-session memory object
  get_mem0_manager()  — registry lookup by session_id
  register_mem0_manager() / clear_mem0_manager()  — lifecycle
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TAMU LLM adapter for mem0
# ---------------------------------------------------------------------------

class TamuMem0LLM:
    """mem0-compatible LLM using TAMU AI gateway."""
    # [Copy entire class body from rag/v4/tamu_mem0_llm.py verbatim]
    # then update any internal imports from rag.v4.* → rag.tools.*

# ---------------------------------------------------------------------------
# Voyage embedder adapter for mem0
# ---------------------------------------------------------------------------

class VoyageMem0Embedder:
    """mem0-compatible embedder using Voyage AI."""
    # [Copy entire class body from rag/v4/voyage_mem0_embedder.py verbatim]
    # then update any internal imports from rag.v4.* → rag.tools.*

# ---------------------------------------------------------------------------
# Mem0Manager
# ---------------------------------------------------------------------------

class Mem0Manager:
    """Manages per-session semantic memories using mem0."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._memory = self._build_memory(session_id)

    @staticmethod
    def _build_memory(session_id: str):
        from mem0 import Memory
        from mem0.configs.llms.base import BaseLlmConfig
        from mem0.utils.factory import EmbedderFactory, LlmFactory
        from qdrant_client import QdrantClient

        # Update class paths to rag.tools.mem0 (not rag.v4.*)
        LlmFactory.provider_to_class["tamu"] = (
            "rag.tools.mem0.TamuMem0LLM",
            BaseLlmConfig,
        )
        EmbedderFactory.provider_to_class["voyage_mem0"] = (
            "rag.tools.mem0.VoyageMem0Embedder"
        )

        collection_name = f"tamubot_{session_id[:8]}"
        qdrant_client = QdrantClient(":memory:")
        config_dict = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "client": qdrant_client,
                    "collection_name": collection_name,
                    "embedding_model_dims": 1024,
                },
            },
            "llm": {"provider": "tamu", "config": {}},
            "embedder": {"provider": "voyage_mem0", "config": {"embedding_dims": 1024}},
            "history_db_path": ":memory:",
        }
        return Memory.from_config(config_dict)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        try:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            self._memory.add(messages, user_id=self.session_id)
        except Exception:
            logger.exception("mem0 add_turn failed (non-fatal)")

    def add_turn_async(self, user_msg: str, assistant_msg: str) -> None:
        t = threading.Thread(target=self.add_turn, args=(user_msg, assistant_msg), daemon=True)
        t.start()

    def search_context(self, query: str, top_k: int = 3) -> str:
        try:
            result = self._memory.search(query, user_id=self.session_id, limit=top_k)
            facts = [r["memory"] for r in result.get("results", []) if r.get("memory")]
            return "\n".join(f"- {f}" for f in facts) if facts else ""
        except Exception:
            logger.exception("mem0 search_context failed (non-fatal)")
            return ""

# ---------------------------------------------------------------------------
# Session registry — maps session_id → Mem0Manager
# ---------------------------------------------------------------------------

_registry: dict[str, Mem0Manager] = {}
_registry_lock = threading.Lock()


def get_mem0_manager(session_id: str) -> Optional[Mem0Manager]:
    """Return existing Mem0Manager for session_id, or None if not registered."""
    with _registry_lock:
        return _registry.get(session_id)


def register_mem0_manager(session_id: str) -> Mem0Manager:
    """Create and register a Mem0Manager for session_id if not exists."""
    with _registry_lock:
        if session_id not in _registry:
            _registry[session_id] = Mem0Manager(session_id)
        return _registry[session_id]


def clear_mem0_manager(session_id: str) -> None:
    """Remove Mem0Manager for session_id."""
    with _registry_lock:
        _registry.pop(session_id, None)
```

> **Note:** For `TamuMem0LLM` and `VoyageMem0Embedder`, copy the class bodies verbatim from `rag/v4/tamu_mem0_llm.py` and `rag/v4/voyage_mem0_embedder.py`. The only change is the class registration strings above.

- [ ] **Step 3: Verify it's importable**

```bash
python -c "from rag.tools.mem0 import Mem0Manager, get_mem0_manager; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add rag/tools/mem0.py
git commit -m "refactor(tools): consolidate mem0 files into tools/mem0.py"
```

---

## Task 8: Create state/pipeline_state.py

Move `rag/v4/state.py` → `rag/state/pipeline_state.py`. Add `RouterResult` dataclass (currently in `rag/router.py`) since it appears in state and must be importable without circular deps.

**Files:**
- Create: `rag/state/pipeline_state.py`

- [ ] **Step 1: Write rag/state/pipeline_state.py**

```python
"""Pipeline state TypedDicts — data contracts for the RAG graph.

All data flowing between nodes is defined here. Nodes only read/write
these typed fields; no other shared state exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import TypedDict

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# RouterResult — structured output of the router node
# ---------------------------------------------------------------------------

def _derive_function(course_ids: list[str], recurrent_search: bool, intent_type: Optional[str]) -> str:
    if not course_ids:
        return "semantic_general" if intent_type is not None else "out_of_scope"
    return "recurrent" if recurrent_search else "hybrid_course"


def _derive_retrieval_mode(course_ids: list[str], recurrent_search: bool) -> str:
    if not course_ids:
        return "semantic"
    if recurrent_search:
        return "hybrid"
    return "hybrid_course"


@dataclass
class RouterResult:
    """Structured output from the query router."""
    course_ids: list[str] = field(default_factory=list)
    intent_type: Optional[str] = None
    recurrent_search: bool = False
    rewritten_query: str = ""
    section: Optional[str] = None
    retrieval_mode: str = ""
    function: str = ""

    def __post_init__(self):
        if not self.function:
            self.function = _derive_function(self.course_ids, self.recurrent_search, self.intent_type)
        if not self.retrieval_mode:
            self.retrieval_mode = _derive_retrieval_mode(self.course_ids, self.recurrent_search)

    @property
    def requires_retrieval(self) -> bool:
        return bool(self.course_ids) or self.intent_type is not None


def normalize_course_id(raw: str) -> str:
    """Normalize 'csce638' → 'CSCE 638'."""
    raw = raw.strip().upper().replace("-", " ")
    match = re.match(r"^([A-Z]+)\s*(\d+.*)$", raw)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return raw


# ---------------------------------------------------------------------------
# PipelineState — main state contract
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    query: str
    router_result: Optional[RouterResult]
    rewritten_query: str
    function: str               # "hybrid_course"|"recurrent"|"semantic_general"|"out_of_scope"
    course_ids: list[str]
    intent_type: Optional[str]
    recurrent_search: bool
    requires_retrieval: bool
    anchor_chunks: list[dict]
    eval_query: str
    discovery_chunks: list[dict]
    retrieved_chunks: list[dict]
    data_gaps: list[tuple[str, str]]
    data_integrity: bool
    conflicted_course_ids: list[str]
    answer: str
    answer_stream: Optional[list]   # list[str] tokens — picklable, checkpointed by LangGraph
    trace: Optional[Any]            # LFTrace — NOT checkpointed (not picklable)
    timing_ms: dict[str, float]
    error: Optional[str]
    node_trace: list[str]


class ConversationMessage(TypedDict, total=False):
    role: str           # "user" | "assistant"
    content: str
    router_result: Optional[dict]   # serializable summary of RouterResult


class ConversationState(PipelineState, total=False):
    """Extends PipelineState with multi-turn session fields."""
    session_id: str
    history: list[ConversationMessage]
    history_summary: str
    history_context: str
    turn_number: int
    router_cache: dict
    retrieval_cache: dict
    answer_cache: dict
```

- [ ] **Step 2: Update rag/state/__init__.py to re-export**

```python
# rag/state/__init__.py
from rag.state.pipeline_state import (
    RouterResult, PipelineState, ConversationState, ConversationMessage,
    normalize_course_id,
)

__all__ = [
    "RouterResult", "PipelineState", "ConversationState",
    "ConversationMessage", "normalize_course_id",
]
```

- [ ] **Step 3: Verify**

```bash
python -c "from rag.state import RouterResult, PipelineState, ConversationState; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add rag/state/pipeline_state.py rag/state/__init__.py
git commit -m "refactor(state): add state/pipeline_state.py with RouterResult"
```

---

## Task 9: Create graph/middleware.py, graph/trace_registry.py, graph/cache_utils.py, graph/exceptions.py

Move infrastructure files from `rag/v4/` to `rag/graph/`.

**Files:**
- Create: `rag/graph/middleware.py`, `rag/graph/trace_registry.py`
- Create: `rag/graph/cache_utils.py`, `rag/graph/exceptions.py`

- [ ] **Step 1: Copy and update trace_registry**

```bash
cp rag/v4/trace_registry.py rag/graph/trace_registry.py
```

Update docstring header in `rag/graph/trace_registry.py`:
```python
"""Thread-safe trace registry for the RAG graph.
...
Canonical location: rag/graph/trace_registry.py
"""
```

- [ ] **Step 2: Copy and update exceptions**

```bash
cp rag/v4/exceptions.py rag/graph/exceptions.py
```

Update docstring in `rag/graph/exceptions.py`:
```python
"""Custom exceptions for the RAG pipeline.
Canonical location: rag/graph/exceptions.py
"""
```

- [ ] **Step 3: Copy cache_utils**

```bash
cp rag/v4/cache_utils.py rag/graph/cache_utils.py
```

- [ ] **Step 4: Create graph/middleware.py**

Copy `rag/v4/middleware.py` to `rag/graph/middleware.py`, then update the internal import at the top:

```python
# In rag/graph/middleware.py — change this import:
# BEFORE:
import rag.v4.trace_registry as _trace_reg
from rag.v4.exceptions import V4PipelineError

# AFTER:
import rag.graph.trace_registry as _trace_reg
from rag.graph.exceptions import V4PipelineError
```

Also update the state import in the wrapper:
```python
# In tracing_middleware wrapper, the state import is fine (PipelineState used as Any)
# No other changes needed in the decorator logic
```

- [ ] **Step 5: Verify**

```bash
python -c "
from rag.graph.middleware import tracing_middleware, timing_middleware, error_guard_middleware
from rag.graph.trace_registry import current_span, push_span, pop_span, child_span
from rag.graph.exceptions import V4PipelineError
from rag.graph.cache_utils import normalize_query
print('OK')
"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add rag/graph/middleware.py rag/graph/trace_registry.py rag/graph/cache_utils.py rag/graph/exceptions.py
git commit -m "refactor(graph): migrate middleware + trace_registry + cache_utils + exceptions"
```

---

## Task 10: Create graph/checkpointer.py and graph/session.py

**Files:**
- Create: `rag/graph/checkpointer.py`, `rag/graph/session.py`

- [ ] **Step 1: Copy checkpointer**

```bash
cp rag/v4/checkpointer.py rag/graph/checkpointer.py
```

Update docstring if it references `v4`:
```python
"""LangGraph SqliteSaver wrapper.
Canonical location: rag/graph/checkpointer.py
"""
```

- [ ] **Step 2: Copy session**

```bash
cp rag/v4/session.py rag/graph/session.py
```

- [ ] **Step 3: Verify**

```bash
python -c "
from rag.graph.checkpointer import make_checkpointer
from rag.graph.session import SessionManager
print('OK')
"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add rag/graph/checkpointer.py rag/graph/session.py
git commit -m "refactor(graph): migrate checkpointer + session"
```

---

## Task 11: Create edges/routing.py

Extract the two conditional edge functions from `rag/v4/graph.py`.

**Files:**
- Create: `rag/edges/routing.py`

- [ ] **Step 1: Write rag/edges/routing.py**

```python
"""Conditional edge functions for the RAG graph.

Each function reads one state field and returns a node name string.
"""
from __future__ import annotations

from rag.state.pipeline_state import PipelineState


def route_after_router(state: PipelineState) -> str:
    """Dispatch to retrieval path based on function type."""
    function = state.get("function", "out_of_scope")
    if function == "out_of_scope":
        return "out_of_scope"
    elif function == "recurrent":
        return "anchor"
    else:
        return "retrieval"


def route_after_retrieval(state: PipelineState) -> str:
    """After retrieval: recurrent → schedule_filter, others → generator."""
    function = state.get("function", "out_of_scope")
    if function == "recurrent":
        return "schedule_filter"
    return "generator"
```

- [ ] **Step 2: Update rag/edges/__init__.py**

```python
# rag/edges/__init__.py
from rag.edges.routing import route_after_router, route_after_retrieval

__all__ = ["route_after_router", "route_after_retrieval"]
```

- [ ] **Step 3: Verify**

```bash
python -c "from rag.edges import route_after_router, route_after_retrieval; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add rag/edges/routing.py rag/edges/__init__.py
git commit -m "refactor(edges): create edges/routing.py from v4/graph.py edge functions"
```

---

## Task 12: Migrate Simple Nodes (out_of_scope, merge, anchor, schedule_filter)

These nodes have minimal logic; the DI change is just replacing `registry.X.method()` calls with direct tool imports.

**Files:**
- Create: `rag/nodes/out_of_scope_node.py`, `rag/nodes/merge_node.py`
- Create: `rag/nodes/anchor_node.py`, `rag/nodes/schedule_filter_node.py`

- [ ] **Step 1: Create nodes/out_of_scope_node.py**

Copy `rag/v4/nodes/out_of_scope_node.py`, then replace:
```python
# BEFORE imports:
from rag.v4.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.v4.state import PipelineState

# AFTER imports:
from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.state.pipeline_state import PipelineState
```

Remove `registry: Any` from function signature:
```python
# BEFORE:
def out_of_scope_node(state: PipelineState, registry: Any) -> dict:

# AFTER:
def out_of_scope_node(state: PipelineState) -> dict:
```

Remove `from typing import Any` if no longer used.

- [ ] **Step 2: Create nodes/anchor_node.py**

Copy `rag/v4/nodes/anchor_node.py`, update imports same as above, change signature, replace registry call:
```python
# BEFORE:
chunks, data_gaps, data_integrity = registry.retriever.fetch_anchor_chunks(course_ids)

# AFTER:
from rag.tools.mongo import fetch_anchor_chunks
chunks, data_gaps, data_integrity = fetch_anchor_chunks(course_ids)
```

- [ ] **Step 3: Create nodes/merge_node.py**

Copy `rag/v4/nodes/merge_node.py`, update imports, change signature, inline `deduplicate_chunks` (from `rag/router.py`) and replace reranker call:

The full `deduplicate_chunks` function from `rag/router.py` (lines 282-297):
```python
def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks by (course_id, chunk_index)."""
    seen: set[tuple] = set()
    result = []
    for c in chunks:
        key = (c.get("course_id", ""), c.get("chunk_index", ""))
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result
```

Replace registry call:
```python
# BEFORE:
combined = deduplicate_chunks(anchor_chunks + discovery_chunks)
reranked = registry.reranker.rerank(query, combined, top_k=len(combined))

# AFTER:
from rag.tools.voyage import rerank
combined = _deduplicate_chunks(anchor_chunks + discovery_chunks)
reranked = rerank(query, combined, top_k=len(combined))
```

Remove old import: `from rag.router import deduplicate_chunks`

- [ ] **Step 4: Create nodes/schedule_filter_node.py**

Copy `rag/v4/nodes/schedule_filter_node.py`, update imports, change signature, replace registry calls:
```python
# BEFORE:
meeting_times = registry.retriever.get_meeting_times(course_ids)
# ...
disc_mt_map = registry.retriever.get_meeting_times(disc_cids)

# AFTER:
from rag.tools.mongo import get_meeting_times
meeting_times = get_meeting_times(course_ids)
# ...
disc_mt_map = get_meeting_times(disc_cids)
```

Also update schedule import:
```python
# BEFORE:
from rag.schedule import filter_conflicting_courses, parse_meeting_times

# AFTER:
from rag.tools.schedule import filter_conflicting_courses, parse_meeting_times
```

- [ ] **Step 5: Verify all four nodes are importable**

```bash
python -c "
from rag.nodes.out_of_scope_node import out_of_scope_node
from rag.nodes.anchor_node import anchor_node
from rag.nodes.merge_node import merge_node
from rag.nodes.schedule_filter_node import schedule_filter_node
print('OK')
"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add rag/nodes/out_of_scope_node.py rag/nodes/anchor_node.py rag/nodes/merge_node.py rag/nodes/schedule_filter_node.py
git commit -m "refactor(nodes): migrate simple nodes with DI removed"
```

---

## Task 13: Migrate History Nodes

**Files:**
- Create: `rag/nodes/history_inject_node.py`, `rag/nodes/history_update_node.py`

- [ ] **Step 1: Create nodes/history_inject_node.py**

Copy `rag/v4/nodes/history_inject_node.py`, update imports:
```python
# BEFORE:
from rag.v4.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.v4.state import PipelineState

# AFTER:
from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.state.pipeline_state import PipelineState
```

Change signature (remove `registry: Any`):
```python
def history_inject_node(state: PipelineState) -> dict:
```

Update mem0 registry import:
```python
# BEFORE:
from rag.v4.mem0_registry import get as get_mem0_manager

# AFTER:
from rag.tools.mem0 import get_mem0_manager
```

- [ ] **Step 2: Create nodes/history_update_node.py**

Copy `rag/v4/nodes/history_update_node.py`, update imports:
```python
# BEFORE:
from rag.v4.middleware import ...
from rag.v4.state import ConversationMessage, ConversationState

# AFTER:
from rag.graph.middleware import ...
from rag.state.pipeline_state import ConversationMessage, ConversationState
```

Change signature (remove `registry: Any`):
```python
def history_update_node(state: ConversationState) -> dict:
```

Update mem0 and cache imports:
```python
# BEFORE:
from rag.v4.mem0_registry import get as get_mem0_manager
from rag.v4.cache_utils import normalize_query

# AFTER:
from rag.tools.mem0 import get_mem0_manager
from rag.graph.cache_utils import normalize_query
```

- [ ] **Step 3: Verify**

```bash
python -c "
from rag.nodes.history_inject_node import history_inject_node
from rag.nodes.history_update_node import history_update_node
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add rag/nodes/history_inject_node.py rag/nodes/history_update_node.py
git commit -m "refactor(nodes): migrate history nodes with DI removed"
```

---

## Task 14: Migrate eval_search_node.py

**Files:**
- Create: `rag/nodes/eval_search_node.py`

The node currently calls `registry.generator_llm.generate_eval_query()` which delegates to `rag.generator.generate_eval_search_string()`. Since we're removing the component layer, the node calls the generator function directly.

- [ ] **Step 1: Create nodes/eval_search_node.py**

Copy `rag/v4/nodes/eval_search_node.py`, update imports and remove DI:

```python
"""Eval search node — generates the discovery search string for the recurrent pass."""
from __future__ import annotations

from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.graph.trace_registry import current_span as _current_span, get as _get_trace
from rag.state.pipeline_state import PipelineState


@tracing_middleware
@timing_middleware
@error_guard_middleware
def eval_search_node(state: PipelineState) -> dict:
    """Generate eval query from anchor chunks. Only runs for recurrent path."""
    from rag.generator import generate_eval_search_string
    query = state.get("rewritten_query") or state.get("query", "")
    anchor_chunks = state.get("anchor_chunks", [])
    trace = _current_span() or _get_trace(state.get("session_id", "")) or state.get("trace")
    node_trace = list(state.get("node_trace", []))
    node_trace.append("eval_search")

    try:
        eval_query = generate_eval_search_string(anchor_chunks, query, "GENERAL", parent_span=trace)
        return {"eval_query": eval_query, "node_trace": node_trace}
    except Exception as e:
        return {"eval_query": query, "error": f"Eval search failed: {e}", "node_trace": node_trace}
```

> **Note:** `rag/generator.py` is kept as-is (not deleted yet) because `eval_search_node` and `generator_node` still reference it until Task 17 cleans up. The shim pattern is used in reverse here: the new node imports from the old location until we inline the logic.

- [ ] **Step 2: Verify**

```bash
python -c "from rag.nodes.eval_search_node import eval_search_node; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add rag/nodes/eval_search_node.py
git commit -m "refactor(nodes): migrate eval_search_node with DI removed"
```

---

## Task 15: Migrate router_node.py — inline classify_query

The router node currently calls `registry.router_llm.classify()` which wraps `classify_query()` from `rag/router.py`. After the refactor, the node calls `classify_query()` directly.

**Files:**
- Create: `rag/nodes/router_node.py`

- [ ] **Step 1: Create nodes/router_node.py**

Copy `rag/v4/nodes/router_node.py`, then apply these changes:

Update imports:
```python
# BEFORE:
from rag.v4.cache_utils import normalize_query
from rag.v4.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.v4.state import PipelineState
from rag.v4.trace_registry import current_span as _current_span, get as _get_trace

# AFTER:
from rag.graph.cache_utils import normalize_query
from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.graph.trace_registry import current_span as _current_span, get as _get_trace
from rag.state.pipeline_state import PipelineState
```

Change signature:
```python
# BEFORE:
def router_node(state: PipelineState, registry: Any) -> dict:

# AFTER:
def router_node(state: PipelineState) -> dict:
```

Replace the registry call with direct classify_query:
```python
# BEFORE:
router_result = registry.router_llm.classify(query, trace=trace, prior_context=prior_context)

# AFTER:
from rag.router import classify_query
router_result = classify_query(query, router_span=trace, prior_context=prior_context)
```

Remove `from typing import Any`.

> **Note:** `rag/router.py` (legacy) is kept until Task 18, so `classify_query` is still importable from `rag.router`.

- [ ] **Step 2: Verify**

```bash
python -c "from rag.nodes.router_node import router_node; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add rag/nodes/router_node.py
git commit -m "refactor(nodes): migrate router_node — call classify_query directly"
```

---

## Task 16: Migrate retrieval_node.py — call tools directly

**Files:**
- Create: `rag/nodes/retrieval_node.py`

- [ ] **Step 1: Create nodes/retrieval_node.py**

Copy `rag/v4/nodes/retrieval_node.py`, apply changes:

Update imports:
```python
# BEFORE:
from rag.router import compute_dynamic_k
from rag.v4.middleware import ...
from rag.v4.state import PipelineState

# AFTER:
from rag.graph.cache_utils import normalize_query
from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.state.pipeline_state import PipelineState
```

Inline `compute_dynamic_k` (copy from `rag/router.py` lines 23-37):
```python
def _compute_dynamic_k(function: str, n_courses: int) -> dict[str, int]:
    """Compute retrieve_k scaled by number of courses."""
    import config
    base = config.PER_COURSE_K[function]
    if function == "semantic_general":
        return dict(base)
    n = max(1, n_courses)
    return {
        "retrieve_k": min(base["retrieve_k"] * n, config.MAX_RETRIEVE_K),
        "rerank_k": min(base["rerank_k"] * n, config.MAX_RERANK_K),
    }
```

Inline `_make_retrieval_cache_key`:
```python
def _make_retrieval_cache_key(function, course_ids, rewritten_query, eval_query):
    if function == "recurrent":
        return f"recurrent|{normalize_query(eval_query)}"
    return f"{sorted(course_ids)}|{normalize_query(rewritten_query)}"
```

Change signature:
```python
def retrieval_node(state: PipelineState) -> dict:
```

Replace all `registry.retriever.*` and `registry.reranker.*` calls:
```python
# BEFORE:
dk = compute_dynamic_k(function, len(course_ids))
# ...
chunks = registry.retriever.hybrid_search(rewritten_query, cid, retrieve_k)
# ...
reranked = registry.reranker.rerank(rewritten_query, all_chunks, top_k=len(all_chunks))
# ...
chunks = registry.retriever.semantic_search(rewritten_query, retrieve_k)
# ...
all_results = registry.retriever.semantic_search(eval_query, retrieve_k)

# AFTER:
from rag.tools.mongo import hybrid_search, semantic_search
from rag.tools.voyage import rerank as voyage_rerank

dk = _compute_dynamic_k(function, len(course_ids))
# ...
chunks = hybrid_search(rewritten_query, cid, retrieve_k)
# ...
reranked = voyage_rerank(rewritten_query, all_chunks, top_k=len(all_chunks))
# ...
chunks = semantic_search(rewritten_query, retrieve_k)
# ...
all_results = semantic_search(eval_query, retrieve_k)
```

- [ ] **Step 2: Verify**

```bash
python -c "from rag.nodes.retrieval_node import retrieval_node; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add rag/nodes/retrieval_node.py
git commit -m "refactor(nodes): migrate retrieval_node — call mongo/voyage tools directly"
```

---

## Task 17: Migrate generator_node.py — call generate_stream directly

**Files:**
- Create: `rag/nodes/generator_node.py`

- [ ] **Step 1: Create nodes/generator_node.py**

Copy `rag/v4/nodes/generator_node.py`, update imports and signature:

```python
"""Generator node — streams the final answer."""
from __future__ import annotations

from rag.graph.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.graph.trace_registry import current_span as _current_span, get as _get_trace
from rag.state.pipeline_state import PipelineState


@tracing_middleware
@timing_middleware
@error_guard_middleware
def generator_node(state: PipelineState) -> dict:
    """Generate the answer. Stores answer_stream as list[str] (picklable for LangGraph)."""
    from rag.generator import generate_stream
    node_trace = list(state.get("node_trace", []))
    node_trace.append("generator")

    # Prefer current node span (set by tracing_middleware), fall back to root trace
    trace = _current_span() or _get_trace(state.get("session_id", "")) or state.get("trace")

    try:
        tokens = list(generate_stream(
            results=state.get("retrieved_chunks", []),
            question=state.get("rewritten_query") or state.get("query", ""),
            function=state.get("function", "semantic_general"),
            course_ids=state.get("course_ids", []),
            intent_type=state.get("intent_type"),
            data_gaps=state.get("data_gaps", []),
            data_integrity=state.get("data_integrity", True),
            conflicted_course_ids=state.get("conflicted_course_ids", []),
            trace=trace,
            history_context=state.get("history_context"),
        ))
        return {
            "answer": "".join(tokens),
            "answer_stream": tokens,
            "node_trace": node_trace,
        }
    except Exception as e:
        err_msg = f"Generation failed: {e}"
        return {
            "answer": err_msg,
            "answer_stream": [err_msg],
            "error": err_msg,
            "node_trace": node_trace,
        }
```

- [ ] **Step 2: Verify**

```bash
python -c "from rag.nodes.generator_node import generator_node; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add rag/nodes/generator_node.py
git commit -m "refactor(nodes): migrate generator_node — call generate_stream directly"
```

---

## Task 18: Create graph/builder.py and graph/pipeline.py

**Files:**
- Create: `rag/graph/builder.py`, `rag/graph/pipeline.py`

- [ ] **Step 1: Write rag/graph/builder.py**

```python
"""Build the RAG LangGraph state machine."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag.edges.routing import route_after_retrieval, route_after_router
from rag.nodes.anchor_node import anchor_node
from rag.nodes.eval_search_node import eval_search_node
from rag.nodes.generator_node import generator_node
from rag.nodes.merge_node import merge_node
from rag.nodes.out_of_scope_node import out_of_scope_node
from rag.nodes.retrieval_node import retrieval_node
from rag.nodes.router_node import router_node
from rag.nodes.schedule_filter_node import schedule_filter_node
from rag.state.pipeline_state import ConversationState, PipelineState


def build_graph():
    """Build and compile the RAG pipeline graph (stateless, no conversation memory)."""
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("anchor", anchor_node)
    graph.add_node("eval_search", eval_search_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("schedule_filter", schedule_filter_node)
    graph.add_node("merge", merge_node)
    graph.add_node("generator", generator_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"out_of_scope": "out_of_scope", "anchor": "anchor", "retrieval": "retrieval"},
    )

    graph.add_edge("anchor", "eval_search")
    graph.add_edge("eval_search", "retrieval")

    graph.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {"schedule_filter": "schedule_filter", "generator": "generator"},
    )

    graph.add_edge("schedule_filter", "merge")
    graph.add_edge("merge", "generator")
    graph.add_edge("generator", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile()


def build_graph_with_memory(checkpointer=None):
    """Build the RAG pipeline graph with conversation memory support."""
    from rag.nodes.history_inject_node import history_inject_node
    from rag.nodes.history_update_node import history_update_node

    graph = StateGraph(ConversationState)

    graph.add_node("history_inject", history_inject_node)
    graph.add_node("router", router_node)
    graph.add_node("anchor", anchor_node)
    graph.add_node("eval_search", eval_search_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("schedule_filter", schedule_filter_node)
    graph.add_node("merge", merge_node)
    graph.add_node("generator", generator_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("history_update", history_update_node)

    graph.set_entry_point("history_inject")
    graph.add_edge("history_inject", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"out_of_scope": "out_of_scope", "anchor": "anchor", "retrieval": "retrieval"},
    )

    graph.add_edge("anchor", "eval_search")
    graph.add_edge("eval_search", "retrieval")

    graph.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {"schedule_filter": "schedule_filter", "generator": "generator"},
    )

    graph.add_edge("schedule_filter", "merge")
    graph.add_edge("merge", "generator")
    graph.add_edge("generator", "history_update")
    graph.add_edge("out_of_scope", "history_update")
    graph.add_edge("history_update", END)

    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return graph.compile(**kwargs)
```

- [ ] **Step 2: Write rag/graph/pipeline.py**

```python
"""RAG pipeline entry point."""
from __future__ import annotations

from typing import Any, Optional

import rag.graph.trace_registry as _trace_registry
from rag.graph.builder import build_graph, build_graph_with_memory
from rag.state.pipeline_state import PipelineState

_graph = None
_memory_graph = None

_INITIAL_STATE: dict = {
    "node_trace": [],
    "timing_ms": {},
    "conflicted_course_ids": [],
    "data_gaps": [],
    "data_integrity": True,
    "anchor_chunks": [],
    "discovery_chunks": [],
    "retrieved_chunks": [],
}


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(
    query: str,
    trace=None,
    return_timing: bool = False,
) -> tuple:
    """Run the RAG pipeline (stateless).

    Returns:
        (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids)
        or if return_timing=True: adds timing_ms dict as 6th element.
    """
    initial_state: PipelineState = {**_INITIAL_STATE, "query": query, "trace": trace}
    result = _get_graph().invoke(initial_state)

    five_tuple = (
        result.get("retrieved_chunks", []),
        result.get("router_result"),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        result.get("conflicted_course_ids", []),
    )
    if return_timing:
        return (*five_tuple, result.get("timing_ms", {}))
    return five_tuple


def run_pipeline_with_memory(
    query: str,
    trace=None,
    thread_config: Optional[dict] = None,
) -> tuple[list[dict], Any, list[tuple[str, str]], bool, list[str], list[str]]:
    """Run the RAG pipeline with conversation memory.

    Returns 6-tuple: (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids, answer_tokens)
    """
    global _memory_graph
    if _memory_graph is None:
        from rag.graph.checkpointer import make_checkpointer
        checkpointer = make_checkpointer()
        _memory_graph = build_graph_with_memory(checkpointer=checkpointer)

    session_id = ""
    if thread_config:
        session_id = thread_config.get("configurable", {}).get("thread_id", "")

    initial_state: dict = {
        **_INITIAL_STATE,
        "query": query,
        "session_id": session_id,
    }

    if session_id and trace is not None:
        _trace_registry.register(session_id, trace)
    try:
        invoke_kwargs = {}
        if thread_config:
            invoke_kwargs["config"] = thread_config
        result = _memory_graph.invoke(initial_state, **invoke_kwargs)
    finally:
        _trace_registry.clear(session_id)

    answer_str = result.get("answer") or ""
    return (
        result.get("retrieved_chunks", []),
        result.get("router_result"),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        result.get("conflicted_course_ids", []),
        [answer_str] if answer_str else [],
    )


def get_current_state(thread_config: dict) -> dict:
    """Read conversation state for a thread. Returns empty dict if no state exists."""
    global _memory_graph
    if _memory_graph is None:
        return {}
    try:
        snapshot = _memory_graph.get_state(thread_config)
        return snapshot.values if snapshot and snapshot.values else {}
    except Exception:
        return {}
```

- [ ] **Step 3: Verify graph builds**

```bash
python -c "from rag.graph.builder import build_graph; g = build_graph(); print('graph nodes:', list(g.nodes)); print('OK')"
```
Expected: lists node names, no errors

- [ ] **Step 4: Commit**

```bash
git add rag/graph/builder.py rag/graph/pipeline.py
git commit -m "refactor(graph): create graph/builder.py + graph/pipeline.py (no DI)"
```

---

## Task 19: Update rag/__init__.py

Shrink public API to just pipeline entry + observability + data models.

**Files:**
- Modify: `rag/__init__.py`

- [ ] **Step 1: Rewrite rag/__init__.py**

```python
"""rag — RAG pipeline public API.

Import only from this module, not from submodules.
"""
from rag.graph.pipeline import get_current_state, run_pipeline, run_pipeline_with_memory
from rag.models import VALID_CATEGORIES, ChunkDoc, CourseDoc, PolicyDoc
from rag.state.pipeline_state import ConversationState, PipelineState, RouterResult
from rag.tools.langfuse import compute_ragas_metrics, get_langfuse, run_ragas_background

__all__ = [
    # Pipeline entry points
    "run_pipeline",
    "run_pipeline_with_memory",
    "get_current_state",
    # Data types
    "ChunkDoc", "CourseDoc", "PolicyDoc", "VALID_CATEGORIES",
    "PipelineState", "ConversationState", "RouterResult",
    # Observability
    "get_langfuse", "run_ragas_background", "compute_ragas_metrics",
]
```

- [ ] **Step 2: Run tests — expect some failures from removed exports**

```bash
make test 2>&1 | grep -E "^(FAIL|ERROR|PASSED|tests/)" | head -30
```

Note which tests fail due to removed imports (expected: tests importing `classify_query`, `generate`, `hybrid_search`, etc. from `rag` will fail). These will be fixed in Task 20.

- [ ] **Step 3: Commit**

```bash
git add rag/__init__.py
git commit -m "refactor: shrink rag/__init__.py to pipeline + observability + models"
```

---

## Task 20: Update Test Imports

**Files:**
- Modify: all test files that import from `rag.v4.*` or removed `rag.*` paths
- Delete: `tests/test_v4_interfaces.py`, `tests/test_v4_swappability.py`

- [ ] **Step 1: Delete tests for the removed DI layer**

```bash
git rm tests/test_v4_interfaces.py tests/test_v4_swappability.py
```

These tests verify `ComponentRegistry`, `Protocols`, and component swapping — concepts that no longer exist.

- [ ] **Step 2: Update test_v4_middleware.py**

```python
# BEFORE:
from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.exceptions import V4PipelineError

# AFTER:
from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.graph.exceptions import V4PipelineError
```

- [ ] **Step 3: Update test_v4_router_node.py**

```python
# BEFORE:
from rag.v4.nodes.router_node import router_node
from rag.router import RouterResult

# AFTER:
from rag.nodes.router_node import router_node
from rag.state.pipeline_state import RouterResult
```

In test body: any `router_node(state, registry=mock)` calls become `router_node(state)`. Mock the underlying `classify_query` at the module level instead:
```python
# BEFORE:
mock_registry = MagicMock()
mock_registry.router_llm.classify.return_value = RouterResult(...)
result = router_node(state, registry=mock_registry)

# AFTER:
from unittest.mock import patch
with patch("rag.router.classify_query", return_value=RouterResult(...)):
    result = router_node(state)
```

- [ ] **Step 4: Update test_v4_graph.py**

```python
# BEFORE:
from rag.v4.graph import build_graph
from rag.v4.interfaces import ComponentRegistry
from rag.router import RouterResult

# AFTER:
from rag.graph.builder import build_graph
from rag.state.pipeline_state import RouterResult
```

Remove all `ComponentRegistry(...)` construction. `build_graph()` now takes no arguments:
```python
# BEFORE:
graph = build_graph(registry)

# AFTER:
graph = build_graph()
```

For tests that inject mock behavior, use `unittest.mock.patch` on the tool functions:
```python
from unittest.mock import patch, MagicMock

# Patch classify_query so router_node doesn't call real LLM
with patch("rag.router.classify_query", return_value=RouterResult(
    course_ids=["TEST_001"], function="hybrid_course", intent_type="ACADEMIC",
)):
    # Patch retrieval tools so no real DB calls
    with patch("rag.tools.mongo.hybrid_search", return_value=[{"course_id": "TEST_001", "content": "chunk"}]):
        with patch("rag.tools.voyage.rerank", side_effect=lambda q, c, top_k: c):
            with patch("rag.generator.generate_stream", return_value=iter(["answer"])):
                result = graph.invoke({...})
```

- [ ] **Step 5: Update test_v4_state.py**

```python
# BEFORE:
from rag.v4.state import PipelineState, ConversationState, ConversationMessage

# AFTER:
from rag.state.pipeline_state import PipelineState, ConversationState, ConversationMessage
```

- [ ] **Step 6: Update test_v4_session.py**

```python
# BEFORE:
from rag.v4.session import SessionManager

# AFTER:
from rag.graph.session import SessionManager
```

- [ ] **Step 7: Update test_v4_history_node.py**

```python
# BEFORE:
from rag.v4.nodes.history_inject_node import history_inject_node
from rag.v4.nodes.history_update_node import history_update_node
from rag.v4.state import ConversationState

# AFTER:
from rag.nodes.history_inject_node import history_inject_node
from rag.nodes.history_update_node import history_update_node
from rag.state.pipeline_state import ConversationState
```

- [ ] **Step 8: Update remaining test files**

Apply the same pattern to these files — update all `rag.v4.*` imports:

| Test file | Old import | New import |
|-----------|-----------|-----------|
| `test_v4_recurrent_path.py` | `rag.v4.nodes.*` | `rag.nodes.*` |
| `test_v4_routing_matrix.py` | `rag.v4.graph`, `rag.router.RouterResult` | `rag.graph.builder`, `rag.state.pipeline_state.RouterResult` |
| `test_v4_pipeline_timing.py` | `rag.v4.middleware`, `rag.v4.pipeline_v4` | `rag.graph.middleware`, `rag.graph.pipeline` |
| `test_v4_mem0_cache.py` | `rag.v4.mem0_registry`, `rag.v4.mem0_manager` | `rag.tools.mem0` |
| `test_v4_observability.py` | `rag.observability` | `rag.tools.langfuse` |
| `test_v4_integration.py` | `rag.v4.pipeline_v4`, `rag.v4.interfaces` | `rag.graph.pipeline`, patch tools |
| `test_v4_components.py` | `rag.v4.components.*` | patch `rag.tools.*` directly |
| `test_v3_v4_parity.py` | `rag.v4.pipeline_v4` | `rag.graph.pipeline` |
| `test_generator.py` | `rag.context_builder`, `rag.gates` | `rag.tools.context`, inline in generator |

- [ ] **Step 9: Run tests**

```bash
make test
```
Expected: all pre-existing passing tests pass; previously failing tests still at same count

- [ ] **Step 10: Commit**

```bash
git add tests/
git commit -m "test: update all test imports to new rag/ structure"
```

---

## Task 21: Update app.py Imports

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update all rag imports in app.py**

Current imports in `app.py` and their replacements:

```python
# BEFORE:
from rag.observability import get_langfuse
# AFTER:
from rag.tools.langfuse import get_langfuse

# BEFORE:
from rag.search_v3 import get_syllabus_urls
# AFTER:
from rag.tools.mongo import get_syllabus_urls

# BEFORE:
from rag.v4.pipeline_v4 import run_pipeline_v4_with_memory
# AFTER:
from rag.graph.pipeline import run_pipeline_with_memory

# BEFORE:
from rag.v4.session import SessionManager
# AFTER:
from rag.graph.session import SessionManager

# BEFORE:
from rag.v4 import mem0_registry
from rag.v4.mem0_manager import Mem0Manager
# AFTER:
from rag.tools import mem0 as mem0_registry
from rag.tools.mem0 import Mem0Manager

# BEFORE:
from rag.v4.cache_utils import normalize_query as _norm
# AFTER:
from rag.graph.cache_utils import normalize_query as _norm

# BEFORE:
from rag.v4.pipeline_v4 import get_current_state
# AFTER:
from rag.graph.pipeline import get_current_state
```

Also update any call: `run_pipeline_v4_with_memory(...)` → `run_pipeline_with_memory(...)`

And any `mem0_registry.get(...)` / `mem0_registry.register(...)` calls:
```python
# BEFORE:
mem0_manager = mem0_registry.get(session_id)

# AFTER — rag.tools.mem0 exposes get_mem0_manager / register_mem0_manager:
mem0_manager = mem0_registry.get_mem0_manager(session_id)
```

- [ ] **Step 2: Verify app.py imports**

```bash
python -c "import app; print('OK')" 2>&1 | head -5
```
Expected: no ImportError

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "refactor(app): update rag.v4.* imports to rag.graph.* / rag.tools.*"
```

---

## Task 22: Delete Legacy Files

**Files:**
- Delete: all shims and original files from `rag/` top-level
- Delete: entire `rag/v4/` directory

- [ ] **Step 1: Verify no remaining imports of legacy paths**

```bash
grep -rn "from rag\.v4\." rag/ app.py tests/ evals/ --include="*.py" | grep -v "\.pyc"
grep -rn "from rag\.search" rag/ app.py tests/ evals/ --include="*.py" | grep -v "tools"
grep -rn "from rag\.generator" rag/ app.py tests/ evals/ --include="*.py"
grep -rn "from rag\.router " rag/ app.py tests/ evals/ --include="*.py"
grep -rn "from rag\.reranker" rag/ app.py tests/ evals/ --include="*.py"
grep -rn "from rag\.gates" rag/ app.py tests/ evals/ --include="*.py"
```
Expected: no output (or only lines inside shim files themselves)

- [ ] **Step 2: Delete legacy top-level files**

```bash
git rm rag/search.py rag/search_v3.py rag/generator.py rag/router.py
git rm rag/reranker.py rag/gates.py rag/models_v3.py
git rm rag/llm_client.py rag/observability.py rag/context_builder.py rag/schedule.py
```

- [ ] **Step 3: Delete entire v4/ directory**

```bash
git rm -r rag/v4/
```

- [ ] **Step 4: Run tests**

```bash
make test
```
Expected: all tests pass

- [ ] **Step 5: Run lint**

```bash
make lint
```
Fix any import errors surfaced by ruff.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: delete legacy rag/ top-level modules and v4/ directory"
```

---

## Task 23: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
make test
```
Expected: all tests pass (or at worst same failure count as pre-refactor)

- [ ] **Step 2: Run typecheck**

```bash
make typecheck
```
Fix any mypy errors related to new import paths.

- [ ] **Step 3: Run lint**

```bash
make lint
```

- [ ] **Step 4: Update rag/CLAUDE.md**

Update the import paths in `rag/CLAUDE.md` to reflect new structure:
```markdown
## Public API

```python
from rag import run_pipeline, run_pipeline_with_memory, get_current_state
from rag import ChunkDoc, CourseDoc, PolicyDoc, VALID_CATEGORIES
from rag import RouterResult, PipelineState, ConversationState
from rag import get_langfuse, run_ragas_background, compute_ragas_metrics
```

## Module Locations (after refactor)

- LLM client: `rag/tools/llm.py` — `call_llm()`, `stream_llm()`
- Observability: `rag/tools/langfuse.py` — `get_langfuse()`, RAGAS
- MongoDB search: `rag/tools/mongo.py` — `hybrid_search()`, `semantic_search()`, etc.
- Voyage AI: `rag/tools/voyage.py` — `embed_query()`, `rerank()`
- State contracts: `rag/state/pipeline_state.py` — `PipelineState`, `RouterResult`
- Graph entry: `rag/graph/pipeline.py` — `run_pipeline()`, `run_pipeline_with_memory()`
```

- [ ] **Step 5: Run end-to-end probe (optional, requires API keys)**

```bash
make probe
```

- [ ] **Step 6: Final commit**

```bash
git add rag/CLAUDE.md tests/CLAUDE.md
git commit -m "docs: update CLAUDE.md import paths after rag/ refactor"
```

---

## Self-Review Notes

- **Spec coverage:** All spec sections have corresponding tasks. RouterResult in `state/` ✓. Tools layer ✓. No DI in nodes ✓. `v4/` deleted ✓. `__init__.py` shrunk ✓.
- **Type consistency:** `RouterResult` used consistently from `rag.state.pipeline_state` in all tasks. `PipelineState` from same location. `fetch_anchor_chunks` signature `(course_ids: list[str])` consistent across Task 6 (mongo.py) and Task 12 (anchor_node uses it).
- **Placeholder check:** `tools/mem0.py` in Task 7 references "Copy entire class body verbatim" — implementer must open `rag/v4/tamu_mem0_llm.py` and `rag/v4/voyage_mem0_embedder.py` and paste the full class bodies. This is intentional: those files are small (30 lines each) and straightforward copies.
- **Import ordering:** Tasks are ordered so each task only imports from modules created in previous tasks. Tools depend on nothing in `rag/` (only `config`). State depends on nothing. Graph depends on state + tools. Nodes depend on graph (middleware) + state + tools. Builder depends on nodes + edges + state.
