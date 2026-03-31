# RAG Pipeline Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix double generation, conversation context in router/generator, Langfuse tracing in v4 memory path, and prompt verbosity.

**Architecture:** Task A (double generation) must run first and alone — it changes the pipeline contract. Tasks B, C, D, E are independent and can run in parallel after A. Each task has its own test file or extends an existing one.

**Tech Stack:** Python, LangGraph, Haystack, Streamlit, Langfuse REST client (MinimalLangfuseClient)

**Spec:** `docs/superpowers/specs/2026-03-31-rag-pipeline-fixes-design.md`

---

## File Map

| File | Task | Change |
|------|------|--------|
| `rag/v4/nodes/generator_node.py` | A, D | `answer_stream` → `list[str]`; fetch trace from registry |
| `rag/v4/nodes/out_of_scope_node.py` | A | `answer_stream` → `list[str]` |
| `rag/v4/graph.py` | A | Remove `_strip_non_serializable` wrapper |
| `rag/v4/pipeline_v4.py` | A, D | Return 6-tuple; call trace registry |
| `app.py` | A | Use `answer_tokens`; remove `generator_order` call |
| `rag/v4/trace_registry.py` | D | New file — thread-safe trace store keyed by session_id |
| `rag/v4/nodes/router_node.py` | B, D | Build `prior_context` from history; fetch trace from registry |
| `rag/v4/components/routers.py` | B | Forward `prior_context` to `classify_query` |
| `rag/router.py` | B | Add `prior_context` param to `classify_query` |
| `rag/v4/nodes/history_update_node.py` | B | Store `specific_categories` in `rr_summary` |
| `rag/v4/nodes/history_inject_node.py` | C | Write `history_context`; stop mangling `rewritten_query` |
| `rag/v4/state.py` | C | Add `history_context: str` to `ConversationState` |
| `rag/generator.py` | C, E | Add `history_context` param; use `COMPARISON_EXTRACTION_SYSTEM` |
| `rag/v4/components/generators.py` | C | Pass `history_context` from state; fetch trace from registry (consolidates Task D generators.py change) |
| `rag/prompts.py` | B, E | Update CONVERSATION CONTEXT; trim ROUTER_PROMPT; slim `_BASE_SYSTEM`; add `COMPARISON_EXTRACTION_SYSTEM` |
| `tests/test_v4_graph.py` | A | Test `answer_stream` is `list[str]` |
| `tests/test_v4_history_node.py` | C | Test `history_inject_node` writes `history_context` |
| `tests/test_v4_router_node.py` | B | Test `prior_context` extraction and forwarding |
| `tests/test_v4_observability.py` | D | Test trace registry |
| `tests/test_generator.py` | C, E | Test `history_context` block; test `COMPARISON_EXTRACTION_SYSTEM` |

---

## Task A: Eliminate Double Generation

**Files:**
- Modify: `rag/v4/nodes/generator_node.py`
- Modify: `rag/v4/nodes/out_of_scope_node.py`
- Modify: `rag/v4/graph.py`
- Modify: `rag/v4/pipeline_v4.py`
- Modify: `app.py`
- Modify: `tests/test_v4_graph.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_v4_graph.py`:

```python
def test_generator_node_answer_stream_is_list():
    """generator_node must return answer_stream as list[str], not a generator."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.generator_node import generator_node

    registry = MagicMock()
    registry.generator_llm.generate_stream.return_value = iter(["Hello ", "world"])

    state = {"query": "test", "node_trace": [], "timing_ms": {}}
    result = generator_node(state, registry=registry)

    assert isinstance(result["answer_stream"], list)
    assert result["answer_stream"] == ["Hello ", "world"]
    assert result["answer"] == "Hello world"


def test_out_of_scope_node_answer_stream_is_list():
    """out_of_scope_node must return answer_stream as list[str]."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.out_of_scope_node import out_of_scope_node

    state = {"query": "test", "node_trace": [], "timing_ms": {}}
    result = out_of_scope_node(state, registry=MagicMock())

    assert isinstance(result["answer_stream"], list)
    assert len(result["answer_stream"]) == 1
    assert "TamuBot" in result["answer_stream"][0]


def test_pipeline_with_memory_returns_six_tuple():
    """run_pipeline_v4_with_memory must return a 6-tuple with answer_tokens as last element."""
    from unittest.mock import MagicMock, patch
    from rag.v4.pipeline_v4 import run_pipeline_v4_with_memory

    mock_result = {
        "retrieved_chunks": [],
        "router_result": MagicMock(function="out_of_scope", course_ids=[], requires_retrieval=False),
        "data_gaps": [],
        "data_integrity": True,
        "conflicted_course_ids": [],
        "answer_stream": ["Howdy!"],
        "function": "out_of_scope",
    }

    with patch("rag.v4.pipeline_v4._memory_graph") as mock_graph:
        mock_graph.invoke.return_value = mock_result
        result = run_pipeline_v4_with_memory("hello", thread_config={"configurable": {"thread_id": "t1"}})

    assert len(result) == 6
    chunks, rr, gaps, integrity, conflicted, tokens = result
    assert isinstance(tokens, list)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_v4_graph.py::test_generator_node_answer_stream_is_list tests/test_v4_graph.py::test_out_of_scope_node_answer_stream_is_list tests/test_v4_graph.py::test_pipeline_with_memory_returns_six_tuple -v 2>&1 | tail -20
```

Expected: FAIL (generator returns generator object, not list; 5-tuple returned)

- [ ] **Step 3: Update `generator_node.py`**

Replace the full function body (keep decorators and imports):

```python
@timing_middleware
@error_guard_middleware
def generator_node(state: PipelineState, registry: Any) -> dict:
    """Generate the answer. Stores answer_stream as list[str] (picklable for LangGraph)."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("generator")

    try:
        stream = registry.generator_llm.generate_stream(state)
        tokens = list(stream)
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

- [ ] **Step 4: Update `out_of_scope_node.py`**

Replace the function body:

```python
@timing_middleware
@error_guard_middleware
def out_of_scope_node(state: PipelineState, registry: Any) -> dict:
    """Write canned response to state as list[str]. No LLM call."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("out_of_scope")
    return {
        "answer": _OOS_RESPONSE,
        "answer_stream": [_OOS_RESPONSE],
        "node_trace": node_trace,
    }
```

- [ ] **Step 5: Update `graph.py` — remove `_strip_non_serializable`**

In `build_graph_with_memory`, change these two lines:

```python
# Before:
graph.add_node("generator", _strip_non_serializable(_bind(generator_node)))
graph.add_node("out_of_scope", _strip_non_serializable(_bind(out_of_scope_node)))

# After:
graph.add_node("generator", _bind(generator_node))
graph.add_node("out_of_scope", _bind(out_of_scope_node))
```

Delete the `_strip_non_serializable` helper function entirely (lines 133–144).

- [ ] **Step 6: Update `pipeline_v4.py` — return 6-tuple**

In `run_pipeline_v4_with_memory`, replace the `return` statement:

```python
# Before:
    return (
        result.get("retrieved_chunks", []),
        result.get("router_result"),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        result.get("conflicted_course_ids", []),
    )

# After:
    return (
        result.get("retrieved_chunks", []),
        result.get("router_result"),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        result.get("conflicted_course_ids", []),
        result.get("answer_stream", []),   # list[str] tokens — picklable
    )
```

- [ ] **Step 7: Update `app.py` — use answer_tokens, remove `generator_order` call**

Find the MongoDB pipeline block. Make these changes:

```python
# Change the unpack (line ~222):
# Before:
source_docs, router_result, data_gaps, data_integrity, conflicted_ids = result
# After:
source_docs, router_result, data_gaps, data_integrity, conflicted_ids, answer_tokens = result
```

```python
# Change the spinner label (line ~219):
# Before:
with st.spinner("Routing query and retrieving information..."):
# After:
with st.spinner("Routing, retrieving, and generating..."):
```

```python
# Remove the generator_order call and replace with token replay.
# Delete these lines (approximately lines 229–255):
#   answer = ""
#   answer_placeholder = st.empty()
#   try:
#       logger.info("Starting generation (streaming)...")
#       stream = generator_order(...)
#       for token in stream:
#           ...
#   except Exception as e:
#       ...

# Replace with:
answer = ""
answer_placeholder = st.empty()
for token in answer_tokens:
    answer += token
    answer_placeholder.markdown(answer + "▌")
answer_placeholder.markdown(answer)
logger.info(f"Generation complete, answer length: {len(answer)}")
```

Also remove the import line `from rag.v3_legacy.pipeline import generator_order` — it is no longer used in the MongoDB path. (Keep the import only if the Vertex AI legacy path still needs it — check if `generator_order` appears anywhere else in `app.py`. If not, remove it.)

- [ ] **Step 8: Run tests**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_v4_graph.py -v 2>&1 | tail -30
```

Expected: all 3 new tests PASS, existing tests still pass.

Also run the full suite:

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/ -v 2>&1 | tail -40
```

Fix any failures from the `history_update_node` test that checks `answer_stream is None` — it should still pass since `history_update_node` explicitly sets `"answer_stream": None` in its return dict.

- [ ] **Step 9: Commit**

```bash
git add rag/v4/nodes/generator_node.py rag/v4/nodes/out_of_scope_node.py rag/v4/graph.py rag/v4/pipeline_v4.py app.py tests/test_v4_graph.py
git commit -m "fix: eliminate double generation — answer_stream as list[str], remove generator_order from app"
```

---

## Task B: Conversation Context in Router

**Depends on:** Task A (pipeline contract change)

**Files:**
- Modify: `rag/v4/nodes/history_update_node.py`
- Modify: `rag/router.py`
- Modify: `rag/v4/components/routers.py`
- Modify: `rag/v4/nodes/router_node.py`
- Modify: `rag/prompts.py` (CONVERSATION CONTEXT section only)
- Modify: `tests/test_v4_router_node.py`
- Modify: `tests/test_v4_history_node.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_v4_router_node.py`:

```python
def test_router_node_passes_prior_context_when_history_present():
    """router_node builds prior_context from history and passes it to classify."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.router_node import router_node
    from rag.router import RouterResult

    registry = MagicMock()
    registry.router_llm.classify.return_value = RouterResult(
        course_ids=["CSCE 638", "CSCE 670"],
        rewritten_query="compare schedule CSCE 638 CSCE 670",
        specific_categories=["SCHEDULE"],
        specific_only=True,
    )

    state = {
        "query": "compare it with CSCE 670",
        "history": [
            {"role": "user", "content": "what's the schedule for CSCE 638?"},
            {
                "role": "assistant",
                "content": "The schedule is MWF 9-10am.",
                "router_result": {
                    "function": "hybrid_course",
                    "course_ids": ["CSCE 638"],
                    "specific_categories": ["SCHEDULE"],
                },
            },
        ],
        "node_trace": [],
        "timing_ms": {},
    }

    router_node(state, registry=registry)

    call_kwargs = registry.router_llm.classify.call_args.kwargs
    assert "prior_context" in call_kwargs
    ctx = call_kwargs["prior_context"]
    assert ctx is not None
    assert "SCHEDULE" in ctx
    assert "CSCE 638" in ctx


def test_router_node_no_prior_context_when_history_empty():
    """router_node passes prior_context=None when history is empty."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.router_node import router_node
    from rag.router import RouterResult

    registry = MagicMock()
    registry.router_llm.classify.return_value = RouterResult(
        course_ids=[], rewritten_query="test"
    )

    state = {"query": "hello", "history": [], "node_trace": [], "timing_ms": {}}
    router_node(state, registry=registry)

    call_kwargs = registry.router_llm.classify.call_args.kwargs
    assert call_kwargs.get("prior_context") is None
```

Add to `tests/test_v4_history_node.py`:

```python
def test_history_update_stores_specific_categories_in_router_result():
    """history_update_node must include specific_categories in rr_summary."""
    from unittest.mock import MagicMock
    from rag.router import RouterResult
    from rag.v4.nodes.history_update_node import history_update_node

    rr = RouterResult(
        course_ids=["CSCE 638"],
        specific_categories=["SCHEDULE"],
        rewritten_query="schedule CSCE 638",
    )
    state = {
        "query": "schedule for CSCE 638?",
        "answer": "MWF 9-10am.",
        "history": [],
        "router_result": rr,
        "turn_number": 0,
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_update_node(state, registry=MagicMock())
    history = result["history"]
    assistant_msg = next(m for m in history if m["role"] == "assistant")
    assert assistant_msg["router_result"]["specific_categories"] == ["SCHEDULE"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_v4_router_node.py::test_router_node_passes_prior_context_when_history_present tests/test_v4_router_node.py::test_router_node_no_prior_context_when_history_empty tests/test_v4_history_node.py::test_history_update_stores_specific_categories_in_router_result -v 2>&1 | tail -20
```

Expected: FAIL

- [ ] **Step 3: Update `history_update_node.py` — store `specific_categories`**

Find the `rr_summary` block and add `specific_categories`:

```python
# Before:
rr_summary = {
    "function": router_result.function,
    "course_ids": router_result.course_ids,
}
# After:
rr_summary = {
    "function": router_result.function,
    "course_ids": router_result.course_ids,
    "specific_categories": router_result.specific_categories,
}
```

- [ ] **Step 4: Update `router.py` — add `prior_context` param**

In `classify_query`, add the parameter and update the hint logic:

```python
def classify_query(
    query: str,
    router_span=None,
    prior_course_ids: Optional[list[str]] = None,
    prior_context: Optional[str] = None,
) -> "RouterResult":
    if prior_context:
        hint = f"[Context: {prior_context}]\n"
        query = hint + query
    elif prior_course_ids:
        hint = f"[Context: previous turn mentioned courses: {', '.join(prior_course_ids)}]\n"
        query = hint + query
    prompt = ROUTER_PROMPT.format(query=query)
    # ... rest of function unchanged
```

- [ ] **Step 5: Update `routers.py` — forward `prior_context`**

In `LLMRouterComponent`, add `prior_context` to `run()` and `classify()`:

```python
@component.output_types(router_result=object)
def run(
    self,
    query: str,
    trace: Optional[Any] = None,
    prior_course_ids: Optional[list[str]] = None,
    prior_context: Optional[str] = None,
) -> dict:
    if self._llm_fn is not None:
        with _call_llm_lock:
            _original = _router_mod.call_llm
            try:
                _router_mod.call_llm = self._llm_fn
                router_result = classify_query(
                    query, router_span=trace,
                    prior_course_ids=prior_course_ids,
                    prior_context=prior_context,
                )
            finally:
                _router_mod.call_llm = _original
    else:
        router_result = classify_query(
            query, router_span=trace,
            prior_course_ids=prior_course_ids,
            prior_context=prior_context,
        )
    return {"router_result": router_result}

def classify(
    self,
    query: str,
    trace: Optional[Any] = None,
    prior_course_ids: Optional[list[str]] = None,
    prior_context: Optional[str] = None,
) -> Any:
    return self.run(
        query=query, trace=trace,
        prior_course_ids=prior_course_ids,
        prior_context=prior_context,
    )["router_result"]
```

- [ ] **Step 6: Update `router_node.py` — extract history context**

Add `_build_prior_context` helper and call it in `router_node`:

```python
from typing import Optional

def _build_prior_context(history: list) -> Optional[str]:
    """Build a context string from the last assistant turn for pronoun/category resolution."""
    if not history:
        return None

    prior_query = ""
    prior_course_ids: list[str] = []
    prior_categories: list[str] = []

    for msg in reversed(history):
        role = msg.get("role", "")
        if not prior_query and role == "user":
            prior_query = msg.get("content", "")[:150]
        if role == "assistant" and not prior_course_ids and not prior_categories:
            rr = msg.get("router_result") or {}
            prior_course_ids = rr.get("course_ids", [])
            prior_categories = rr.get("specific_categories", [])
        if prior_query and (prior_course_ids or prior_categories):
            break

    if not prior_query:
        return None

    parts = [f"previous query: \"{prior_query}\""]
    if prior_course_ids:
        parts.append(f"courses: {', '.join(prior_course_ids)}")
    if prior_categories:
        parts.append(f"categories: {', '.join(prior_categories)}")
    return ", ".join(parts)
```

In `router_node`, replace the `registry.router_llm.classify(query, trace=trace)` call:

```python
prior_context = _build_prior_context(state.get("history", []))

# ...cache check unchanged...

try:
    router_result = registry.router_llm.classify(
        query, prior_context=prior_context, trace=trace
    )
```

- [ ] **Step 7: Update `ROUTER_PROMPT` CONVERSATION CONTEXT section**

In `rag/prompts.py`, replace only the `CONVERSATION CONTEXT` block (lines 12–21):

```python
ROUTER_PROMPT = """\
You are a query parser for a Texas A&M University course assistant.
Extract structured variables from the user's question and emit JSON.

CONVERSATION CONTEXT
The query may begin with a [Context: ...] line containing prior turn information.
Use it to resolve pronouns and infer omitted categories from the previous turn.
Examples:
- Context "previous query: 'what's the schedule for CSCE 638?', courses: CSCE 638, categories: SCHEDULE",
  query "compare it with CSCE 670"
  → course_ids=["CSCE 638", "CSCE 670"], specific_categories=["SCHEDULE"], specific_only=true
- Context "courses: CSCE 670", query "which has more assignments"
  → course_ids=["CSCE 670"]

COURSE IDs
# ... rest of prompt unchanged from current ...
```

(Only replace the CONVERSATION CONTEXT section — do not change any other part of the prompt in this task. Prompt trimming is Task E.)

- [ ] **Step 8: Run tests**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_v4_router_node.py tests/test_v4_history_node.py -v 2>&1 | tail -30
```

Expected: all new tests PASS, existing tests still pass.

- [ ] **Step 9: Commit**

```bash
git add rag/v4/nodes/history_update_node.py rag/router.py rag/v4/components/routers.py rag/v4/nodes/router_node.py rag/prompts.py tests/test_v4_router_node.py tests/test_v4_history_node.py
git commit -m "feat: inject full prior-turn context into router for category-aware follow-up queries"
```

---

## Task C: Conversation Context in Generator

**Depends on:** Task A

**Files:**
- Modify: `rag/v4/state.py`
- Modify: `rag/v4/nodes/history_inject_node.py`
- Modify: `rag/generator.py`
- Modify: `rag/v4/components/generators.py`
- Modify: `tests/test_v4_history_node.py`
- Modify: `tests/test_generator.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_v4_history_node.py`:

```python
def test_history_inject_writes_history_context_not_rewritten_query():
    """history_inject_node stores context in history_context; rewritten_query stays clean."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.history_inject_node import history_inject_node

    original_query = "what are the prerequisites?"
    state = {
        "query": original_query,
        "rewritten_query": original_query,
        "history": [
            {"role": "user", "content": "Tell me about CSCE 638"},
            {"role": "assistant", "content": "CSCE 638 is a graduate ML course."},
        ],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())

    # rewritten_query must not be modified
    assert result.get("rewritten_query", original_query) == original_query

    # history_context must be set and contain prior turn content
    assert "history_context" in result
    assert "CSCE 638" in result["history_context"]


def test_history_inject_empty_history_no_history_context():
    """With no history, history_inject_node does not set history_context."""
    from unittest.mock import MagicMock
    from rag.v4.nodes.history_inject_node import history_inject_node

    state = {
        "query": "what is CSCE 221?",
        "rewritten_query": "what is CSCE 221?",
        "history": [],
        "node_trace": [],
        "timing_ms": {},
    }
    result = history_inject_node(state, registry=MagicMock())
    assert result.get("history_context", "") == ""
    assert result.get("rewritten_query", state["rewritten_query"]) == state["rewritten_query"]
```

Add to `tests/test_generator.py`:

```python
def test_generate_stream_includes_conversation_history_block():
    """generate_stream with history_context includes <conversation_history> XML block."""
    from unittest.mock import patch
    from rag.generator import generate_stream

    captured_messages = []

    def mock_stream_llm(messages, **kwargs):
        captured_messages.extend(messages)
        yield "Answer [Source 1]"

    chunks = [{"content": "Grading is 40% exams.", "course_id": "CSCE 638", "category": "GRADING"}]
    history_ctx = "User: What is CSCE 638?\nAssistant: It is a grad ML course."

    with patch("rag.generator.stream_llm", side_effect=mock_stream_llm):
        list(generate_stream(
            results=chunks,
            question="What is the grading?",
            function="hybrid_course",
            history_context=history_ctx,
        ))

    user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
    assert "<conversation_history>" in user_msg
    assert history_ctx in user_msg
    assert "Question: What is the grading?" in user_msg
    # conversation_history block must appear BEFORE the Question line
    assert user_msg.index("<conversation_history>") < user_msg.index("Question:")


def test_generate_stream_no_history_context_no_block():
    """generate_stream without history_context does not include <conversation_history> block."""
    from unittest.mock import patch
    from rag.generator import generate_stream

    captured_messages = []

    def mock_stream_llm(messages, **kwargs):
        captured_messages.extend(messages)
        yield "Answer [Source 1]"

    chunks = [{"content": "Grading is 40% exams.", "course_id": "CSCE 638", "category": "GRADING"}]

    with patch("rag.generator.stream_llm", side_effect=mock_stream_llm):
        list(generate_stream(
            results=chunks,
            question="What is the grading?",
            function="hybrid_course",
        ))

    user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
    assert "<conversation_history>" not in user_msg
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_v4_history_node.py::test_history_inject_writes_history_context_not_rewritten_query tests/test_v4_history_node.py::test_history_inject_empty_history_no_history_context tests/test_generator.py::test_generate_stream_includes_conversation_history_block tests/test_generator.py::test_generate_stream_no_history_context_no_block -v 2>&1 | tail -20
```

Expected: FAIL

- [ ] **Step 3: Add `history_context` to `state.py`**

In `ConversationState`:

```python
class ConversationState(PipelineState, total=False):
    session_id: str
    history: list[ConversationMessage]
    history_summary: str
    history_context: str          # formatted history block for generator (set by history_inject_node)
    turn_number: int
    router_cache: dict
    retrieval_cache: dict
    answer_cache: dict
```

- [ ] **Step 4: Rewrite `history_inject_node.py`**

Replace the entire file:

```python
"""History inject node — builds history_context for generator from last N turns.

Runs AFTER router. Writes to history_context state key (used by generator).
Does NOT modify rewritten_query — that stays clean for retrieval embedding.
"""
from __future__ import annotations

from typing import Any, Optional

import config
from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.state import PipelineState


@timing_middleware
@error_guard_middleware
def history_inject_node(state: PipelineState, registry: Any) -> dict:
    """Build history_context for the generator from mem0 or raw history."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("history_inject")

    current_query = state.get("rewritten_query") or state.get("query", "")
    history_context = _build_history_context(state, current_query)

    result: dict = {"node_trace": node_trace}
    if history_context:
        result["history_context"] = history_context
    return result


def _build_history_context(state: PipelineState, current_query: str) -> str:
    """Return a formatted history string for the generator, or '' if none available."""
    # Try mem0 semantic retrieval first
    if config.MEM0_ENABLED:
        session_id = state.get("session_id", "")
        if session_id:
            from rag.v4.mem0_registry import get as get_mem0_manager
            mem0_manager = get_mem0_manager(session_id)
            if mem0_manager is not None:
                ctx = mem0_manager.search_context(current_query, top_k=3)
                if ctx:
                    return ctx

    # Fall back to raw windowed history
    history = state.get("history", [])
    history_summary = state.get("history_summary", "") or ""

    if not history and not history_summary:
        return ""

    lines: list[str] = []
    if history_summary:
        lines.append(f"[Summary of earlier turns: {history_summary}]")

    recent = history[-6:]  # last 3 turns = 6 messages
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")[:200]
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)
```

- [ ] **Step 5: Update `generator.py` — add `history_context` param**

In `generate_stream`, add the parameter and update `user_message` construction:

```python
def generate_stream(
    results: list[dict],
    question: str,
    function: str = "semantic_general",
    course_ids: list[str] | None = None,
    intent_type: str | None = None,
    specific_categories: list[str] | None = None,
    specific_only: bool = False,
    data_gaps: list[tuple[str, str]] | None = None,
    data_integrity: bool = True,
    conflicted_course_ids: list[str] | None = None,
    trace=None,
    history_context: str | None = None,          # NEW
):
```

Update the `user_message` line (search for `user_message = f"{context_xml}\n\nQuestion: {question}"`):

```python
    if history_context:
        user_message = (
            f"{context_xml}\n\n"
            f"<conversation_history>\n{history_context}\n</conversation_history>\n\n"
            f"Question: {question}"
        )
    else:
        user_message = f"{context_xml}\n\nQuestion: {question}"
```

Apply the same change to `generate()` (the non-streaming variant) — same parameter, same `user_message` logic.

- [ ] **Step 6: Update `generators.py` — pass `history_context` and fetch trace from registry**

This step covers both the Task C change (history_context) and the Task D change (trace registry), keeping all `generators.py` edits in one place:

```python
    def generate_stream(self, state: Any) -> Iterator[str]:
        from rag.v4.trace_registry import get as _get_trace
        stream_fn = self._get_stream_fn()
        router_result = state.get("router_result")
        # Fetch trace from registry if not in state (v4 memory path excludes trace from state)
        trace = state.get("trace") or _get_trace(state.get("session_id", ""))
        return stream_fn(
            results=state.get("retrieved_chunks", []),
            question=state.get("rewritten_query") or state.get("query", ""),
            function=state.get("function", "semantic_general"),
            course_ids=state.get("course_ids", []),
            intent_type=state.get("intent_type"),
            specific_categories=state.get("specific_categories", []),
            specific_only=router_result.specific_only if router_result else False,
            data_gaps=state.get("data_gaps", []),
            data_integrity=state.get("data_integrity", True),
            conflicted_course_ids=state.get("conflicted_course_ids", []),
            trace=trace,
            history_context=state.get("history_context"),
        )
```

- [ ] **Step 7: Run tests**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_v4_history_node.py tests/test_generator.py -v 2>&1 | tail -30
```

Expected: all new tests PASS, existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add rag/v4/state.py rag/v4/nodes/history_inject_node.py rag/generator.py rag/v4/components/generators.py tests/test_v4_history_node.py tests/test_generator.py
git commit -m "feat: inject conversation history as <conversation_history> block in generator user message"
```

---

## Task D: Fix Langfuse Tracing in v4 Memory Path

**Depends on:** Task A

**Files:**
- Create: `rag/v4/trace_registry.py`
- Modify: `rag/v4/pipeline_v4.py`
- Modify: `rag/v4/nodes/router_node.py`
- Modify: `rag/v4/nodes/generator_node.py`
- Modify: `tests/test_v4_observability.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_v4_observability.py`:

```python
def test_trace_registry_register_and_get():
    from rag.v4.trace_registry import register, get, clear
    mock_trace = object()
    register("session-abc", mock_trace)
    assert get("session-abc") is mock_trace
    clear("session-abc")


def test_trace_registry_clear_removes_entry():
    from rag.v4.trace_registry import register, get, clear
    register("session-xyz", object())
    clear("session-xyz")
    assert get("session-xyz") is None


def test_trace_registry_empty_session_id_is_noop():
    from rag.v4.trace_registry import register, get
    register("", object())
    assert get("") is None


def test_trace_registry_unknown_session_returns_none():
    from rag.v4.trace_registry import get
    assert get("never-registered-session-99") is None


def test_pipeline_registers_trace_before_invoke():
    """run_pipeline_v4_with_memory registers trace in registry before invoking the graph."""
    from unittest.mock import MagicMock, patch
    import rag.v4.trace_registry as reg

    mock_trace = MagicMock()
    registered = {}

    original_register = reg.register
    def spy_register(session_id, trace):
        registered[session_id] = trace
        original_register(session_id, trace)

    mock_graph_result = {
        "retrieved_chunks": [],
        "router_result": MagicMock(function="out_of_scope", course_ids=[], requires_retrieval=False),
        "data_gaps": [],
        "data_integrity": True,
        "conflicted_course_ids": [],
        "answer_stream": [],
        "function": "out_of_scope",
    }

    with patch("rag.v4.pipeline_v4._memory_graph") as mock_graph, \
         patch.object(reg, "register", side_effect=spy_register):
        mock_graph.invoke.return_value = mock_graph_result
        from rag.v4.pipeline_v4 import run_pipeline_v4_with_memory
        run_pipeline_v4_with_memory(
            "hello",
            trace=mock_trace,
            thread_config={"configurable": {"thread_id": "t-trace-test"}},
        )

    assert "t-trace-test" in registered
    assert registered["t-trace-test"] is mock_trace
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_v4_observability.py::test_trace_registry_register_and_get tests/test_v4_observability.py::test_trace_registry_clear_removes_entry tests/test_v4_observability.py::test_trace_registry_empty_session_id_is_noop tests/test_v4_observability.py::test_trace_registry_unknown_session_returns_none tests/test_v4_observability.py::test_pipeline_registers_trace_before_invoke -v 2>&1 | tail -20
```

Expected: FAIL (`rag.v4.trace_registry` does not exist)

- [ ] **Step 3: Create `rag/v4/trace_registry.py`**

```python
"""Thread-safe trace registry for v4 pipeline.

Stores active Langfuse trace objects keyed by session_id. Used to pass
traces into graph nodes without including them in LangGraph state
(LFTrace objects are not picklable and cannot be checkpointed).
"""
from __future__ import annotations

import threading
from typing import Any, Optional

_active: dict[str, Any] = {}
_lock = threading.Lock()


def register(session_id: str, trace: Any) -> None:
    """Store a trace for the given session. No-op if session_id is empty."""
    if session_id:
        with _lock:
            _active[session_id] = trace


def get(session_id: str) -> Optional[Any]:
    """Retrieve trace for session_id, or None if not registered."""
    if not session_id:
        return None
    with _lock:
        return _active.get(session_id)


def clear(session_id: str) -> None:
    """Remove trace for session_id after pipeline completes."""
    with _lock:
        _active.pop(session_id, None)
```

- [ ] **Step 4: Update `pipeline_v4.py` — register/clear trace around invoke**

Add imports near the top:

```python
from rag.v4.trace_registry import clear as _clear_trace, register as _register_trace
```

In `run_pipeline_v4_with_memory`, add registry calls around `graph.invoke`:

```python
    if session_id and trace is not None:
        _register_trace(session_id, trace)
    try:
        result = _memory_graph.invoke(initial_state, **invoke_kwargs)
    finally:
        _clear_trace(session_id)
```

- [ ] **Step 5: Update `router_node.py` — fetch trace from registry**

Add import:

```python
from rag.v4.trace_registry import get as _get_trace
```

In `router_node`, replace `trace = state.get("trace")` with:

```python
    trace = _get_trace(state.get("session_id", ""))
```

- [ ] **Step 6: Update `generator_node.py` — fetch trace from registry**

Add import:

```python
from rag.v4.trace_registry import get as _get_trace
```

In `generator_node`, replace `state.get("trace")` (it is not used directly in generator_node itself — the trace is passed through `registry.generator_llm.generate_stream(state)` which reads `state.get("trace")`). The `generators.py` fix is handled in Task C Step 6. No further change to `generator_node.py` is needed here beyond the import being available for future use.

- [ ] **Step 7: Run tests**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_v4_observability.py -v 2>&1 | tail -20
```

Expected: all new tests PASS.

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: full suite passes.

- [ ] **Step 8: Commit**

```bash
git add rag/v4/trace_registry.py rag/v4/pipeline_v4.py rag/v4/nodes/router_node.py rag/v4/nodes/generator_node.py rag/v4/components/generators.py tests/test_v4_observability.py
git commit -m "fix: restore Langfuse tracing in v4 memory path via trace registry"
```

---

## Task E: Prompt Simplification

**Depends on:** Nothing (independent)

**Files:**
- Modify: `rag/prompts.py`
- Modify: `rag/generator.py`
- Modify: `tests/test_generator.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_generator.py`:

```python
def test_base_system_no_chain_of_thought_instruction():
    from rag.prompts import _BASE_SYSTEM
    assert "Before answering, identify which chunk" not in _BASE_SYSTEM


def test_comparison_extraction_system_exists_and_is_compact():
    from rag.prompts import COMPARISON_EXTRACTION_SYSTEM
    assert len(COMPARISON_EXTRACTION_SYSTEM) < 300
    assert "extract" in COMPARISON_EXTRACTION_SYSTEM.lower()


def test_router_prompt_has_all_required_output_fields():
    from rag.prompts import ROUTER_PROMPT
    for field in ["course_ids", "specific_categories", "specific_only",
                  "category_confidence", "intent_type", "recurrent_search", "rewritten_query"]:
        assert field in ROUTER_PROMPT, f"ROUTER_PROMPT missing field: {field}"


def test_generate_comparison_uses_comparison_extraction_system(monkeypatch):
    """generate_comparison passes COMPARISON_EXTRACTION_SYSTEM as the system prompt."""
    from unittest.mock import MagicMock
    from rag.prompts import COMPARISON_EXTRACTION_SYSTEM
    import rag.generator as gen_mod

    captured = []

    def mock_call_llm(messages, **kwargs):
        captured.extend(messages)
        result = MagicMock()
        result.text = '{"courses": []}'
        result.input_tokens = None
        result.output_tokens = None
        return result

    monkeypatch.setattr(gen_mod, "call_llm", mock_call_llm)
    monkeypatch.setattr("rag.search_v3.get_missing_sections", lambda cid: [])

    gen_mod.generate_comparison([], "compare CSCE 638 and CSCE 670", ["CSCE 638", "CSCE 670"])

    system_msg = next((m["content"] for m in captured if m["role"] == "system"), None)
    assert system_msg == COMPARISON_EXTRACTION_SYSTEM
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_generator.py::test_base_system_no_chain_of_thought_instruction tests/test_generator.py::test_comparison_extraction_system_exists_and_is_compact tests/test_generator.py::test_router_prompt_has_all_required_output_fields tests/test_generator.py::test_generate_comparison_uses_comparison_extraction_system -v 2>&1 | tail -20
```

Expected: FAIL

- [ ] **Step 3: Update `prompts.py` — trim `ROUTER_PROMPT`, slim `_BASE_SYSTEM`, add `COMPARISON_EXTRACTION_SYSTEM`**

Replace `ROUTER_PROMPT` with this trimmed version:

```python
ROUTER_PROMPT = """\
You are a query parser for a Texas A&M University course assistant.
Extract structured variables from the user's question and emit JSON.

CONVERSATION CONTEXT
The query may begin with a [Context: ...] line containing prior turn information.
Use it to resolve pronouns and infer omitted categories from the previous turn.
Examples:
- Context "previous query: 'what's the schedule for CSCE 638?', courses: CSCE 638, categories: SCHEDULE",
  query "compare it with CSCE 670"
  → course_ids=["CSCE 638", "CSCE 670"], specific_categories=["SCHEDULE"], specific_only=true
- Context "courses: CSCE 670", query "which has more assignments"
  → course_ids=["CSCE 670"]

COURSE IDs
Identify all course IDs mentioned. Normalize: uppercase department + space + number
("csce638" → "CSCE 638", "CSCE-670" → "CSCE 670").
Extract ONLY courses the student is directly asking about — not prereq background.
Example: "I got a B in MATH 151, can I take this course?" → course_ids=[]
If the question uses "this course"/"this class" with no named course ID, set course_ids=[].

CATEGORIES
Valid categories: COURSE_OVERVIEW, INSTRUCTOR, PREREQUISITES, LEARNING_OUTCOMES, MATERIALS,
GRADING, SCHEDULE, ATTENDANCE_AND_MAKEUP, AI_POLICY, UNIVERSITY_POLICIES, SUPPORT_SERVICES

- specific_categories: categories the question targets (or [] if none clearly targeted)
- specific_only: true if ONLY those categories are asked about; false for broad/general questions
- category_confidence: 0.0–1.0

Examples:
- "What is the grading breakdown for CSCE 638?" → specific_categories=["GRADING"], specific_only=true, 0.95
- "Tell me about CSCE 670" → specific_categories=[], specific_only=false, 1.0
- "Tell me about CSCE 638, especially the grading" → specific_categories=["GRADING"], specific_only=false, 0.85
- "Can I use ChatGPT in CSCE 638?" → specific_categories=["AI_POLICY"], specific_only=true, 0.95
- "What materials and grading does CSCE 638 require?" → specific_categories=["MATERIALS","GRADING"], specific_only=true, 0.9

INTENT TYPE
Set intent_type = non-null ONLY for TAMU academic questions that are evaluative, advisory,
or discovery queries with no specific course ID. Null for purely factual questions and
non-TAMU topics.

Valid values: "ACADEMIC" | "CAREER" | "DIFFICULTY" | "PLANNING" | "ADMINISTRATIVE" | "GENERAL" | null

Examples:
- "Compare the grading of CSCE 638 and CSCE 670" → null (factual comparison)
- "Is CSCE 638 harder than CSCE 670?" → "DIFFICULTY" (evaluative)
- "What is the TAMU academic integrity policy?" → "ACADEMIC" (discovery, no course_id)
- "If I don't access Perusall through Canvas, will my grades show up?" → "ADMINISTRATIVE"

RECURRENT SEARCH
Set recurrent_search = true ONLY when the user wants to discover unknown courses using
a named course as an anchor ("What should I take with CS 638?", "What follows CS 638?").
False when the question is about named courses only, or no course ID is mentioned.

QUERY REWRITING
Expand with synonyms for retrieval:
- "late work" → "attendance makeup deadline extensions late submission"
- "ChatGPT"/"AI tools" → "AI policy artificial intelligence generative AI tools"
- "prereqs" → "prerequisites required courses corequisites"
- "grade breakdown" → "grading policy grade distribution weight percentage"

Output ONLY a JSON object with these fields:
{{
  "course_ids": [],
  "section": null,
  "specific_categories": [],
  "specific_only": false,
  "category_confidence": 1.0,
  "intent_type": null,
  "recurrent_search": false,
  "rewritten_query": "..."
}}

Respond with ONLY valid JSON, no other text.

User question: {query}
"""
```

Replace `_BASE_SYSTEM` — remove rule 3, merge its "cannot find" instruction into rule 1, renumber:

```python
_BASE_SYSTEM = """\
You are TamuBot, an academic assistant for Texas A&M University.
You help students find information about courses, syllabi, policies, and schedules.

RULES:
1. Answer ONLY based on the provided <context>. Never invent information. \
If the context does not contain the answer, state \
"I cannot find that information in the provided context" and do NOT use training data.
2. Cite your sources using [Source N] notation matching the source numbers in the context.
3. Do NOT answer questions outside TAMU academics — politely decline.
4. Be concise but thorough. Use markdown formatting for readability.
5. When using markdown tables, do NOT pad cells with extra spaces. Keep columns compact.
"""
```

Add `COMPARISON_EXTRACTION_SYSTEM` after the `_BASE_SYSTEM` block:

```python
# Minimal system prompt for generate_comparison() — JSON extraction only.
# No Markdown table overlay (rendered in Python), no advisory overlay.
COMPARISON_EXTRACTION_SYSTEM = """\
You are a structured data extractor for Texas A&M University course comparisons.
Extract the requested fields accurately from the provided <context>. Do not invent information.
If a field is not found in the context, use an empty string.
"""
```

- [ ] **Step 4: Update `generator.py` — use `COMPARISON_EXTRACTION_SYSTEM` in `generate_comparison`**

Add `COMPARISON_EXTRACTION_SYSTEM` to the import at the top of `generator.py`:

```python
from rag.prompts import (
    _BASE_SYSTEM,
    _FUNCTION_PROMPTS,
    _FUNCTION_TEMPERATURES,
    _HYBRID_COURSE_COMBINED,
    _HYBRID_COURSE_DEFAULT,
    _HYBRID_COURSE_SPECIFIC,
    _SEMANTIC_TYPE_PROMPTS,
    COMPARISON_EXTRACTION_SYSTEM,   # NEW
    UNCERTAINTY_INJECTION,
)
```

In `generate_comparison`, replace the `build_system_prompt(...)` call and the `system_prompt` variable:

```python
# Before (3 lines):
    system_prompt = build_system_prompt(
        function="hybrid_course",
        course_ids=course_ids,
        intent_type="GENERAL",
        specific_categories=[],
        specific_only=False,
    )

# After (1 line):
    system_prompt = COMPARISON_EXTRACTION_SYSTEM
```

- [ ] **Step 5: Run tests**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/test_generator.py -v 2>&1 | tail -30
```

Expected: all 4 new tests PASS, existing tests still pass.

Full suite:

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/ -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add rag/prompts.py rag/generator.py tests/test_generator.py
git commit -m "refactor: trim router prompt ~30%, slim _BASE_SYSTEM, minimal comparison extraction prompt"
```

---

## Final Verification

After all tasks complete:

- [ ] **Run full test suite**

```bash
cd /home/artem/dev/TAMUBOT && python -m pytest tests/ -v 2>&1 | tail -40
```

Expected: all tests pass.

- [ ] **Smoke test via make**

```bash
cd /home/artem/dev/TAMUBOT && make lint
```

- [ ] **Manual probe** (optional — costs tokens)

Start the app and send a 2-turn conversation:
1. "What is the schedule for CSCE 638?"
2. "Compare it with CSCE 670"

Verify in Langfuse that `TamuBot_Complete_Pipeline` shows Router_Stage and Generator_Stage spans. Verify the second query comparison is scoped to SCHEDULE only.
