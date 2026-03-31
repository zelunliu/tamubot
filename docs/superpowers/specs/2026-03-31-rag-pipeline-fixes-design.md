# RAG Pipeline Fixes — Design Spec
**Date:** 2026-03-31
**Branch:** feature/lightrag-top-level (applies to main RAG pipeline, not LightRAG spike)

---

## Problem Summary

Five interconnected issues found in the v4 memory pipeline:

| # | Issue | Root Cause |
|---|-------|------------|
| 1 | Comparison mode too slow / too many tokens | Double generation on every query; verbose comparison system prompt with irrelevant overlays |
| 2 | Router and generator prompts too verbose | `ROUTER_PROMPT` ~140 lines with redundant examples; `_BASE_SYSTEM` has chain-of-thought instruction |
| 3 | Tracing only visible in probe mode | `run_pipeline_v4_with_memory()` excludes `trace` from state (not picklable); all nodes see `None` |
| 4 | Session memory not working | App.py streams from a second `generator_order()` call with raw query (no history context); router gets no prior category context; history format mangles `rewritten_query` |
| 5 | Execution strategy | These fixes are independent enough for separate subagent tasks |

---

## Root Cause: Double Generation

`run_pipeline_v4_with_memory()` runs the full v4 graph **including `generator_node`** (LLM call #1). App.py then also calls `generator_order()` from `rag.v3_legacy.pipeline` (LLM call #2). The user-visible streamed answer comes from call #2, which uses the raw `prompt` with no history context. History is updated from call #1.

Consequence:
- Non-recurrent queries: 1 router + 2 generator LLM calls (intended: 1+1)
- Recurrent queries: 1 router + 1 EvalSearch + 2 generator calls (intended: 1+1+1)
- Comparison queries: `generate_comparison()` runs twice (2× tokens, 2× latency)
- Session memory enrichment from `history_inject_node` never reaches the user-visible answer

---

## Fix 1 — Eliminate Double Generation (Option B)

**Decision:** Keep `generator_node` in v4 graph. Remove `generator_order()` call from app.py.

### Changes

**`rag/v4/nodes/generator_node.py`**
- Return `answer_stream` as `list[str]` (tokens collected by `list(stream)`) instead of a generator object
- `list[str]` is picklable → LangGraph can checkpoint it
- Remove `_replay()` generator function

**`rag/v4/graph.py`**
- Remove `_strip_non_serializable` wrapper from `generator_node` and `out_of_scope_node`
- `answer_stream` is now picklable; `trace` is never in state so no stripping needed

**`rag/v4/pipeline_v4.py`**
- `run_pipeline_v4_with_memory()` returns 6-tuple: add `answer_tokens: list[str]` as last element
- `run_pipeline_v4()` (no-memory path) is unchanged — still returns 5-tuple

**`app.py`**
- Use `answer_tokens` from graph result to drive UI display (iterate over list)
- Remove `generator_order()` call from the memory path
- Keep `from rag.v3_legacy.pipeline import generator_order` only for non-memory path (Vertex AI legacy)

### Token/latency impact
- Comparison: `generate_comparison()` runs once instead of twice
- Recurrent: EvalSearch + generator once each (as designed)
- Non-recurrent: router + generator once each (as designed)

### Streaming trade-off
`generator_node` already calls `list(stream)` (blocking). Answer appears in rapid replay from collected token list. True token-by-token streaming from LLM to UI is lost on the memory path. Accepted trade-off.

---

## Fix 2 — Conversation Context in Router

**Decision:** Always inject prior turn context into the router hint (not just course IDs).

### What the router needs
Example: Turn 1 = "what's the schedule for CSCE 638?", Turn 2 = "compare it with CSCE 670"
- Without history: router extracts `specific_categories=[]` → full comparison retrieved
- With history: router extracts `specific_categories=["SCHEDULE"]`, `specific_only=True` → schedule-only comparison

### Changes

**`rag/router.py` — `classify_query()`**
- Add `prior_context: str | None = None` parameter
- When set, prepend `[Context: {prior_context}]` to query before formatting the prompt
- Replaces the existing `prior_course_ids` parameter (which becomes a subset of prior_context)

**`rag/v4/nodes/router_node.py`**
- Read `history` from state (ConversationState has it)
- Extract from last assistant message's `router_result` summary: `course_ids`, `specific_categories`
- Extract last user message content (capped at 150 chars)
- Build context string: `"previous query: '{query}', courses: {ids}, categories: {cats}"`
- Pass to `registry.router_llm.classify(query, prior_context=prior_context, trace=trace)`

**`rag/v4/components/routers.py` — `LLMRouterComponent.classify()`**
- Accept and forward `prior_context` to `classify_query()`

**`rag/prompts.py` — `ROUTER_PROMPT`**
- Update `CONVERSATION CONTEXT` section to document the richer context format
- Remove the current `prior_course_ids`-only examples; replace with full context examples

---

## Fix 3 — Conversation Context in Generator

**Decision:** Separate history context from `rewritten_query`. Inject as a labeled XML block in the generator's user message.

### Changes

**`rag/v4/nodes/history_inject_node.py`**
- Stop prepending context to `rewritten_query`
- Instead, write mem0/history context to new state key `history_context: str`
- `rewritten_query` stays clean (router's retrieval query only)

**`rag/v4/state.py`**
- Add `history_context: str` field to `ConversationState`

**`rag/generator.py` — `generate_stream()` and `generate()`**
- Add `history_context: str | None = None` parameter
- When set, insert before the question in `user_message`:
  ```
  <conversation_history>
  {history_context}
  </conversation_history>

  Question: {question}
  ```

**`rag/v4/components/generators.py` — `LLMGeneratorComponent.generate_stream()`**
- Pass `history_context=state.get("history_context")` to `generate_stream()`

### Why not mangle `rewritten_query`
`rewritten_query` is used for retrieval vector search. Prepending `[Previous context:...]` to it degrades embedding quality. The generator's `user_message` is the right place for conversational context.

---

## Fix 4 — Tracing in v4 Memory Path

**Problem:** `trace` (LFTrace object) is not picklable → excluded from `initial_state` in `run_pipeline_v4_with_memory()` → all nodes see `state.get("trace") == None` → no Router_Stage, Generator_Stage, etc. in Langfuse for the memory path.

### Solution: Module-level trace registry

**`rag/v4/pipeline_v4.py`**
```python
_active_traces: dict[str, Any] = {}

def register_trace(session_id: str, trace: Any) -> None: ...
def get_active_trace(session_id: str) -> Any: ...
def clear_trace(session_id: str) -> None: ...
```
- `run_pipeline_v4_with_memory()` calls `register_trace(session_id, trace)` before `graph.invoke()`
- Clears in a `finally` block after invoke
- Thread-safe: keyed by session_id (unique per user session)

**`rag/v4/nodes/router_node.py`** and **`rag/v4/nodes/generator_node.py`**
- Replace `trace = state.get("trace")` with `trace = get_active_trace(state.get("session_id", ""))`
- Other nodes (retrieval, anchor, etc.) that already pass `parent_span` can similarly fetch trace

**Result:** Full Router_Stage + Generator_Stage (+ Comparison_Extraction / EvalSearch_Stage for recurrent) appear in Langfuse under `TamuBot_Complete_Pipeline` for the memory path, matching probe mode.

---

## Fix 5 — Prompt Simplification

### 5a. `ROUTER_PROMPT` (~30% trim)
- Condense `COURSE IDs` section: remove 4 of 6 examples, keep the 2 most distinctive
- Condense `CATEGORIES` section: collapse field descriptions; keep one example per ambiguous case
- Condense `INTENT TYPE` section: collapse valid values list inline; remove 3 of 7 examples
- Condense `RECURRENT SEARCH` section: remove 2 redundant examples
- Keep all logic and JSON output format exactly as-is — only removing redundant examples

### 5b. `_BASE_SYSTEM` (generator)
- Remove rule 3: "Before answering, identify which chunk contains the answer." — this is a chain-of-thought instruction that burns output tokens without improving accuracy; the citation requirement in rule 2 already handles grounding
- Keep rules 1, 2, 4, 5, 6

### 5c. `generate_comparison()` system prompt
- Replace `build_system_prompt("hybrid_course", ..., intent_type="GENERAL")` with a dedicated `COMPARISON_EXTRACTION_SYSTEM` constant
- Content: minimal JSON extraction instruction only — no Markdown table overlay (rendered in Python), no advisory overlay, no uncertainty injection
- Estimated: reduces comparison system prompt from ~600 chars to ~150 chars

---

## Fix 6 — Subagent Execution Strategy

Five independent tasks, suggested order (each maps to different files):

| Task | Files | Dependency |
|------|-------|------------|
| A: Eliminate double generation | `generator_node.py`, `graph.py`, `pipeline_v4.py`, `app.py` | First — unblocks accurate token count |
| B: Router conversation context | `router_node.py`, `routers.py`, `router.py`, `prompts.py` | After A |
| C: Generator conversation context | `history_inject_node.py`, `state.py`, `generator.py`, `generators.py` | After A |
| D: Tracing fix | `pipeline_v4.py`, `router_node.py`, `generator_node.py` | After A |
| E: Prompt simplification | `prompts.py`, `generator.py` | Independent |

Tasks B, C, D, E can run in parallel after A completes.

---

## Out of Scope
- LightRAG spike (`lightrag/`) — separate branch concern
- RAGAS evaluation pipeline — no changes
- Vertex AI legacy path — no changes
- `generate()` (non-streaming) — updated only where it shares code with `generate_stream()`
