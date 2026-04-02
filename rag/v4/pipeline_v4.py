"""v4 pipeline entry point — same 5-tuple return signature as v3 run_pipeline()."""
from __future__ import annotations

from typing import Any, Optional

import rag.v4.trace_registry as _trace_registry
from rag.v4.graph import build_graph, build_graph_with_memory
from rag.v4.registry_factory import make_default_registry
from rag.v4.state import PipelineState

# Module-level compiled graph singleton (built lazily on first call)
_graph = None
_registry = None

# Module-level memory graph singleton (built lazily on first call)
_memory_graph = None
_memory_registry = None


def _get_graph():
    global _graph, _registry
    if _graph is None:
        _registry = make_default_registry()
        _graph = build_graph(_registry)
    return _graph


def run_pipeline_v4(
    query: str,
    trace=None,
    return_timing: bool = False,
) -> tuple:
    """Run the v4 pipeline.

    Args:
        query: User query string
        trace: Optional Langfuse trace
        return_timing: If True, returns 6-tuple with timing_ms dict appended.
                       Default False preserves backwards-compatible 5-tuple.

    Returns:
        (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids)
        or if return_timing=True:
        (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids, timing_ms)
    """
    initial_state: PipelineState = {
        "query": query,
        "trace": trace,
        "node_trace": [],
        "timing_ms": {},
        "conflicted_course_ids": [],
        "data_gaps": [],
        "data_integrity": True,
        "anchor_chunks": [],
        "discovery_chunks": [],
        "retrieved_chunks": [],
    }

    graph = _get_graph()
    result = graph.invoke(initial_state)

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


def run_pipeline_v4_with_memory(
    query: str,
    trace=None,
    thread_config: Optional[dict] = None,
) -> tuple[list[dict], Any, list[tuple[str, str]], bool, list[str], list[str]]:
    """Run v4 pipeline with conversation memory (Phase 5).

    Same 5-tuple return as run_pipeline_v4. thread_config enables LangGraph checkpointing.

    Note: `trace` is accepted for API compatibility but is NOT passed into initial_state.
    Langfuse trace objects are not picklable and would crash SqliteSaver at the first
    checkpoint. Observability for stateful sessions is handled at the app.py level via
    the outer trace context — per-node spans are not available on this path.

    Args:
        query: User query string
        trace: Accepted but intentionally ignored — not picklable, cannot be checkpointed
        thread_config: LangGraph thread config dict, e.g. {"configurable": {"thread_id": "..."}}

    Returns:
        (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids, answer_tokens)
    """
    global _memory_graph, _memory_registry
    if _memory_graph is None:
        from rag.v4.checkpointer import make_checkpointer
        _memory_registry = make_default_registry()
        checkpointer = make_checkpointer()
        _memory_graph = build_graph_with_memory(_memory_registry, checkpointer=checkpointer)

    # Extract session_id (= thread_id) for mem0 manager lookup in nodes
    session_id = ""
    if thread_config:
        session_id = thread_config.get("configurable", {}).get("thread_id", "")

    initial_state: dict = {
        "query": query,
        "session_id": session_id,  # needed by history nodes to look up Mem0Manager
        # trace intentionally excluded — not picklable, cannot be checkpointed
        "node_trace": [],
        "timing_ms": {},
        "conflicted_course_ids": [],
        "data_gaps": [],
        "data_integrity": True,
        "anchor_chunks": [],
        "discovery_chunks": [],
        "retrieved_chunks": [],
    }

    invoke_kwargs = {}
    if thread_config:
        invoke_kwargs["config"] = thread_config

    if session_id and trace is not None:
        _trace_registry.register(session_id, trace)
    try:
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
    """Read the current conversation state for a thread (for answer cache checks).

    Returns empty dict if no state exists or graph not yet initialized.
    """
    global _memory_graph
    if _memory_graph is None:
        return {}
    try:
        snapshot = _memory_graph.get_state(thread_config)
        return snapshot.values if snapshot and snapshot.values else {}
    except Exception:
        return {}
