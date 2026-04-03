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
