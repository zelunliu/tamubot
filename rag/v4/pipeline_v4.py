"""v4 pipeline entry point — same 5-tuple return signature as v3 run_pipeline()."""
from __future__ import annotations
import functools
from typing import Any, Iterator, Optional

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
) -> tuple[list[dict], Any, list[tuple[str, str]], bool, list[str]]:
    """Run the v4 pipeline. Returns same 5-tuple as v3 run_pipeline():

    (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids)

    timing_ms is tracked internally in state but not exposed to callers,
    keeping the return signature identical to v3 run_pipeline().
    """
    import time

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

    t_start = time.perf_counter()
    graph = _get_graph()
    result = graph.invoke(initial_state)
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # Store timing internally (available via state) but return v3-compatible 5-tuple
    _ = {**result.get("timing_ms", {}), "total_ms": elapsed_ms}

    return (
        result.get("retrieved_chunks", []),
        result.get("router_result"),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        result.get("conflicted_course_ids", []),
    )


def run_pipeline_v4_with_memory(
    query: str,
    trace=None,
    thread_config: Optional[dict] = None,
) -> tuple[list[dict], Any, list[tuple[str, str]], bool, list[str]]:
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
        (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids)
    """
    global _memory_graph, _memory_registry
    if _memory_graph is None:
        from rag.v4.checkpointer import make_checkpointer
        _memory_registry = make_default_registry()
        checkpointer = make_checkpointer()
        _memory_graph = build_graph_with_memory(_memory_registry, checkpointer=checkpointer)

    initial_state: dict = {
        "query": query,
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

    result = _memory_graph.invoke(initial_state, **invoke_kwargs)

    return (
        result.get("retrieved_chunks", []),
        result.get("router_result"),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        result.get("conflicted_course_ids", []),
    )
