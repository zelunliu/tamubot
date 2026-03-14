"""v4 pipeline entry point — same 6-tuple return signature as v3 run_pipeline()."""
from __future__ import annotations
import functools
from typing import Any, Iterator, Optional

from rag.v4.graph import build_graph
from rag.v4.registry_factory import make_v3_registry
from rag.v4.state import PipelineState

# Module-level compiled graph singleton (built lazily on first call)
_graph = None
_registry = None


def _get_graph():
    global _graph, _registry
    if _graph is None:
        _registry = make_v3_registry()
        _graph = build_graph(_registry)
    return _graph


def run_pipeline_v4(
    query: str,
    trace=None,
) -> tuple[list[dict], Any, list[tuple[str, str]], bool, list[str], dict]:
    """Run the v4 pipeline. Returns same 6-tuple as v3 run_pipeline():

    (chunks, router_result, data_gaps, data_integrity, conflicted_course_ids, timing_ms)
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

    return (
        result.get("retrieved_chunks", []),
        result.get("router_result"),
        result.get("data_gaps", []),
        result.get("data_integrity", True),
        result.get("conflicted_course_ids", []),
        {**result.get("timing_ms", {}), "total_ms": elapsed_ms},
    )
