"""Middleware decorators for v4 nodes.

@timing_middleware   — populates state["timing_ms"][node_name] after each node
@error_guard_middleware — catches V4PipelineError, writes state["error"], graph continues
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable

from rag.v4.exceptions import V4PipelineError


def timing_middleware(node_fn: Callable) -> Callable:
    """Decorator: records elapsed time for node_fn in state["timing_ms"][node_name]."""
    node_name = node_fn.__name__

    @functools.wraps(node_fn)
    def wrapper(state: Any, **kwargs) -> dict:
        t_start = time.perf_counter()
        result = node_fn(state, **kwargs)
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

        # Merge timing into existing timing_ms dict
        existing_timing = dict(state.get("timing_ms", {}))
        existing_timing[node_name] = elapsed_ms
        result["timing_ms"] = existing_timing
        return result

    return wrapper


def error_guard_middleware(node_fn: Callable) -> Callable:
    """Decorator: catches V4PipelineError and writes state["error"] instead of crashing.

    The graph always reaches END even on partial failure.
    node_trace from state is preserved in the error dict so the graph trace
    remains consistent.
    Non-V4PipelineError exceptions are re-raised (unexpected errors should surface).
    """
    @functools.wraps(node_fn)
    def wrapper(state: Any, **kwargs) -> dict:
        try:
            return node_fn(state, **kwargs)
        except V4PipelineError as e:
            return {
                "error": f"{node_fn.__name__} failed: {e}",
                "node_trace": list(state.get("node_trace", [])),
            }

    return wrapper
