"""Middleware decorators for v4 nodes.

@tracing_middleware  — creates a per-node Langfuse span (outermost, reads timing from inner)
@timing_middleware   — populates state["timing_ms"][node_name] after each node
@error_guard_middleware — catches V4PipelineError, writes state["error"], graph continues

Decorator order on every node must be:
    @tracing_middleware
    @timing_middleware
    @error_guard_middleware
    def node_fn(state, registry): ...
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable

import rag.v4.trace_registry as _trace_reg
from rag.v4.exceptions import V4PipelineError

# Short display labels for Langfuse span names (node_fn.__name__ → "node.{label}")
_DISPLAY: dict[str, str] = {
    "router_node": "router",
    "history_inject_node": "history_inject",
    "anchor_node": "anchor",
    "eval_search_node": "eval_search",
    "retrieval_node": "retrieval",
    "schedule_filter_node": "schedule_filter",
    "merge_node": "merge",
    "generator_node": "generator",
    "history_update_node": "history_update",
    "out_of_scope_node": "out_of_scope",
}


def _node_span_input(state: Any, label: str) -> dict:
    """Extract Langfuse-friendly input metadata for the given node label."""
    if label == "router":
        return {
            "query": (state.get("query") or "")[:200],
            "history_count": len(state.get("history", [])),
        }
    if label == "history_inject":
        recent = [
            {"role": m.get("role"), "content": (m.get("content") or "")[:200]}
            for m in state.get("history", [])[-4:]
        ]
        return {
            "turn_number": state.get("turn_number", 0),
            "history_count": len(state.get("history", [])),
            "has_summary": bool(state.get("history_summary")),
            "history_summary_preview": (state.get("history_summary") or "")[:300],
            "recent_turns": recent,
        }
    if label == "anchor":
        return {
            "course_ids": state.get("course_ids", []),
            "specific_categories": state.get("specific_categories", []),
        }
    if label == "eval_search":
        return {
            "query": (state.get("rewritten_query") or state.get("query") or "")[:200],
            "anchor_chunk_count": len(state.get("anchor_chunks", [])),
        }
    if label == "retrieval":
        return {
            "function": state.get("function"),
            "course_ids": state.get("course_ids", []),
            "query": (state.get("rewritten_query") or state.get("query") or "")[:200],
            "eval_query": (state.get("eval_query") or "")[:200],
        }
    if label == "schedule_filter":
        return {
            "course_ids": state.get("course_ids", []),
            "discovery_chunk_count": len(state.get("discovery_chunks", [])),
        }
    if label == "merge":
        return {
            "anchor_chunk_count": len(state.get("anchor_chunks", [])),
            "discovery_chunk_count": len(state.get("discovery_chunks", [])),
        }
    if label == "generator":
        return {
            "function": state.get("function"),
            "course_ids": state.get("course_ids", []),
            "chunk_count": len(state.get("retrieved_chunks", [])),
            "has_history_context": bool(state.get("history_context")),
            "history_context": (state.get("history_context") or "")[:500],
        }
    if label == "history_update":
        return {
            "query": (state.get("query") or "")[:200],
            "answer_preview": (state.get("answer") or "")[:300],
            "turn_number": state.get("turn_number", 0),
        }
    if label == "out_of_scope":
        return {"query": (state.get("query") or "")[:200]}
    return {}


def _node_span_output(result: dict, label: str) -> dict:
    """Extract Langfuse-friendly output metadata from the node result dict."""
    node_trace = result.get("node_trace", [])
    if label == "router":
        return {
            "function": result.get("function"),
            "course_ids": result.get("course_ids", []),
            "intent_type": result.get("intent_type"),
            "rewritten_query": (result.get("rewritten_query") or "")[:200],
            "specific_categories": result.get("specific_categories", []),
            "cache_hit": "router_cache_hit" in node_trace,
        }
    if label == "history_inject":
        hc = result.get("history_context") or ""
        return {
            "history_context_len": len(hc),
            "history_context": hc[:500],
        }
    if label == "anchor":
        return {
            "anchor_chunk_count": len(result.get("anchor_chunks", [])),
            "data_gaps": result.get("data_gaps", []),
            "data_integrity": result.get("data_integrity"),
        }
    if label == "eval_search":
        return {"eval_query": (result.get("eval_query") or "")[:200]}
    if label == "retrieval":
        n = len(result.get("retrieved_chunks") or []) + len(result.get("discovery_chunks") or [])
        return {
            "chunk_count": n,
            "cache_hit": "retrieval_cache_hit" in node_trace,
        }
    if label == "schedule_filter":
        return {
            "discovery_chunk_count": len(result.get("discovery_chunks", [])),
            "conflicted_course_ids": result.get("conflicted_course_ids", []),
        }
    if label == "merge":
        return {"retrieved_chunk_count": len(result.get("retrieved_chunks", []))}
    if label == "generator":
        return {"answer_preview": (result.get("answer") or "")[:300]}
    if label == "history_update":
        return {
            "turn_number": result.get("turn_number"),
            "history_count": len(result.get("history", [])),
            "compressed": result.get("history_compressed", False),
        }
    if label == "out_of_scope":
        return {"responded": True}
    return {}


def tracing_middleware(node_fn: Callable) -> Callable:
    """Outermost decorator: creates a per-node LFSpan child of the root trace.

    Must be the outermost decorator so timing_middleware runs inside and its
    result["timing_ms"][node_name] is available when the span is closed.

    The node span is pushed onto the thread-local span stack for the duration
    of the node call so child components (VoyageReranker, MongoDocumentStore)
    can retrieve it via trace_registry.current_span() without signature changes.
    """
    node_name = node_fn.__name__
    label = _DISPLAY.get(node_name, node_name)

    @functools.wraps(node_fn)
    def wrapper(state: Any, **kwargs) -> dict:
        session_id = state.get("session_id", "")
        parent = (
            _trace_reg.current_span()
            or _trace_reg.get(session_id)
            or state.get("trace")
        )

        span = None
        if parent is not None:
            try:
                span = parent.span(
                    name=f"node.{label}",
                    input=_node_span_input(state, label),
                )
            except Exception:
                span = None

        if span is not None:
            _trace_reg.push_span(span)

        try:
            result = node_fn(state, **kwargs)
        except Exception:
            # Unhandled exception (not caught by error_guard_middleware)
            if span is not None:
                _trace_reg.pop_span()
                try:
                    span.end(metadata={"error": "unhandled_exception"})
                except Exception:
                    pass
            raise

        # Normal exit: end span with output + timing, then pop
        if span is not None:
            elapsed_ms = result.get("timing_ms", {}).get(node_name)
            error = result.get("error")
            try:
                span.end(
                    output=_node_span_output(result, label),
                    metadata={
                        "elapsed_ms": elapsed_ms,
                        **({"error": error} if error else {}),
                    },
                )
            except Exception:
                pass
            _trace_reg.pop_span()

        return result

    return wrapper


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
