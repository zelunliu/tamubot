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
        # Fall back to rewritten query
        return {"eval_query": query, "error": f"Eval search failed: {e}", "node_trace": node_trace}
