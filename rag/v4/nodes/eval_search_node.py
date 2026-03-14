"""Eval search node — generates the discovery search string for the recurrent pass."""
from __future__ import annotations
from typing import Any
from rag.v4.state import PipelineState


def eval_search_node(state: PipelineState, registry: Any) -> dict:
    """Generate eval query from anchor chunks. Only runs for recurrent path."""
    query = state.get("rewritten_query") or state.get("query", "")
    anchor_chunks = state.get("anchor_chunks", [])
    trace = state.get("trace")
    node_trace = list(state.get("node_trace", []))
    node_trace.append("eval_search")

    try:
        eval_query = registry.generator_llm.generate_eval_query(query, anchor_chunks, trace=trace)
        return {"eval_query": eval_query, "node_trace": node_trace}
    except Exception as e:
        # Fall back to rewritten query
        return {"eval_query": query, "error": f"Eval search failed: {e}", "node_trace": node_trace}
