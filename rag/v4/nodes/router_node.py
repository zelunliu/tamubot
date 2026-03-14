"""Router node — classifies query and populates routing fields in state."""
from __future__ import annotations
from typing import Any
from rag.v4.state import PipelineState
from rag.v4.exceptions import V4RouterError
from rag.v4.middleware import error_guard_middleware, timing_middleware


@timing_middleware
@error_guard_middleware
def router_node(state: PipelineState, registry: Any) -> dict:
    """Classify query using registry.router_llm, write all RouterResult fields to state.

    Falls back to out_of_scope on any error.
    """
    query = state.get("query", "")
    trace = state.get("trace")
    node_trace = list(state.get("node_trace", []))
    node_trace.append("router")

    try:
        router_result = registry.router_llm.classify(query, trace=trace)
        return {
            "router_result": router_result,
            "function": router_result.function,
            "course_ids": router_result.course_ids,
            "rewritten_query": router_result.rewritten_query or query,
            "intent_type": router_result.intent_type,
            "specific_categories": router_result.specific_categories,
            "recurrent_search": router_result.recurrent_search,
            "requires_retrieval": router_result.requires_retrieval,
            "node_trace": node_trace,
        }
    except Exception as e:
        return {
            "function": "out_of_scope",
            "course_ids": [],
            "rewritten_query": query,
            "requires_retrieval": False,
            "error": f"Router failed: {e}",
            "node_trace": node_trace,
        }
