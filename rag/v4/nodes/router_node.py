"""Router node — classifies query and populates routing fields in state."""
from __future__ import annotations

from typing import Any

import config
from rag.v4.cache_utils import normalize_query
from rag.v4.exceptions import V4RouterError
from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.state import PipelineState


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

    # Cache check — skip LLM call on exact-match hit
    if config.SESSION_CACHE_ENABLED:
        cache_key = normalize_query(query)
        cached = state.get("router_cache", {}).get(cache_key)
        if cached:
            node_trace.append("router_cache_hit")
            return {
                "router_result": None,  # RouterResult object not available from cache dict
                "function": cached["function"],
                "course_ids": cached["course_ids"],
                "rewritten_query": cached.get("rewritten_query") or query,
                "intent_type": cached.get("intent_type"),
                "specific_categories": cached.get("specific_categories", []),
                "recurrent_search": cached.get("recurrent_search", False),
                "requires_retrieval": cached.get("requires_retrieval", True),
                "node_trace": node_trace,
            }

    try:
        router_result = registry.router_llm.classify(query, trace=trace)

        # Cache write — store serialized result for future identical queries
        router_cache_update = {}
        if config.SESSION_CACHE_ENABLED:
            cache_key = normalize_query(query)
            existing_cache = state.get("router_cache", {})
            router_cache_update = {
                **existing_cache,
                cache_key: {
                    "function": router_result.function,
                    "course_ids": router_result.course_ids,
                    "rewritten_query": router_result.rewritten_query or query,
                    "intent_type": router_result.intent_type,
                    "specific_categories": router_result.specific_categories,
                    "recurrent_search": router_result.recurrent_search,
                    "requires_retrieval": router_result.requires_retrieval,
                }
            }

        return {
            "router_result": router_result,
            "function": router_result.function,
            "course_ids": router_result.course_ids,
            "rewritten_query": router_result.rewritten_query or query,
            "intent_type": router_result.intent_type,
            "specific_categories": router_result.specific_categories,
            "recurrent_search": router_result.recurrent_search,
            "requires_retrieval": router_result.requires_retrieval,
            "router_cache": router_cache_update,
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
