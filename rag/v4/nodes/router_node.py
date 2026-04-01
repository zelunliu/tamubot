"""Router node — classifies query and populates routing fields in state."""
from __future__ import annotations

from typing import Any, Optional

import config
from rag.v4.cache_utils import normalize_query
from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.state import PipelineState


def _build_prior_context(history: list) -> Optional[str]:
    """Build a context string from the last assistant turn for pronoun/category resolution."""
    if not history:
        return None

    prior_query = ""
    prior_course_ids: list[str] = []
    prior_categories: list[str] = []

    for msg in reversed(history):
        role = msg.get("role", "")
        if not prior_query and role == "user":
            prior_query = msg.get("content", "")[:150]
        if role == "assistant" and not prior_course_ids and not prior_categories:
            rr = msg.get("router_result") or {}
            prior_course_ids = rr.get("course_ids", [])
            prior_categories = rr.get("specific_categories", [])
        if prior_query and (prior_course_ids or prior_categories):
            break

    if not prior_query:
        return None

    parts = [f"previous query: \"{prior_query}\""]
    if prior_course_ids:
        parts.append(f"courses: {', '.join(prior_course_ids)}")
    if prior_categories:
        parts.append(f"categories: {', '.join(prior_categories)}")
    return ", ".join(parts)


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

    prior_context = _build_prior_context(state.get("history", []))

    try:
        router_result = registry.router_llm.classify(query, trace=trace, prior_context=prior_context)

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
