"""History inject node — prepends last N turns to rewritten_query.

Runs AFTER router (preserves per-turn function classification accuracy).
Only modifies rewritten_query if history exists.
"""
from __future__ import annotations

from typing import Any

import config
from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.state import PipelineState


@timing_middleware
@error_guard_middleware
def history_inject_node(state: PipelineState, registry: Any) -> dict:
    """Prepend last 3 turns of history to rewritten_query."""
    history = state.get("history", [])
    rewritten_query = state.get("rewritten_query") or state.get("query", "")
    node_trace = list(state.get("node_trace", []))
    node_trace.append("history_inject")

    # Try mem0 semantic context first (if enabled)
    if config.MEM0_ENABLED:
        session_id = state.get("session_id", "")
        if session_id:
            from rag.v4.mem0_registry import get as get_mem0_manager
            mem0_manager = get_mem0_manager(session_id)
            if mem0_manager is not None:
                mem0_context = mem0_manager.search_context(rewritten_query, top_k=3)
                if mem0_context:
                    enriched_query = f"[Context:\n{mem0_context}\n]\n\nCurrent question: {rewritten_query}"
                    return {"rewritten_query": enriched_query, "node_trace": node_trace}
    # Fall through to existing raw history logic (if mem0 returned nothing)

    if not history:
        return {"node_trace": node_trace}

    # Take last 3 turns (6 messages: 3 user + 3 assistant)
    recent = history[-6:]  # last 3 turns = last 6 messages
    context_lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            context_lines.append(f"{role.capitalize()}: {content[:200]}")

    if context_lines:
        context_str = "\n".join(context_lines)
        enriched_query = f"[Previous context:\n{context_str}\n]\n\nCurrent question: {rewritten_query}"
        return {"rewritten_query": enriched_query, "node_trace": node_trace}

    return {"node_trace": node_trace}
