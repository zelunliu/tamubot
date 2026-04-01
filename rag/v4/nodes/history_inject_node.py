"""History inject node — builds history_context for generator from last N turns.

Runs AFTER router. Writes to history_context state key (used by generator).
Does NOT modify rewritten_query — that stays clean for retrieval embedding.
"""
from __future__ import annotations

from typing import Any, Optional

import config
from rag.v4.middleware import error_guard_middleware, timing_middleware
from rag.v4.state import PipelineState


@timing_middleware
@error_guard_middleware
def history_inject_node(state: PipelineState, registry: Any) -> dict:
    """Build history_context for the generator from mem0 or raw history."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("history_inject")

    current_query = state.get("rewritten_query") or state.get("query", "")
    history_context = _build_history_context(state, current_query)

    result: dict = {"node_trace": node_trace}
    if history_context:
        result["history_context"] = history_context
    return result


def _build_history_context(state: PipelineState, current_query: str) -> str:
    """Return a formatted history string for the generator, or '' if none available."""
    # Try mem0 semantic retrieval first
    if config.MEM0_ENABLED:
        session_id = state.get("session_id", "")
        if session_id:
            from rag.v4.mem0_registry import get as get_mem0_manager
            mem0_manager = get_mem0_manager(session_id)
            if mem0_manager is not None:
                ctx = mem0_manager.search_context(current_query, top_k=3)
                if ctx:
                    return ctx

    # Fall back to raw windowed history
    history = state.get("history", [])
    history_summary = state.get("history_summary", "") or ""

    if not history and not history_summary:
        return ""

    lines: list[str] = []
    if history_summary:
        lines.append(f"[Summary of earlier turns: {history_summary}]")

    recent = history[-6:]  # last 3 turns = 6 messages
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")[:200]
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)
