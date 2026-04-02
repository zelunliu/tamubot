"""History update node — appends current turn; compresses if > MAX_HISTORY_TURNS.

Runs AFTER generator. Compression triggered when history length > V4_MAX_HISTORY_TURNS.
When ENABLE_HISTORY_SUMMARY=True, evicted turns are LLM-compressed into history_summary
instead of being dropped.
"""
from __future__ import annotations

from typing import Any

import config
from rag.v4.middleware import error_guard_middleware, timing_middleware, tracing_middleware
from rag.v4.state import ConversationMessage, ConversationState


@tracing_middleware
@timing_middleware
@error_guard_middleware
def history_update_node(state: ConversationState, registry: Any) -> dict:
    """Append current turn to history; compress older turns if needed."""
    history = list(state.get("history", []))
    node_trace = list(state.get("node_trace", []))
    node_trace.append("history_update")

    # Append current turn
    query = state.get("query", "")
    answer = state.get("answer", "")

    if query:
        history.append(ConversationMessage(role="user", content=query))
    if answer:
        router_result = state.get("router_result")
        rr_summary = None
        if router_result is not None:
            try:
                rr_summary = {
                    "function": router_result.function,
                    "course_ids": router_result.course_ids,
                }
            except Exception:
                pass
        history.append(ConversationMessage(role="assistant", content=answer, router_result=rr_summary))

    turn_number = state.get("turn_number", 0) + 1

    # Compress if over limit
    max_turns = config.V4_MAX_HISTORY_TURNS
    history_summary = state.get("history_summary", "") or ""
    history_compressed = False
    if len(history) > max_turns * 2:  # *2 because each turn = 2 messages
        history_compressed = True
        if config.ENABLE_HISTORY_SUMMARY:
            history, history_summary = _compress_history_with_llm(
                history, max_turns, history_summary, registry
            )
        else:
            history = _compress_history(history, max_turns)

    # --- answer cache ---
    answer_cache_update = {}
    if config.SESSION_CACHE_ENABLED and query and answer:
        from rag.v4.cache_utils import normalize_query
        existing_cache = state.get("answer_cache", {})
        answer_cache_update = {**existing_cache, normalize_query(query): answer}

    # --- async mem0 fact extraction ---
    if config.MEM0_ENABLED and query and answer:
        session_id = state.get("session_id", "")
        if session_id:
            from rag.v4.mem0_registry import get as get_mem0_manager
            mem0_manager = get_mem0_manager(session_id)
            if mem0_manager is not None:
                mem0_manager.add_turn_async(query, answer)

    return {
        "history": history,
        "history_summary": history_summary,
        "turn_number": turn_number,
        "node_trace": node_trace,
        "answer_cache": answer_cache_update,
        "history_compressed": history_compressed,
        # Clear non-checkpointable fields before graph exits
        "answer_stream": None,
        "trace": None,
    }


def _compress_history(history: list, max_turns: int) -> list:
    """Keep last max_turns turns, drop older messages."""
    return history[-(max_turns * 2):]


def _compress_history_with_llm(
    history: list, max_turns: int, existing_summary: str, registry: Any
) -> tuple[list, str]:
    """Summarize oldest turns with LLM, keep last max_turns turns.

    Returns (trimmed_history, new_summary).
    Falls back to plain truncation if registry.generator_llm is None.
    """
    keep_count = max_turns * 2
    turns_to_compress = history[:-keep_count]
    kept_history = history[-keep_count:]

    generator = getattr(registry, "generator_llm", None)
    if generator is None:
        return kept_history, existing_summary

    new_summary = generator.summarize_history(turns_to_compress, existing_summary)
    return kept_history, new_summary
