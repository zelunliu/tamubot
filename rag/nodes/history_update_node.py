"""History update node — appends current turn; truncates if > MAX_HISTORY_TURNS.

Runs AFTER generator.
"""
from __future__ import annotations

import config
from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import ConversationMessage, ConversationState


@timing_middleware
@error_guard_middleware
def history_update_node(state: ConversationState) -> dict:
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

    # Truncate if over limit
    max_turns = config.V4_MAX_HISTORY_TURNS
    history_summary = state.get("history_summary", "") or ""
    history_compressed = False
    if len(history) > max_turns * 2:  # *2 because each turn = 2 messages
        history_compressed = True
        history = _compress_history(history, max_turns)

    # --- answer cache ---
    answer_cache_update = {}
    if config.SESSION_CACHE_ENABLED and query and answer:
        from rag.graph.cache_utils import normalize_query
        existing_cache = state.get("answer_cache", {})
        answer_cache_update = {**existing_cache, normalize_query(query): answer}

    # --- async mem0 fact extraction ---
    if config.MEM0_ENABLED and query and answer:
        session_id = state.get("session_id", "")
        if session_id:
            from rag.tools.mem0 import get_mem0_manager
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
    }


def _compress_history(history: list, max_turns: int) -> list:
    """Keep last max_turns turns, drop older messages."""
    return history[-(max_turns * 2):]
