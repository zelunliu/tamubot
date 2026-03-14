"""History update node — appends current turn; compresses if > MAX_HISTORY_TURNS.

Runs AFTER generator. Compression triggered when history length > V4_MAX_HISTORY_TURNS.
"""
from __future__ import annotations
from typing import Any

import config
from rag.v4.state import PipelineState, ConversationMessage
from rag.v4.middleware import error_guard_middleware, timing_middleware


@timing_middleware
@error_guard_middleware
def history_update_node(state: PipelineState, registry: Any) -> dict:
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
    if len(history) > max_turns * 2:  # *2 because each turn = 2 messages
        history = _compress_history(history, registry, max_turns)

    return {
        "history": history,
        "turn_number": turn_number,
        "node_trace": node_trace,
        # Clear non-checkpointable fields before graph exits
        "answer_stream": None,
        "trace": None,
    }


def _compress_history(history: list, registry: Any, max_turns: int) -> list:
    """Keep last max_turns turns, drop older messages."""
    # Simple windowing: keep the last max_turns * 2 messages
    # LLM compression would be ideal but windowing is sufficient for now
    return history[-(max_turns * 2):]
