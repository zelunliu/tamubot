"""History update node — appends current turn; updates session summary via LLM.

Runs AFTER generator.
"""
from __future__ import annotations

import config
from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import ConversationMessage, ConversationState
from rag.tools.llm import call_llm
from rag.tools.mem0 import Mem0Manager


@timing_middleware
@error_guard_middleware
def history_update_node(state: ConversationState) -> dict:
    """Append current turn to history; update session summary; fire mem0 async."""
    history = list(state.get("history", []))
    node_trace = list(state.get("node_trace", []))
    node_trace.append("history_update")

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
    history_summary = state.get("history_summary", "") or ""

    # Truncate raw history if over limit (keep last N turns)
    max_turns = config.V4_MAX_HISTORY_TURNS
    history_compressed = False
    if len(history) > max_turns * 2:
        history_compressed = True
        history = history[-(max_turns * 2):]

    # Incremental session summary (LLM) — skip first turn (nothing to summarize yet)
    if config.MEM0_ENABLED and query and answer and turn_number > 1:
        history_summary = _update_summary(history_summary, query, answer)

    # answer cache
    answer_cache_update = {}
    if config.SESSION_CACHE_ENABLED and query and answer:
        from rag.graph.cache_utils import normalize_query
        existing_cache = state.get("answer_cache", {})
        answer_cache_update = {**existing_cache, normalize_query(query): answer}

    # async mem0 fact extraction
    if config.MEM0_ENABLED and query and answer:
        session_id = state.get("session_id", "")
        if session_id:
            Mem0Manager(session_id).add_turn_async(query, answer)

    return {
        "history": history,
        "history_summary": history_summary,
        "turn_number": turn_number,
        "node_trace": node_trace,
        "answer_cache": answer_cache_update,
        "history_compressed": history_compressed,
        "answer_stream": None,
    }


def _update_summary(existing_summary: str, user_msg: str, assistant_msg: str) -> str:
    """Call LLM to produce an updated concise session summary."""
    prior = f"Prior summary: {existing_summary}" if existing_summary else "No prior summary."
    messages = [
        {
            "role": "user",
            "content": (
                f"{prior}\n\n"
                f"Latest turn:\nUser: {user_msg}\nAssistant: {assistant_msg}\n\n"
                "Update the session summary to include the latest turn. "
                "Be concise (2-4 sentences max). Return only the updated summary text."
            ),
        }
    ]
    try:
        result = call_llm(messages, temperature=0.0, max_tokens=4096)
        return result.text.strip()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("summary update failed (non-fatal)")
        return existing_summary
