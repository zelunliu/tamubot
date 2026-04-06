"""History update node — appends current turn; updates session summary via LLM.

Runs AFTER generator.
"""
from __future__ import annotations

import logging

import config
from rag.graph.middleware import error_guard_middleware, timing_middleware

logger = logging.getLogger("tamubot")
from rag.state.pipeline_state import ConversationMessage, PipelineState
from rag.tools.llm import call_llm
from rag.tools.mem0 import Mem0Manager


@timing_middleware
@error_guard_middleware
def history_update_node(state: PipelineState) -> dict:
    """Append current turn to history; update session summary; fire mem0 async."""
    history = list(state.get("history", []))
    logger.info("history_update: start turn_number=%d prior_history_len=%d", state.get("turn_number", 0), len(history))
    node_trace = list(state.get("node_trace", []))
    node_trace.append("history_update")

    query = state.get("query", "")
    # Use coreference-resolved query for summary (router may have expanded it further);
    # fall back to raw query if not set.
    resolved_query = state.get("rewritten_query") or query
    answer = state.get("answer", "")

    if query:
        history.append(ConversationMessage(role="user", content=query))
    if answer:
        rr_summary = {
            "function": state.get("function", "out_of_scope"),
            "course_ids": state.get("course_ids", []),
        }
        history.append(ConversationMessage(role="assistant", content=answer, router_result=rr_summary))

    turn_number = state.get("turn_number", 0) + 1
    history_summary = state.get("history_summary", "") or ""

    # Truncate raw history if over limit (keep last N turns)
    max_turns = config.V4_MAX_HISTORY_TURNS
    history_compressed = False
    if len(history) > max_turns * 2:
        history_compressed = True
        history = history[-(max_turns * 2):]

    # Incremental session summary (LLM) — always update, independent of MEM0_ENABLED
    if query and answer:
        history_summary = _update_summary(history_summary, resolved_query, answer)

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
            try:
                Mem0Manager(session_id).add_turn_async(query, answer)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("mem0 add_turn_async failed (non-fatal)")

    logger.info("history_update: done history_len=%d summary_len=%d", len(history), len(history_summary))
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
        updated = result.text.strip()
        return updated if updated else existing_summary
    except Exception:
        import logging
        logging.getLogger(__name__).exception("summary update failed (non-fatal)")
        return existing_summary
