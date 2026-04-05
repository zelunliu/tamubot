"""History inject node — builds hybrid history_context for generator.

Context layers:
  Facts  — mem0 semantic search results (long-term entities/preferences)
  Gist   — history_summary (incremental session summary, mid-term)
  Flow   — last 2 raw turns verbatim (short-term immediate context)

Runs BEFORE router.
"""
from __future__ import annotations

import logging

import config
from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import ConversationState
from rag.tools.mem0 import Mem0Manager

logger = logging.getLogger("tamubot")

try:
    from langfuse import get_client as _lf_get_client, observe as _lf_observe
except ImportError:
    _lf_get_client = lambda: None  # type: ignore[assignment]
    def _lf_observe(**_kw):  # type: ignore[misc]
        return lambda fn: fn


@timing_middleware
@error_guard_middleware
@_lf_observe(name="history_inject_node")
def history_inject_node(state: ConversationState) -> dict:
    """Build hybrid history_context for the generator."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("history_inject")

    current_query = state.get("query", "")
    history_context = _build_hybrid_context(state, current_query)

    history_summary = state.get("history_summary", "") or ""
    history_turns = len(state.get("history", []))

    # Augment query with full history_context so router + generator see prior context
    original_query = state.get("query", "")
    rewritten_query = f"{original_query}\n\n{history_context}" if history_context else None

    logger.info(
        "history_inject: summary_len=%d history_turns=%d context_len=%d rewritten_query_len=%d",
        len(history_summary),
        history_turns,
        len(history_context),
        len(rewritten_query or ""),
    )
    lf = _lf_get_client()
    if lf is not None:
        try:
            lf.update_current_span(metadata={
                "history_summary_len": len(history_summary),
                "history_turns": history_turns,
                "history_context_len": len(history_context),
                "rewritten_query_len": len(rewritten_query or ""),
            })
        except Exception:
            pass

    # Always write history_context to clear any stale checkpoint value
    return {
        "node_trace": node_trace,
        "history_context": history_context,
        "rewritten_query": rewritten_query,
    }


def _build_hybrid_context(state: ConversationState, current_query: str) -> str:
    """Assemble Facts + Gist + Flow context string."""
    sections: list[str] = []

    # Layer 1: Facts (mem0 semantic search)
    if config.MEM0_ENABLED:
        session_id = state.get("session_id", "")
        if session_id:
            try:
                facts = Mem0Manager(session_id).search_context(current_query, top_k=3)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("mem0 search failed (non-fatal)")
                facts = ""
            if facts:
                sections.append(f"[Relevant facts about this user]\n{facts}")

    # Layer 2: Gist (incremental session summary)
    history_summary = state.get("history_summary", "") or ""
    if history_summary:
        sections.append(f"[Session summary]\n{history_summary}")

    # Layer 3: Flow (last 2 raw turns = 4 messages)
    history = state.get("history") or []
    recent = history[-4:]
    if recent:
        flow_lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = (msg.get("content", "") or "")[:200]
            if content:
                flow_lines.append(f"{role}: {content}")
        if flow_lines:
            sections.append("[Recent conversation]\n" + "\n".join(flow_lines))

    return "\n\n".join(sections)
