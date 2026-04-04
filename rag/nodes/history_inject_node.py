"""History inject node — builds hybrid history_context for generator.

Context layers:
  Facts  — mem0 semantic search results (long-term entities/preferences)
  Gist   — history_summary (incremental session summary, mid-term)
  Flow   — last 2 raw turns verbatim (short-term immediate context)

Runs BEFORE router.
"""
from __future__ import annotations

import config
from rag.graph.middleware import error_guard_middleware, timing_middleware
from rag.state.pipeline_state import PipelineState
from rag.tools.mem0 import Mem0Manager


@timing_middleware
@error_guard_middleware
def history_inject_node(state: PipelineState) -> dict:
    """Build hybrid history_context for the generator."""
    node_trace = list(state.get("node_trace", []))
    node_trace.append("history_inject")

    current_query = state.get("rewritten_query") or state.get("query", "")
    history_context = _build_hybrid_context(state, current_query)

    result: dict = {"node_trace": node_trace}
    if history_context:
        result["history_context"] = history_context
    return result


def _build_hybrid_context(state: PipelineState, current_query: str) -> str:
    """Assemble Facts + Gist + Flow context string."""
    sections: list[str] = []

    # Layer 1: Facts (mem0 semantic search)
    if config.MEM0_ENABLED:
        session_id = state.get("session_id", "")
        if session_id:
            facts = Mem0Manager(session_id).search_context(current_query, top_k=3)
            if facts:
                sections.append(f"[Relevant facts about this user]\n{facts}")

    # Layer 2: Gist (incremental session summary)
    history_summary = state.get("history_summary", "") or ""
    if history_summary:
        sections.append(f"[Session summary]\n{history_summary}")

    # Layer 3: Flow (last 2 raw turns = 4 messages)
    history = state.get("history", [])
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
