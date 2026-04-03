"""SessionManager — maps Streamlit session IDs to LangGraph thread_ids.

Canonical location: rag/graph/session.py

Critical: trace (LFTrace) and answer_stream (Iterator) are NOT picklable.
SessionManager strips these before checkpoint and re-injects trace after restore.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional


class SessionManager:
    """Maps Streamlit session IDs to LangGraph thread_id configs.

    Usage:
        manager = SessionManager()
        thread_config = manager.get_thread_config(st.session_state._session_id)
        result = graph.invoke(state, config=thread_config)
    """

    def __init__(self):
        # session_id (str) → thread_id (str)
        self._sessions: dict[str, str] = {}

    def get_thread_config(self, session_id: str) -> dict:
        """Get or create a LangGraph thread config for a Streamlit session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = str(uuid.uuid4())
        thread_id = self._sessions[session_id]
        return {"configurable": {"thread_id": thread_id}}

    def clear_session(self, session_id: str) -> None:
        """Remove a session (e.g., on logout or reset)."""
        self._sessions.pop(session_id, None)

    def strip_non_checkpointable(self, state: dict) -> dict:
        """Remove trace and answer_stream before checkpoint write."""
        return {k: v for k, v in state.items() if k not in ("trace", "answer_stream")}

    def inject_trace(self, state: dict, trace: Optional[Any]) -> dict:
        """Re-inject trace after checkpoint restore."""
        return {**state, "trace": trace}
