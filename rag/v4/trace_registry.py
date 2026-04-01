"""Thread-safe trace registry for v4 pipeline.

Stores active Langfuse trace objects keyed by session_id. Used to pass
traces into graph nodes without including them in LangGraph state
(LFTrace objects are not picklable and cannot be checkpointed).
"""
from __future__ import annotations

import threading
from typing import Any, Optional

_active: dict[str, Any] = {}
_lock = threading.Lock()


def register(session_id: str, trace: Any) -> None:
    """Store a trace for the given session. No-op if session_id is empty."""
    if session_id:
        with _lock:
            _active[session_id] = trace


def get(session_id: str) -> Optional[Any]:
    """Retrieve trace for session_id, or None if not registered."""
    if not session_id:
        return None
    with _lock:
        return _active.get(session_id)


def clear(session_id: str) -> None:
    """Remove trace for session_id after pipeline completes."""
    with _lock:
        _active.pop(session_id, None)
