"""Thread-safe trace registry for v4 pipeline.

Stores active Langfuse trace objects keyed by session_id. Used to pass
traces into graph nodes without including them in LangGraph state
(LFTrace objects are not picklable and cannot be checkpointed).

Also maintains a per-thread span stack so child components (reranker,
document store) can create nested child spans without signature changes.
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


# ---------------------------------------------------------------------------
# Thread-local span stack — used by tracing_middleware to propagate the
# current node span to child components (reranker, document store, etc.)
# without changing any function signatures.
# ---------------------------------------------------------------------------
_tl = threading.local()


def push_span(span: Any) -> None:
    """Push a span onto the per-thread stack. Called by tracing_middleware on node entry."""
    if not hasattr(_tl, "spans"):
        _tl.spans = []
    _tl.spans.append(span)


def pop_span() -> Optional[Any]:
    """Pop and return the top span. Called by tracing_middleware on node exit."""
    if hasattr(_tl, "spans") and _tl.spans:
        return _tl.spans.pop()
    return None


def current_span() -> Optional[Any]:
    """Return the innermost active span for the current thread, or None.

    Used by components (reranker, document store, generator) to obtain
    their parent span for creating nested child spans.
    """
    if hasattr(_tl, "spans") and _tl.spans:
        return _tl.spans[-1]
    return None
