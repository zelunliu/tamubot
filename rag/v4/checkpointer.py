"""Factory for LangGraph checkpointers.

"memory" → MemorySaver (default, no persistence across process restarts)
"sqlite" → SqliteSaver("sessions.db") (persists across restarts)
"""
from __future__ import annotations
from typing import Optional

import config


def make_checkpointer(backend: Optional[str] = None):
    """Create a LangGraph checkpointer based on config.V4_CHECKPOINTER_BACKEND.

    Args:
        backend: Override backend ("memory" | "sqlite"). Defaults to config.V4_CHECKPOINTER_BACKEND.

    Returns:
        A LangGraph checkpointer (MemorySaver or SqliteSaver)
    """
    backend = backend or config.V4_CHECKPOINTER_BACKEND

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver("sessions.db")
        except ImportError:
            # langgraph-checkpoint-sqlite not installed; fall back to memory
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
    else:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
