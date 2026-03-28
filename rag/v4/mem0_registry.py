"""Global per-session Mem0Manager registry.

Nodes access mem0 managers via session_id from state.
App registers managers when sessions start, unregisters on session end.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.v4.mem0_manager import Mem0Manager

_managers: dict[str, "Mem0Manager"] = {}


def register(session_id: str, manager: "Mem0Manager") -> None:
    """Register a Mem0Manager for a session."""
    _managers[session_id] = manager


def get(session_id: str) -> "Mem0Manager | None":
    """Get the Mem0Manager for a session, or None if not registered."""
    return _managers.get(session_id)


def unregister(session_id: str) -> None:
    """Remove a session's Mem0Manager (e.g. on session end)."""
    _managers.pop(session_id, None)
