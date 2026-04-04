"""Mem0 Cloud API support: manager backed by mem0.MemoryClient.

mem0 is an optional dependency. Importing this module without mem0 installed is
safe as long as Mem0Manager is never instantiated.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import config

logger = logging.getLogger(__name__)

try:
    from mem0 import MemoryClient
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False


class Mem0Manager:
    """Thin wrapper around mem0.MemoryClient for per-session semantic memory.

    Stateless — no local vector store or LLM adapter. All persistence is
    managed by the Mem0 Cloud API. Instantiate on demand; no registry needed.

    Usage:
        manager = Mem0Manager(session_id="abc123")
        manager.add_turn(user_msg, assistant_msg)        # blocking
        manager.add_turn_async(user_msg, assistant_msg)  # fire-and-forget
        context_str = manager.search_context(query, top_k=3)
    """

    def __init__(self, session_id: str):
        if not _MEM0_AVAILABLE:
            raise ImportError("mem0 is not installed; cannot use Mem0Manager")
        self.session_id = session_id
        self._client = MemoryClient(api_key=config.MEM0_API_KEY)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Extract and store facts from a conversation turn (blocking)."""
        try:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            self._client.add(messages, user_id=self.session_id)
        except Exception:
            logger.exception("mem0 add_turn failed (non-fatal)")

    def add_turn_async(self, user_msg: str, assistant_msg: str) -> None:
        """Fire-and-forget: run add_turn in a background thread."""
        t = threading.Thread(
            target=self.add_turn,
            args=(user_msg, assistant_msg),
            daemon=True,
        )
        t.start()

    def search_context(self, query: str, top_k: int = 3) -> str:
        """Search for relevant memories; return compact context string or ''."""
        try:
            results = self._client.search(query, user_id=self.session_id, limit=top_k)
            # MemoryClient.search returns a list of dicts with a "memory" key
            facts = [r["memory"] for r in results if r.get("memory")]
            if not facts:
                return ""
            return "\n".join(f"- {f}" for f in facts)
        except Exception:
            logger.exception("mem0 search_context failed (non-fatal)")
            return ""


# ---------------------------------------------------------------------------
# Session registry (backward-compat shims — used by app.py and existing tests)
# Nodes should instantiate Mem0Manager directly going forward.
# ---------------------------------------------------------------------------

_registry: dict[str, Mem0Manager] = {}
_registry_lock = threading.Lock()


def get_mem0_manager(session_id: str) -> Optional[Mem0Manager]:
    """Get the Mem0Manager for a session from the registry, or None if absent."""
    with _registry_lock:
        return _registry.get(session_id)


def register_mem0_manager(session_id: str) -> Mem0Manager:
    """Get or create a Mem0Manager for a session (creates if absent)."""
    with _registry_lock:
        if session_id not in _registry:
            _registry[session_id] = Mem0Manager(session_id)
        return _registry[session_id]


def clear_mem0_manager(session_id: str) -> None:
    """Remove a session's Mem0Manager from the registry."""
    with _registry_lock:
        _registry.pop(session_id, None)


# Backward-compat aliases matching old mem0_registry interface

def get(session_id: str) -> Optional[Mem0Manager]:
    """Alias for get_mem0_manager (backward compat)."""
    return get_mem0_manager(session_id)


def register(session_id: str, manager: Mem0Manager) -> None:
    """Register an existing Mem0Manager instance for a session (backward compat)."""
    with _registry_lock:
        _registry[session_id] = manager


def unregister(session_id: str) -> None:
    """Alias for clear_mem0_manager (backward compat)."""
    clear_mem0_manager(session_id)
