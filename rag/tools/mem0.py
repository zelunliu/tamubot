"""Consolidated mem0 support: LLM adapter, embedder, manager, and session registry.

Consolidates:
- rag/v4/tamu_mem0_llm.py        — TamuMem0LLM class
- rag/v4/voyage_mem0_embedder.py  — VoyageMem0Embedder class
- rag/v4/mem0_manager.py          — Mem0Manager class
- rag/v4/mem0_registry.py         — session registry functions

mem0 is an optional dependency. Importing this module without mem0 installed is
safe as long as Mem0Manager is never instantiated and TamuMem0LLM /
VoyageMem0Embedder are not constructed directly.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TamuMem0LLM  (requires mem0)
# ---------------------------------------------------------------------------

try:
    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.llms.base import LLMBase

    class TamuMem0LLM(LLMBase):
        """mem0 LLM adapter backed by the TAMU AI gateway via call_llm().

        The TAMU gateway always returns SSE. This adapter wraps call_llm()
        which handles the streaming internally, presenting a blocking interface
        to mem0.
        """

        def __init__(self, config: Optional[BaseLlmConfig] = None):
            if config is None:
                config = BaseLlmConfig()
            super().__init__(config)

        def generate_response(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
            **kwargs,
        ) -> str:
            """Generate a response using the TAMU AI gateway."""
            from rag.llm_client import call_llm
            result = call_llm(messages, temperature=0.1, max_tokens=4096)
            return result.text

except ImportError:
    class TamuMem0LLM:  # type: ignore[no-redef]
        """Placeholder — mem0 not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("mem0 is not installed; cannot use TamuMem0LLM")


# ---------------------------------------------------------------------------
# VoyageMem0Embedder  (requires mem0)
# ---------------------------------------------------------------------------

try:
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.base import EmbeddingBase

    class VoyageMem0Embedder(EmbeddingBase):
        """mem0 embedder backed by Voyage AI voyage-3 model (1024-dim embeddings)."""

        DIMS = 1024

        def __init__(self, config: Optional[BaseEmbedderConfig] = None):
            if config is None:
                config = BaseEmbedderConfig(embedding_dims=self.DIMS)
            super().__init__(config)
            self._client = None

        def _get_client(self):
            if self._client is None:
                import voyageai
                import config as app_config
                self._client = voyageai.Client(api_key=app_config.VOYAGE_API_KEY)
            return self._client

        def embed(
            self,
            text: str,
            memory_action: Optional[Literal["add", "search", "update"]] = None,
        ) -> list:
            """Embed text using Voyage AI voyage-3 model."""
            client = self._get_client()
            result = client.embed([text], model="voyage-3", input_type="query")
            return result.embeddings[0]

except ImportError:
    class VoyageMem0Embedder:  # type: ignore[no-redef]
        """Placeholder — mem0 not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("mem0 is not installed; cannot use VoyageMem0Embedder")


# ---------------------------------------------------------------------------
# Mem0Manager
# ---------------------------------------------------------------------------

class Mem0Manager:
    """Manages per-session semantic memories using mem0.

    Wraps mem0.Memory with:
    - In-memory Qdrant vector store (no disk, session-scoped)
    - TAMU AI gateway LLM (fact extraction)
    - Voyage AI embedder (same model used by RAG pipeline)

    Usage:
        manager = Mem0Manager(session_id="abc123")
        # After each turn (async/background):
        manager.add_turn(user_msg, assistant_msg)
        # Before retrieval:
        context_str = manager.search_context(query, top_k=3)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._memory = self._build_memory(session_id)

    @staticmethod
    def _build_memory(session_id: str):
        """Initialize mem0.Memory with in-memory Qdrant + TAMU LLM + Voyage embedder."""
        from mem0 import Memory
        from mem0.configs.llms.base import BaseLlmConfig
        from mem0.utils.factory import EmbedderFactory, LlmFactory
        from qdrant_client import QdrantClient

        # Register custom providers (idempotent — safe to call multiple times)
        LlmFactory.provider_to_class["tamu"] = (
            "rag.tools.mem0.TamuMem0LLM",
            BaseLlmConfig,
        )
        EmbedderFactory.provider_to_class["voyage_mem0"] = (
            "rag.tools.mem0.VoyageMem0Embedder"
        )

        collection_name = f"tamubot_{session_id[:8]}"
        qdrant_client = QdrantClient(":memory:")

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "client": qdrant_client,
                    "collection_name": collection_name,
                    "embedding_model_dims": 1024,
                },
            },
            "llm": {"provider": "tamu", "config": {}},
            "embedder": {"provider": "voyage_mem0", "config": {"embedding_dims": 1024}},
            "history_db_path": ":memory:",  # keep history in RAM, no disk writes
        }
        return Memory.from_config(config)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Extract and store facts from a conversation turn (blocking).

        Call this from a background thread to avoid blocking the response stream.
        """
        try:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            self._memory.add(messages, user_id=self.session_id)
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
        """Search for relevant memories and return a compact context string.

        Returns empty string if no memories found (e.g. first turn).
        """
        try:
            result = self._memory.search(query, user_id=self.session_id, limit=top_k)
            facts = [r["memory"] for r in result.get("results", []) if r.get("memory")]
            if not facts:
                return ""
            return "\n".join(f"- {f}" for f in facts)
        except Exception:
            logger.exception("mem0 search_context failed (non-fatal)")
            return ""


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

_registry: dict[str, Mem0Manager] = {}
_registry_lock = threading.Lock()


def get_mem0_manager(session_id: str) -> Optional[Mem0Manager]:
    """Get the Mem0Manager for a session, or None if not registered."""
    with _registry_lock:
        return _registry.get(session_id)


def register_mem0_manager(session_id: str) -> Mem0Manager:
    """Get or create a Mem0Manager for a session (creates if absent)."""
    with _registry_lock:
        if session_id not in _registry:
            _registry[session_id] = Mem0Manager(session_id)
        return _registry[session_id]


def clear_mem0_manager(session_id: str) -> None:
    """Remove a session's Mem0Manager (e.g. on session end)."""
    with _registry_lock:
        _registry.pop(session_id, None)


# Backward-compat aliases matching rag/v4/mem0_registry.py interface
# (get / register / unregister)

def get(session_id: str) -> Optional[Mem0Manager]:
    """Alias for get_mem0_manager (backward compat with mem0_registry.get)."""
    return get_mem0_manager(session_id)


def register(session_id: str, manager: Mem0Manager) -> None:
    """Register an existing Mem0Manager for a session (backward compat).

    Unlike register_mem0_manager, this accepts a pre-built manager instance.
    """
    with _registry_lock:
        _registry[session_id] = manager


def unregister(session_id: str) -> None:
    """Alias for clear_mem0_manager (backward compat with mem0_registry.unregister)."""
    clear_mem0_manager(session_id)
