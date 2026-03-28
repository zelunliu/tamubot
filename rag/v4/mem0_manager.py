"""Mem0Manager — per-session semantic memory using mem0.

Wraps mem0.Memory with:
- In-memory Qdrant vector store (no disk, session-scoped)
- TAMU AI gateway LLM (fact extraction)
- Voyage AI embedder (same model used by RAG pipeline)

Usage:
    manager = Mem0Manager.create(session_id="abc123")
    # After each turn (async/background):
    manager.add_turn(user_msg, assistant_msg)
    # Before retrieval:
    context_str = manager.search_context(query, top_k=3)
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class Mem0Manager:
    """Manages per-session semantic memories using mem0."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._memory = self._build_memory(session_id)

    @staticmethod
    def _build_memory(session_id: str):
        """Initialize mem0.Memory with in-memory Qdrant + TAMU LLM + Voyage embedder."""
        from qdrant_client import QdrantClient
        from mem0 import Memory
        from mem0.utils.factory import LlmFactory, EmbedderFactory
        from mem0.configs.llms.base import BaseLlmConfig

        # Register custom providers (idempotent — safe to call multiple times)
        LlmFactory.provider_to_class["tamu"] = (
            "rag.v4.tamu_mem0_llm.TamuMem0LLM",
            BaseLlmConfig,
        )
        EmbedderFactory.provider_to_class["voyage_mem0"] = (
            "rag.v4.voyage_mem0_embedder.VoyageMem0Embedder"
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
