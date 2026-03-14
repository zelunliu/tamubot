"""Bridges MongoDocumentStore + VoyageEmbedder into RetrieverComponent protocol."""
from __future__ import annotations
from typing import Any, Optional

from rag.v4.components.document_stores import MongoDocumentStore
from rag.v4.components.embedders import VoyageEmbedder


class MongoRetrieverAdapter:
    """Wraps MongoDocumentStore + VoyageEmbedder to satisfy RetrieverComponent.

    This is the default production retriever for Phase 3+.
    """

    def __init__(self, store: MongoDocumentStore, embedder: VoyageEmbedder):
        self.store = store
        self.embedder = embedder
        store.set_embedder(embedder)

    def hybrid_search(
        self, query: str, course_id: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        return self.store.hybrid_search(query, course_id, retrieve_k, embedding)

    def semantic_search(
        self, query: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        return self.store.semantic_search(query, retrieve_k, embedding)

    def fetch_anchor_chunks(
        self, course_ids: list[str], categories: list[str]
    ) -> tuple[list[dict], list[tuple[str, str]], bool]:
        return self.store.fetch_anchor_chunks(course_ids, categories)

    def get_meeting_times(self, course_ids: list[str]) -> dict[str, Any]:
        return self.store.get_meeting_times(course_ids)

    def get_syllabus_urls(self, course_ids: list[str]) -> dict[str, str]:
        return self.store.get_syllabus_urls(course_ids)
