"""Haystack-compatible reranking components."""
from __future__ import annotations
from typing import Optional
from haystack import component
import config


@component
class VoyageReranker:
    """Reranks chunks using voyage-rerank-2 via the voyageai client."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.VOYAGE_RERANK_MODEL or "rerank-2"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import voyageai
            self._client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
        return self._client

    @component.output_types(chunks=list)
    def run(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
        specific_categories: Optional[list[str]] = None,
    ) -> dict:
        if not chunks:
            return {"chunks": []}
        client = self._get_client()
        texts = [c.get("text", c.get("content", "")) for c in chunks]
        result = client.rerank(query, texts, model=self.model, top_k=min(top_k, len(chunks)))
        reranked = [chunks[r.index] for r in result.results]
        # Apply stratified selection if specific_categories provided
        if specific_categories:
            from rag.reranker import stratified_select
            reranked = stratified_select(reranked, specific_categories)
        return {"chunks": reranked}


@component
class IdentityReranker:
    """Pass-through reranker. Returns chunks unchanged. Used for testing."""

    @component.output_types(chunks=list)
    def run(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
        specific_categories: Optional[list[str]] = None,
    ) -> dict:
        return {"chunks": chunks[:top_k] if chunks else []}
