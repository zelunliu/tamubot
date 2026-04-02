"""Haystack-compatible reranking components."""
from __future__ import annotations

import time
from typing import Optional

from haystack import component

import config
from rag.v4.trace_registry import current_span as _current_span


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

        parent = _current_span()
        rerank_span = None
        t0 = time.perf_counter()
        if parent is not None:
            try:
                rerank_span = parent.span(
                    name="rerank.voyage",
                    input={"n_in": len(chunks), "query": query[:100]},
                    metadata={"model": self.model},
                )
            except Exception:
                rerank_span = None

        try:
            client = self._get_client()
            texts = [c.get("text", c.get("content", "")) for c in chunks]
            result = client.rerank(query, texts, model=self.model, top_k=min(top_k, len(chunks)))
            reranked = [chunks[r.index] for r in result.results]
            # Apply stratified selection if specific_categories provided
            if specific_categories:
                from rag.reranker import stratified_select
                reranked = stratified_select(reranked, specific_categories)
        except Exception:
            if rerank_span is not None:
                try:
                    rerank_span.end(metadata={
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "error": True,
                    })
                except Exception:
                    pass
            raise

        if rerank_span is not None:
            try:
                rerank_span.end(
                    output={"n_out": len(reranked)},
                    metadata={"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)},
                )
            except Exception:
                pass

        return {"chunks": reranked}

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
        specific_categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """Satisfy RerankerComponent protocol — delegates to run() and unwraps result."""
        result = self.run(query=query, chunks=chunks, top_k=top_k, specific_categories=specific_categories)
        return result["chunks"]


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

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
        specific_categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """Satisfy RerankerComponent protocol — delegates to run() and unwraps result."""
        result = self.run(query=query, chunks=chunks, top_k=top_k, specific_categories=specific_categories)
        return result["chunks"]
