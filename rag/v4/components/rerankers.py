"""Haystack-compatible reranking components."""
from __future__ import annotations

from haystack import component

import config
from rag.v4.trace_registry import child_span as _child_span


@component
class VoyageReranker:
    """Reranks chunks using voyage-rerank-2 via the voyageai client."""

    def __init__(self, model: str | None = None):
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
    ) -> dict:
        if not chunks:
            return {"chunks": []}

        with _child_span(
            "rerank.voyage",
            {"n_in": len(chunks), "query": query[:100]},
            metadata={"model": self.model},
        ) as ctx:
            client = self._get_client()
            texts = [c.get("text", c.get("content", "")) for c in chunks]
            result = client.rerank(query, texts, model=self.model, top_k=min(top_k, len(chunks)))
            reranked = [chunks[r.index] for r in result.results]
            ctx["output"] = {"n_out": len(reranked)}

        return {"chunks": reranked}

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Satisfy RerankerComponent protocol — delegates to run() and unwraps result."""
        result = self.run(query=query, chunks=chunks, top_k=top_k)
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
    ) -> dict:
        return {"chunks": chunks[:top_k] if chunks else []}

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Satisfy RerankerComponent protocol — delegates to run() and unwraps result."""
        result = self.run(query=query, chunks=chunks, top_k=top_k)
        return result["chunks"]
