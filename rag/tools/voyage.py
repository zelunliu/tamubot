"""Voyage AI tool — embedding and reranking.

Exposes:
  embed_query(text) -> list[float]
  rerank(query, chunks, top_k) -> list[dict]
  stratified_select(chunks, k) -> list[dict]

Canonical location: rag/tools/voyage.py
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import voyageai

import config

EMBEDDING_MODEL = "voyage-3"
RERANK_MODEL = "rerank-2"

_voyage: Optional[voyageai.Client] = None


def _get_client() -> voyageai.Client:
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _voyage


def embed_query(text: str) -> list[float]:
    """Embed a query string using Voyage AI voyage-3."""
    client = _get_client()
    result = client.embed([text], model=EMBEDDING_MODEL, input_type="query")
    return result.embeddings[0]


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Cross-encoder rerank chunks by relevance to query, return top_k.

    Preserves all original chunk fields. Adds/updates 'score' field.
    Returns chunks sorted descending by score.
    Falls back to original order (sliced to top_k) on any Voyage error.
    """
    if not chunks:
        return []
    top_k = min(top_k, len(chunks))
    client = _get_client()
    texts = [c.get("content", "") for c in chunks]
    try:
        response = client.rerank(query=query, documents=texts, model=RERANK_MODEL, top_k=top_k)
        results = []
        for item in response.results:
            chunk = dict(chunks[item.index])
            chunk["score"] = item.relevance_score
            results.append(chunk)
        return results
    except Exception:
        return chunks[:top_k]


def stratified_select(chunks: list[dict], k: int) -> list[dict]:
    """Select up to k chunks with at most ceil(k/n_courses) per course.

    Used as a fallback when Voyage reranking is unavailable.
    """
    if not chunks or k <= 0:
        return []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        buckets[c.get("course_id", "_")].append(c)
    n = len(buckets)
    per_course = math.ceil(k / n) if n else k
    selected = []
    for course_chunks in buckets.values():
        selected.extend(course_chunks[:per_course])
    return selected[:k]
