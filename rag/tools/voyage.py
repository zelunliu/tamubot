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
from langfuse import observe

import config

EMBEDDING_MODEL = "voyage-3"
RERANK_MODEL = config.VOYAGE_RERANK_MODEL

_voyage: Optional[voyageai.Client] = None


def _get_client() -> voyageai.Client:
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _voyage


@observe(name="pipeline.retrieval.embed")
def embed_query(text: str) -> list[float]:
    """Embed a query string using Voyage AI voyage-3."""
    client = _get_client()
    result = client.embed([text], model=EMBEDDING_MODEL, input_type="query")
    return result.embeddings[0]


def knee_filter(
    chunks: list[dict],
    *,
    min_floor: int = 2,
    abs_threshold: float = 0.15,
    min_gap_fallback: float = 0.05,
) -> list[dict]:
    """Cut reranked chunks at the first meaningful score drop.

    Primary: cut at the first consecutive gap exceeding abs_threshold.
    Fallback: cut at the largest gap, but only if it exceeds min_gap_fallback
              (avoids over-filtering when all scores are uniformly high).
    Always keeps at least min_floor chunks regardless of gap size.
    """
    if len(chunks) <= min_floor:
        return chunks

    scores = [c.get("score", 0.0) for c in chunks]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

    # Primary: first absolute gap exceeding threshold
    for i, gap in enumerate(gaps):
        if gap > abs_threshold:
            return chunks[: max(i + 1, min_floor)]

    # Fallback: largest gap (only if meaningful noise floor exceeded)
    max_gap = max(gaps)
    if max_gap > min_gap_fallback:
        cut = gaps.index(max_gap)
        return chunks[: max(cut + 1, min_floor)]

    return chunks


@observe(name="pipeline.retrieval.rerank")
def rerank(query: str, chunks: list[dict], top_k: int, *, apply_knee: bool = True) -> list[dict]:
    """Cross-encoder rerank chunks by relevance to query, return top_k.

    Preserves all original chunk fields. Adds/updates 'score' field.
    Returns chunks sorted descending by score.
    Always drops chunks below config.RERANK_SCORE_THRESHOLD (floor: config.RERANK_SCORE_MIN_CHUNKS).
    If config.RERANK_KNEE_ENABLED and apply_knee, additionally applies knee-point filtering.
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
        # Fixed score threshold — always active
        min_chunks = config.RERANK_SCORE_MIN_CHUNKS
        filtered = [c for c in results if c.get("score", 0.0) >= config.RERANK_SCORE_THRESHOLD]
        results = filtered if len(filtered) >= min_chunks else results[:min_chunks]
        # Knee-point filter — optional, off by default
        if config.RERANK_KNEE_ENABLED and apply_knee:
            results = knee_filter(
                results,
                min_floor=config.RERANK_KNEE_MIN_CHUNKS,
                abs_threshold=config.RERANK_KNEE_ABS_THRESHOLD,
                min_gap_fallback=config.RERANK_KNEE_MIN_GAP_FALLBACK,
            )
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
