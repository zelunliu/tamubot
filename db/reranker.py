"""Voyage AI reranking for the retrieval pipeline.

Uses voyage-rerank-2 cross-encoder to rerank over-retrieved candidates
down to the most relevant top-k results.
"""

from itertools import chain

import voyageai

import config


_voyage = None


def _get_voyage():
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _voyage


def rerank(
    query: str,
    documents: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Rerank a list of chunk documents using Voyage AI cross-encoder.

    Args:
        query: the user's (possibly rewritten) query
        documents: list of chunk dicts, each must have 'content' (and optionally 'title')
        top_k: number of results to keep (defaults to config.RERANK_TOP_K)

    Returns:
        Reranked list of chunk dicts, trimmed to top_k.
    """
    if not documents:
        return []

    top_k = top_k or config.RERANK_TOP_K

    # Build text representations for the reranker
    doc_texts = []
    for doc in documents:
        parts = []
        if doc.get("course_id"):
            parts.append(f"{doc['course_id']} Section {doc.get('section', '?')}")
        if doc.get("category"):
            parts.append(doc["category"])
        if doc.get("title"):
            parts.append(doc["title"])
        parts.append(doc.get("content", ""))
        doc_texts.append(" | ".join(parts))

    client = _get_voyage()
    result = client.rerank(
        query=query,
        documents=doc_texts,
        model=config.VOYAGE_RERANK_MODEL,
        top_k=min(top_k, len(documents)),
    )

    return [documents[r.index] for r in result.results]


def rerank_multi_course(
    query: str,
    course_groups: dict[str, list[dict]],
    top_k_per_course: int = 3,
) -> list[dict]:
    """Balanced reranking for multi-course comparison queries.

    Reranks each course's documents independently, then interleaves to ensure
    at least top_k_per_course results per course.

    Args:
        query: the user's query
        course_groups: mapping of course_id → list of chunk dicts
        top_k_per_course: how many chunks to keep per course

    Returns:
        Interleaved list of reranked chunks with balanced representation.
    """
    if not course_groups:
        return []

    reranked_groups = {}
    for course_id, docs in course_groups.items():
        if docs:
            reranked_groups[course_id] = rerank(query, docs, top_k=top_k_per_course)
        else:
            reranked_groups[course_id] = []

    # Interleave: round-robin across courses for balanced representation
    result = []
    max_len = max((len(v) for v in reranked_groups.values()), default=0)
    for i in range(max_len):
        for course_id in sorted(reranked_groups.keys()):
            if i < len(reranked_groups[course_id]):
                result.append(reranked_groups[course_id][i])

    return result
