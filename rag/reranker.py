"""Voyage AI reranking for the retrieval pipeline.

Uses voyage-rerank-2 cross-encoder to rerank over-retrieved candidates
down to the most relevant top-k results.
"""


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
    parent_span=None,
) -> list[dict]:
    """Rerank a list of chunk documents using Voyage AI cross-encoder.

    Args:
        query:       The user's (possibly rewritten) query.
        documents:   List of chunk dicts, each must have 'content' (and optionally 'title').
        top_k:       Number of results to keep (defaults to config.RERANK_TOP_K).
        parent_span: Optional Langfuse span; creates a Voyage_Reranker child span.

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

    # Voyage_Reranker sub-span
    rerank_span = None
    if parent_span is not None:
        try:
            rerank_span = parent_span.span(
                name="Voyage_Reranker",
                input={
                    "query": query,
                    "n_docs": len(documents),
                    "top_k": top_k,
                    "model": config.VOYAGE_RERANK_MODEL,
                },
            )
        except Exception:
            rerank_span = None

    try:
        client = _get_voyage()
        result = client.rerank(
            query=query,
            documents=doc_texts,
            model=config.VOYAGE_RERANK_MODEL,
            top_k=min(top_k, len(documents)),
        )
    except Exception as e:
        if rerank_span is not None:
            try:
                rerank_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    if rerank_span is not None:
        try:
            scores = [r.relevance_score for r in result.results]
            rerank_span.end(
                metadata={
                    "n_returned": len(result.results),
                    "relevance_scores": scores,
                    "min_score": min(scores) if scores else None,
                    "max_score": max(scores) if scores else None,
                }
            )
        except Exception:
            pass

    return [documents[r.index] for r in result.results]


def stratified_select(
    ranked_chunks: list[dict],
    specific_categories: list[str],
    n_per_slot: int | None = None,
    fallback_per_course: int | None = None,
) -> list[dict]:
    """Pick top n_per_slot chunks per (course_id, category) slot from a pre-ranked list.

    Args:
        ranked_chunks:       Reranker-sorted chunks (best first).
        specific_categories: Categories requested by router. If non-empty, only
                             populate slots for these categories (ignore others).
                             If empty, fall back to top fallback_per_course per course_id.
        n_per_slot:          Max chunks per (course_id, category) pair.
                             Defaults to config.CHUNKS_PER_SLOT.
        fallback_per_course: Used when no specific categories — top N per unique course_id.
                             Defaults to config.STRATIFIED_FALLBACK_PER_COURSE.

    Returns:
        Selected chunks, preserving reranker order within slots.
    """
    n_per_slot = n_per_slot if n_per_slot is not None else config.CHUNKS_PER_SLOT
    fallback_per_course = (
        fallback_per_course if fallback_per_course is not None
        else config.STRATIFIED_FALLBACK_PER_COURSE
    )

    if not specific_categories:
        # No category constraint: top fallback_per_course per course_id
        counts: dict[str, int] = {}
        selected = []
        for chunk in ranked_chunks:
            cid = chunk.get("course_id", "")
            if counts.get(cid, 0) < fallback_per_course:
                counts[cid] = counts.get(cid, 0) + 1
                selected.append(chunk)
        return selected

    # Category-stratified: top n_per_slot per (course_id, category)
    slot_counts: dict[tuple[str, str], int] = {}
    selected = []
    for chunk in ranked_chunks:
        cat = chunk.get("category", "")
        cid = chunk.get("course_id", "")
        if cat not in specific_categories:
            continue
        key = (cid, cat)
        if slot_counts.get(key, 0) < n_per_slot:
            slot_counts[key] = slot_counts.get(key, 0) + 1
            selected.append(chunk)

    # If no chunks matched any category (e.g. untagged v3 chunks), fall back to per-course top-N
    if not selected:
        counts: dict[str, int] = {}
        for chunk in ranked_chunks:
            cid = chunk.get("course_id", "")
            if counts.get(cid, 0) < fallback_per_course:
                counts[cid] = counts.get(cid, 0) + 1
                selected.append(chunk)

    return selected


def rerank_multi_course(
    query: str,
    course_groups: dict[str, list[dict]],
    top_k_per_course: int = 3,
    parent_span=None,
) -> list[dict]:
    """Balanced reranking for multi-course comparison queries.

    Reranks each course's documents independently, then interleaves to ensure
    at least top_k_per_course results per course.

    Args:
        query:            The user's query.
        course_groups:    Mapping of course_id → list of chunk dicts.
        top_k_per_course: How many chunks to keep per course.
        parent_span:      Optional Langfuse span forwarded to each rerank() call.

    Returns:
        Interleaved list of reranked chunks with balanced representation.
    """
    if not course_groups:
        return []

    reranked_groups = {}
    for course_id, docs in course_groups.items():
        if docs:
            reranked_groups[course_id] = rerank(
                query, docs, top_k=top_k_per_course, parent_span=parent_span
            )
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
