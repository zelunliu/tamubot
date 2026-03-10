"""Search functions over the V2 MongoDB Atlas collections (chunks_v2, courses_v2).

Four search methods:
    V2A  hybrid_search_v2      — RRF: vector + BM25 on content+header_text
    V2B  vector_search_v2      — Pure semantic vector search
    V2C  header_cluster_v2     — Top-20 unfiltered → group/merge by header_text
    V2D  bm25_search_v2        — BM25 only, header_text 2x boost

All methods accept an optional course_id filter (applied for course-specific queries).
V2C is the only corpus-wide method; V2A/B/D all filter to the relevant course.
"""

from collections import defaultdict
from typing import Optional

import voyageai
from pymongo import MongoClient

import config

MONGODB_URI = config.MONGODB_URI
DB_NAME = config.MONGODB_DB
VOYAGE_API_KEY = config.VOYAGE_API_KEY

EMBEDDING_MODEL = "voyage-3"
CHUNKS_COLLECTION = "chunks_v2"

_client: Optional[MongoClient] = None
_voyage: Optional[voyageai.Client] = None


def _get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client[DB_NAME]


def _get_voyage():
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _voyage


def _embed_query(query: str) -> list[float]:
    client = _get_voyage()
    result = client.embed([query], model=EMBEDDING_MODEL, input_type="query")
    return result.embeddings[0]


def _atlas_filter(course_id: str | None, term: str | None) -> dict | None:
    """Build Atlas vector-search filter dict."""
    f: dict = {}
    if course_id:
        f["course_id"] = course_id
    if term:
        f["term"] = term
    return f if f else None


def _mongo_filter(course_id: str | None, term: str | None) -> dict:
    """Build standard MongoDB filter dict."""
    f: dict = {}
    if course_id:
        f["course_id"] = course_id
    if term:
        f["term"] = term
    return f


def _projection() -> dict:
    return {"$project": {"embedding": 0}}


def _rrf_fuse(ranked_lists: list[list[dict]], k_param: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked result lists."""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for result_list in ranked_lists:
        for rank, doc in enumerate(result_list):
            doc_id = str(doc["_id"])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_param + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    return [doc_map[did] for did in sorted(scores, key=lambda x: scores[x], reverse=True)]


def _build_vector_stage(embedding: list[float], k: int, filters: dict | None) -> dict:
    stage: dict = {
        "$vectorSearch": {
            "index": "vector_index_v2",
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": k * 10,
            "limit": k,
        }
    }
    if filters:
        stage["$vectorSearch"]["filter"] = filters
    return stage


def _build_text_stage(query: str, k: int, course_id: str | None, term: str | None, header_boost: bool = False) -> dict:
    """Build $search stage. Optionally boosts header_text matches 2x."""
    should_clauses: list[dict] = [
        {"text": {"query": query, "path": "content"}},
    ]
    if header_boost:
        should_clauses.append(
            {"text": {"query": query, "path": "header_text", "score": {"boost": {"value": 2}}}}
        )
    else:
        should_clauses.append({"text": {"query": query, "path": "header_text"}})

    compound: dict = {"should": should_clauses, "minimumShouldMatch": 1}

    filter_clauses = []
    if course_id:
        filter_clauses.append({"equals": {"path": "course_id", "value": course_id}})
    if term:
        filter_clauses.append({"equals": {"path": "term", "value": term}})
    if filter_clauses:
        compound["filter"] = filter_clauses

    return {
        "$search": {
            "index": "text_index_v2",
            "compound": compound,
        }
    }


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_search_v2(
    query: str,
    course_id: str | None = None,
    term: str | None = None,
    k: int = 5,
) -> list[dict]:
    """V2A: RRF fusion of vector + BM25 on content+header_text.

    Filtered by course_id when provided (course-specific queries).
    """
    db = _get_db()
    embedding = _embed_query(query)
    atlas_f = _atlas_filter(course_id, term)

    vector_pipeline = [
        _build_vector_stage(embedding, k, atlas_f),
        _projection(),
    ]
    vector_results = list(db[CHUNKS_COLLECTION].aggregate(vector_pipeline))

    try:
        text_pipeline = [
            _build_text_stage(query, k, course_id, term, header_boost=False),
            {"$limit": k},
            _projection(),
        ]
        text_results = list(db[CHUNKS_COLLECTION].aggregate(text_pipeline))
    except Exception:
        text_results = []

    return _rrf_fuse([vector_results, text_results])[:k]


def vector_search_v2(
    query: str,
    course_id: str | None = None,
    term: str | None = None,
    k: int = 5,
) -> list[dict]:
    """V2B: Pure semantic vector search, filtered by course_id."""
    db = _get_db()
    embedding = _embed_query(query)
    atlas_f = _atlas_filter(course_id, term)

    pipeline = [
        _build_vector_stage(embedding, k, atlas_f),
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        _projection(),
    ]
    return list(db[CHUNKS_COLLECTION].aggregate(pipeline))


def header_cluster_v2(
    query: str,
    k: int = 5,
    candidates: int = 20,
) -> list[dict]:
    """V2C: Corpus-wide vector search, then group/merge by header_text.

    Returns one merged result per unique header_text (up to k groups).
    This is the only method that does NOT filter by course_id.
    """
    db = _get_db()
    embedding = _embed_query(query)

    pipeline = [
        _build_vector_stage(embedding, candidates, None),
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        _projection(),
    ]
    raw = list(db[CHUNKS_COLLECTION].aggregate(pipeline))

    # Group by header_text (None → "")
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in raw:
        key = (doc.get("header_text") or "").strip().lower()
        groups[key].append(doc)

    # Merge each group: concatenate content, take top score
    merged: list[dict] = []
    for key, docs in groups.items():
        top = max(docs, key=lambda d: d.get("score", 0))
        combined = top.copy()
        if len(docs) > 1:
            combined["content"] = "\n\n---\n\n".join(d["content"] for d in docs)
            combined["_merged_from"] = len(docs)
        merged.append(combined)

    merged.sort(key=lambda d: d.get("score", 0), reverse=True)
    return merged[:k]


def bm25_search_v2(
    query: str,
    course_id: str | None = None,
    term: str | None = None,
    k: int = 5,
) -> list[dict]:
    """V2D: BM25-only text search with header_text 2x boost, filtered by course_id."""
    db = _get_db()

    pipeline = [
        _build_text_stage(query, k, course_id, term, header_boost=True),
        {"$limit": k},
        _projection(),
    ]
    try:
        return list(db[CHUNKS_COLLECTION].aggregate(pipeline))
    except Exception as e:
        raise RuntimeError(f"V2D BM25 search failed (is text_index_v2 built?): {e}") from e
