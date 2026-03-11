"""Search functions over the V3 MongoDB Atlas collections (chunks_v3, courses_v3).

V3 uses flat token chunks (~600 tokens each) with no header or category fields.

Three search methods:
    V3A  hybrid_search_v3  — RRF: vector + BM25 on content only
    V3B  vector_search_v3  — Pure semantic vector search

All methods accept optional course_id / term filters.

Also exposes a v1-compatible API so pipeline.py needs no logic changes:
    hybrid_search(query, filters, k, parent_span)
    search_semantic(query, top_k)
    fetch_anchor_chunks(course_ids, categories)   — ignores categories; all chunks per course
    search_by_course(course_id, term)
    get_missing_sections(course_id)
"""

from typing import Optional

import voyageai
from pymongo import MongoClient

import config

MONGODB_URI = config.MONGODB_URI
DB_NAME = config.MONGODB_DB
VOYAGE_API_KEY = config.VOYAGE_API_KEY

EMBEDDING_MODEL = "voyage-3"
CHUNKS_COLLECTION = "chunks_v3"
COURSES_COLLECTION = "courses_v3"

# ⚠️ Confirm these names match the Atlas indexes on chunks_v3 before running.
VECTOR_INDEX = "vector_index_v3"
TEXT_INDEX = "text_index_v3"

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


def _embed_query(query: str, parent_span=None) -> list[float]:
    """Embed a query string using Voyage AI, optionally traced under parent_span."""
    embed_span = None
    if parent_span is not None:
        try:
            embed_span = parent_span.span(
                name="Voyage_Embeddings",
                input={"query": query, "model": EMBEDDING_MODEL},
            )
        except Exception:
            embed_span = None

    try:
        client = _get_voyage()
        result = client.embed([query], model=EMBEDDING_MODEL, input_type="query")
        embedding = result.embeddings[0]
    except Exception as e:
        if embed_span is not None:
            try:
                embed_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    if embed_span is not None:
        try:
            embed_span.end(metadata={"embedding_dim": len(embedding)})
        except Exception:
            pass

    return embedding


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
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": k * 10,
            "limit": k,
        }
    }
    if filters:
        stage["$vectorSearch"]["filter"] = filters
    return stage


def _build_text_stage(query: str, k: int, filters: Optional[dict] = None) -> dict:
    """Build $search stage for BM25 on content field."""
    compound: dict = {
        "must": [
            {"text": {"query": query, "path": "content"}},
        ]
    }

    if filters:
        filter_clauses = []
        for field, value in filters.items():
            if isinstance(value, list):
                filter_clauses.append({"in": {"path": field, "value": value}})
            else:
                filter_clauses.append({"equals": {"path": field, "value": value}})
        compound["filter"] = filter_clauses

    return {
        "$search": {
            "index": TEXT_INDEX,
            "compound": compound,
        }
    }


def _build_text_stage_course(query: str, k: int, course_id: str | None, term: str | None) -> dict:
    """Build $search stage with inline filter clauses (for course-specific search)."""
    compound: dict = {
        "should": [{"text": {"query": query, "path": "content"}}],
        "minimumShouldMatch": 1,
    }

    filter_clauses = []
    if course_id:
        filter_clauses.append({"equals": {"path": "course_id", "value": course_id}})
    if term:
        filter_clauses.append({"equals": {"path": "term", "value": term}})
    if filter_clauses:
        compound["filter"] = filter_clauses

    return {
        "$search": {
            "index": TEXT_INDEX,
            "compound": compound,
        }
    }


# ── V3 native API ──────────────────────────────────────────────────────────────

def hybrid_search_v3(
    query: str,
    course_id: str | None = None,
    term: str | None = None,
    k: int = 5,
    parent_span=None,
) -> list[dict]:
    """V3A: RRF fusion of vector + BM25 on content field."""
    db = _get_db()
    embedding = _embed_query(query, parent_span=parent_span)
    atlas_f = _atlas_filter(course_id, term)

    search_span = None
    if parent_span is not None:
        try:
            search_span = parent_span.span(
                name="MongoDB_Hybrid_Search",
                input={"query": query, "course_id": course_id, "k": k},
            )
        except Exception:
            search_span = None

    try:
        vector_pipeline = [
            _build_vector_stage(embedding, k, atlas_f),
            _projection(),
        ]
        vector_results = list(db[CHUNKS_COLLECTION].aggregate(vector_pipeline))

        try:
            text_pipeline = [
                _build_text_stage_course(query, k, course_id, term),
                {"$limit": k},
                _projection(),
            ]
            text_results = list(db[CHUNKS_COLLECTION].aggregate(text_pipeline))
        except Exception:
            text_results = []

        result = _rrf_fuse([vector_results, text_results])[:k]
    except Exception as e:
        if search_span is not None:
            try:
                search_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    if search_span is not None:
        try:
            search_span.end(metadata={
                "n_vector_results": len(vector_results),
                "n_text_results": len(text_results),
                "n_fused": len(result),
            })
        except Exception:
            pass

    return result


def vector_search_v3(
    query: str,
    course_id: str | None = None,
    term: str | None = None,
    k: int = 5,
) -> list[dict]:
    """V3B: Pure semantic vector search, optionally filtered by course_id."""
    db = _get_db()
    embedding = _embed_query(query)
    atlas_f = _atlas_filter(course_id, term)

    pipeline = [
        _build_vector_stage(embedding, k, atlas_f),
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        _projection(),
    ]
    return list(db[CHUNKS_COLLECTION].aggregate(pipeline))


# ── V1-compatible API (for pipeline.py with no logic changes) ──────────────────

def hybrid_search(
    query: str,
    filters: Optional[dict] = None,
    k: int = 5,
    parent_span=None,
) -> list[dict]:
    """V1-compatible wrapper around hybrid_search_v3.

    Accepts a filters dict (e.g. {"course_id": "CSCE 638"}) and delegates
    to hybrid_search_v3.
    """
    course_id = filters.get("course_id") if filters else None
    term = filters.get("term") if filters else None
    # Handle list-valued course_id (multi-course filters not supported natively;
    # fall back to unfiltered search and let the caller filter post-hoc)
    if isinstance(course_id, list):
        course_id = None
    return hybrid_search_v3(query, course_id=course_id, term=term, k=k, parent_span=parent_span)


def search_semantic(
    query: str,
    filters: Optional[dict] = None,
    top_k: int = 5,
    parent_span=None,
) -> list[dict]:
    """V1-compatible corpus-wide vector search (no course filter)."""
    db = _get_db()
    embedding = _embed_query(query, parent_span=parent_span)

    atlas_filters = None
    if filters:
        atlas_filters = {}
        for field, value in filters.items():
            if isinstance(value, list):
                atlas_filters[field] = {"$in": value}
            else:
                atlas_filters[field] = value

    search_span = None
    if parent_span is not None:
        try:
            search_span = parent_span.span(
                name="MongoDB_Vector_Search",
                input={"query": query, "top_k": top_k},
            )
        except Exception:
            search_span = None

    try:
        pipeline = [
            _build_vector_stage(embedding, top_k, atlas_filters),
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            _projection(),
        ]
        results = list(db[CHUNKS_COLLECTION].aggregate(pipeline))
    except Exception as e:
        if search_span is not None:
            try:
                search_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    if search_span is not None:
        try:
            search_span.end(metadata={"n_results": len(results)})
        except Exception:
            pass

    return results


def search_by_course(
    course_id: str,
    term: Optional[str] = None,
) -> list[dict]:
    """All chunks for a course, sorted by chunk_index."""
    db = _get_db()
    query: dict = {"course_id": course_id}
    if term:
        query["term"] = term
    return list(
        db[CHUNKS_COLLECTION]
        .find(query, {"embedding": 0})
        .sort([("section", 1), ("chunk_index", 1)])
    )


def fetch_anchor_chunks(
    course_ids: list[str],
    categories: list[str],  # ignored in v3 — no category field
) -> tuple[list[dict], list[tuple[str, str]], bool]:
    """Fetch all chunks for each course_id from chunks_v3.

    The `categories` parameter is accepted for API compatibility but ignored —
    v3 chunks have no category field.  Data gaps are tracked at course granularity
    (did we find *any* chunks for this course?).

    Args:
        course_ids:  Anchor course IDs (e.g. ["CSCE 638"]).
        categories:  Ignored (kept for v1 API compatibility).

    Returns:
        chunks         — all retrieved chunks sorted by (course_id, chunk_index)
        data_gaps      — [(course_id, ""), ...] for courses with zero chunks found
        data_integrity — True if len(data_gaps) == 0
    """
    chunks: list[dict] = []
    data_gaps: list[tuple[str, str]] = []

    for course_id in course_ids:
        results = search_by_course(course_id)
        if results:
            chunks.extend(results)
        else:
            data_gaps.append((course_id, ""))

    integrity = len(data_gaps) == 0
    return chunks, data_gaps, integrity


def get_syllabus_urls(course_ids: list[str]) -> dict[str, str]:
    """Return a mapping of course_id → syllabus_url for the given course IDs.

    Looks up the courses_v3 collection. Courses without a URL are omitted.
    """
    db = _get_db()
    docs = db[COURSES_COLLECTION].find(
        {"course_id": {"$in": course_ids}},
        {"course_id": 1, "syllabus_url": 1},
    )
    return {
        d["course_id"]: d["syllabus_url"]
        for d in docs
        if d.get("syllabus_url")
    }


def get_missing_sections(course_id: str) -> list[str]:
    """Return category names documented as missing for a course.

    In v3 the courses collection has no categories_present field — returns
    an empty list (no per-category completeness data).
    """
    db = _get_db()
    doc = db[COURSES_COLLECTION].find_one({"course_id": course_id}, {"chunk_count": 1})
    if not doc:
        return []
    # No per-category completeness tracking in v3
    return []
