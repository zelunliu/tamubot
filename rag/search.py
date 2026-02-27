"""Search functions over the MongoDB Atlas chunks/policies/courses collections.

Provides:
    hybrid_search   — $rankFusion of vector + text search
    search_semantic — pure vector search
    search_by_course — metadata filter lookup
    get_policy      — policy collection lookup
    aggregate_query — count/comparison aggregations
"""

from typing import Optional

import voyageai
from pymongo import MongoClient

import config

MONGODB_URI = config.MONGODB_URI
DB_NAME = config.MONGODB_DB
VOYAGE_API_KEY = config.VOYAGE_API_KEY

EMBEDDING_MODEL = "voyage-3"

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


def _build_vector_stage(
    query_embedding: list[float],
    k: int,
    filters: Optional[dict] = None,
) -> dict:
    """Build a $vectorSearch stage."""
    stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": k * 10,
            "limit": k,
        }
    }
    if filters:
        stage["$vectorSearch"]["filter"] = filters
    return stage


def _build_text_stage(
    query: str,
    k: int,
    filters: Optional[dict] = None,
) -> dict:
    """Build a $search stage for full-text BM25."""
    compound: dict = {
        "must": [
            {
                "text": {
                    "query": query,
                    "path": ["content", "title"],
                }
            }
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
            "index": "text_index",
            "compound": compound,
        }
    }


def _projection():
    """Standard projection excluding the raw embedding to reduce payload."""
    return {
        "$project": {
            "embedding": 0,
        }
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _rrf_fuse(ranked_lists: list[list[dict]], k_param: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Each doc is identified by its _id. Returns docs sorted by fused score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for result_list in ranked_lists:
        for rank, doc in enumerate(result_list):
            doc_id = str(doc["_id"])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_param + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[did] for did in sorted_ids]


def hybrid_search(
    query: str,
    filters: Optional[dict] = None,
    k: int = 5,
    parent_span=None,
) -> list[dict]:
    """Hybrid search using manual RRF of vector + text pipelines.

    Runs vector search and text search separately, then fuses with RRF.
    Compatible with M0 free tier (no $rankFusion needed).

    Args:
        query:       Natural-language question.
        filters:     Optional filter dict, e.g. {"course_id": "CSCE 120"}.
        k:           Number of results to return.
        parent_span: Optional Langfuse span; creates Voyage_Embeddings +
                     MongoDB_Hybrid_Search child spans underneath it.

    Returns:
        List of chunk documents (dicts) ranked by fused score.
    """
    db = _get_db()
    query_embedding = _embed_query(query, parent_span=parent_span)

    # Build atlas filters for vector search
    atlas_filters = None
    if filters:
        atlas_filters = {}
        for field, value in filters.items():
            if isinstance(value, list):
                atlas_filters[field] = {"$in": value}
            else:
                atlas_filters[field] = value

    # MongoDB search sub-span
    search_span = None
    if parent_span is not None:
        try:
            search_span = parent_span.span(
                name="MongoDB_Hybrid_Search",
                input={"query": query, "filters": filters, "k": k},
            )
        except Exception:
            search_span = None

    try:
        # Vector search pipeline
        vector_pipeline = [
            _build_vector_stage(query_embedding, k, atlas_filters),
            _projection(),
        ]
        vector_results = list(db["chunks"].aggregate(vector_pipeline))

        # Text search pipeline
        text_pipeline = [
            _build_text_stage(query, k, filters),
            {"$limit": k},
            _projection(),
        ]
        try:
            text_results = list(db["chunks"].aggregate(text_pipeline))
        except Exception:
            # Text index may not be ready yet — fall back to vector only
            text_results = []

        # Fuse and return top-k
        fused = _rrf_fuse([vector_results, text_results])
        result = fused[:k]
    except Exception as e:
        if search_span is not None:
            try:
                search_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    if search_span is not None:
        try:
            search_span.end(
                metadata={
                    "n_vector_results": len(vector_results),
                    "n_text_results": len(text_results),
                    "n_fused": len(result),
                }
            )
        except Exception:
            pass

    return result


def search_semantic(
    query: str,
    filters: Optional[dict] = None,
    top_k: int = 5,
) -> list[dict]:
    """Pure vector similarity search."""
    db = _get_db()
    query_embedding = _embed_query(query)

    atlas_filters = None
    if filters:
        atlas_filters = {}
        for field, value in filters.items():
            if isinstance(value, list):
                atlas_filters[field] = {"$in": value}
            else:
                atlas_filters[field] = value

    pipeline = [
        _build_vector_stage(query_embedding, top_k, atlas_filters),
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        _projection(),
    ]

    return list(db["chunks"].aggregate(pipeline))


def search_by_course(
    course_id: str,
    category: Optional[str] = None,
    term: Optional[str] = None,
) -> list[dict]:
    """Direct metadata lookup — no embedding needed."""
    db = _get_db()
    query: dict = {"course_id": course_id}
    if category:
        query["category"] = category
    if term:
        query["term"] = term

    return list(
        db["chunks"]
        .find(query, {"embedding": 0})
        .sort([("section", 1), ("chunk_index", 1)])
    )


def search_by_course_categories(
    course_id: str,
    categories: list[str],
    term: Optional[str] = None,
) -> list[dict]:
    """Direct metadata lookup by course_id and a list of categories — no embedding needed.

    Returns all matching chunks sorted by the provided category order, then by
    section and chunk_index within each category.

    Args:
        course_id:   Course ID to filter on (e.g. "CSCE 638").
        categories:  List of category strings to include (matched via $in).
        term:        Optional term filter.

    Returns:
        List of chunk documents (dicts) without embeddings.
    """
    if not categories:
        return []

    db = _get_db()
    query: dict = {"course_id": course_id, "category": {"$in": categories}}
    if term:
        query["term"] = term

    results = list(
        db["chunks"]
        .find(query, {"embedding": 0})
        .sort([("section", 1), ("chunk_index", 1)])
    )

    # Re-sort to preserve the requested category order
    category_order = {cat: i for i, cat in enumerate(categories)}
    results.sort(
        key=lambda d: (
            category_order.get(d.get("category", ""), len(categories)),
            d.get("chunk_index", 0),
        )
    )
    return results


def get_missing_sections(course_id: str) -> list[str]:
    """Return categories documented as missing from the original syllabus for a course.

    Derived by comparing VALID_CATEGORIES against categories_present stored in the
    courses collection (populated during ingestion from completeness_check data).
    Returns an empty list if the course is not found.
    """
    from rag.models import VALID_CATEGORIES
    db = _get_db()
    doc = db["courses"].find_one({"course_id": course_id}, {"categories_present": 1})
    if not doc:
        return []
    categories_present = set(doc.get("categories_present", []))
    return [c for c in VALID_CATEGORIES if c not in categories_present]


def get_policy(policy_name: str) -> Optional[dict]:
    """Look up a boilerplate policy by name (case-insensitive substring)."""
    db = _get_db()
    return db["policies"].find_one(
        {"policy_name": {"$regex": policy_name, "$options": "i"}}
    )


def multi_course_retrieve(
    query: str,
    course_ids: list[str],
    category: Optional[str] = None,
    k_per_course: int = 10,
    parent_span=None,
) -> list[dict]:
    """Parallel filtered hybrid searches for multi-course comparison queries.

    Runs one hybrid_search per course_id with filters, then combines all results.

    Args:
        query:       Search query (ideally rewritten by router).
        course_ids:  List of course IDs to retrieve for.
        category:    Optional category filter applied to all courses.
        k_per_course: How many candidates to retrieve per course.
        parent_span: Optional Langfuse span forwarded to each hybrid_search call.

    Returns:
        Combined list of chunk documents from all courses.
    """
    all_results = []
    for course_id in course_ids:
        filters: dict = {"course_id": course_id}
        if category:
            filters["category"] = category
        results = hybrid_search(query, filters=filters, k=k_per_course, parent_span=parent_span)
        all_results.extend(results)
    return all_results


def aggregate_query(
    category: str,
    course_id: Optional[str] = None,
    term: Optional[str] = None,
) -> list[dict]:
    """Aggregate query for comparisons (e.g. 'how many sections of CSCE 120?')."""
    db = _get_db()
    match: dict = {}
    if category:
        match["categories_present"] = category
    if course_id:
        match["course_id"] = course_id
    if term:
        match["term"] = term

    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": "$course_id",
                "sections": {"$push": "$section"},
                "instructors": {"$addToSet": "$instructor.name"},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    return list(db["courses"].aggregate(pipeline))
