"""Haystack-compatible MongoDB document store implementing RetrieverComponent."""
from __future__ import annotations
from typing import Any, Optional
import config

CHUNKS_COLLECTION = "chunks_v3"
COURSES_COLLECTION = "courses_v3"
VECTOR_INDEX = "vector_index_v3"
TEXT_INDEX = "text_index_v3"


class MongoDocumentStore:
    """Implements RetrieverComponent protocol over MongoDB Atlas chunks_v3.

    Returns plain dict (not Haystack Document) to preserve v3 chunk schema compatibility.
    Index names read from constants matching rag/search_v3.py.
    """

    def __init__(self, mongo_client=None):
        self._client = mongo_client
        self._embedder = None

    def _get_db(self):
        if self._client is None:
            from pymongo import MongoClient
            self._client = MongoClient(config.MONGODB_URI)
        return self._client[config.MONGODB_DB]

    def set_embedder(self, embedder):
        """Inject embedder (VoyageEmbedder or NullEmbedder)."""
        self._embedder = embedder

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            raise RuntimeError("No embedder set on MongoDocumentStore")
        result = self._embedder.run(text=text)
        return result["embedding"]

    def _projection(self):
        return {"$project": {
            "_id": 0, "course_id": 1, "chunk_index": 1, "text": 1,
            "category": 1, "header": 1, "score": 1,
        }}

    def _atlas_filter(self, course_id: Optional[str] = None) -> Optional[dict]:
        if course_id:
            return {"course_id": {"$eq": course_id}}
        return None

    def hybrid_search(
        self, query: str, course_id: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        db = self._get_db()
        emb = embedding or self._embed(query)
        atlas_f = self._atlas_filter(course_id)

        vector_pipeline = [
            {"$search": {
                "index": VECTOR_INDEX,
                "knnBeta": {"vector": emb, "path": "embedding", "k": retrieve_k,
                            **({"filter": atlas_f} if atlas_f else {})},
            }},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            self._projection(),
        ]

        text_pipeline_stages = [
            {"$search": {
                "index": TEXT_INDEX,
                "text": {"query": query, "path": "text",
                         **({"filter": atlas_f} if atlas_f else {})},
            }},
            {"$limit": retrieve_k},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            self._projection(),
        ]

        try:
            vector_results = list(db[CHUNKS_COLLECTION].aggregate(vector_pipeline))
        except Exception:
            vector_results = []

        try:
            text_results = list(db[CHUNKS_COLLECTION].aggregate(text_pipeline_stages))
        except Exception:
            text_results = []

        from rag.search_v3 import _rrf_fuse  # reuse existing RRF fusion
        return _rrf_fuse([vector_results, text_results])[:retrieve_k]

    def semantic_search(
        self, query: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        db = self._get_db()
        emb = embedding or self._embed(query)

        pipeline = [
            {"$search": {
                "index": VECTOR_INDEX,
                "knnBeta": {"vector": emb, "path": "embedding", "k": retrieve_k},
            }},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            self._projection(),
        ]
        return list(db[CHUNKS_COLLECTION].aggregate(pipeline))

    def fetch_anchor_chunks(
        self, course_ids: list[str], categories: list[str]
    ) -> tuple[list[dict], list[tuple[str, str]], bool]:
        from rag.search_v3 import fetch_anchor_chunks as v3_fetch
        return v3_fetch(course_ids, categories)

    def get_meeting_times(self, course_ids: list[str]) -> dict[str, Any]:
        from rag.search_v3 import get_meeting_times as v3_get_mt
        return v3_get_mt(course_ids)

    def get_syllabus_urls(self, course_ids: list[str]) -> dict[str, str]:
        from rag.search_v3 import get_syllabus_urls as v3_get_urls
        return v3_get_urls(course_ids)
