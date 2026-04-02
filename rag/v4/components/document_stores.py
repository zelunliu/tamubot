"""Haystack-compatible MongoDB document store implementing RetrieverComponent."""
from __future__ import annotations

import time
from typing import Any, Optional

import config
from rag.v4.trace_registry import current_span as _current_span

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

        parent = _current_span()
        embed_span = None
        t0 = time.perf_counter()
        if parent is not None:
            try:
                embed_span = parent.span(
                    name="embed.voyage",
                    input={"text_preview": text[:100]},
                )
            except Exception:
                embed_span = None

        try:
            result = self._embedder.run(text=text)
        except Exception:
            if embed_span is not None:
                try:
                    embed_span.end(metadata={
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "error": True,
                    })
                except Exception:
                    pass
            raise

        if embed_span is not None:
            try:
                embed_span.end(
                    output={"embedding_dim": len(result["embedding"])},
                    metadata={"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)},
                )
            except Exception:
                pass

        return result["embedding"]

    def _projection(self):
        # _id intentionally included: _rrf_fuse uses doc["_id"] as dedup key
        # Field names match actual chunks_v3 schema: content (text), header_text
        return {"$project": {
            "course_id": 1, "chunk_index": 1, "content": 1,
            "header_text": 1, "anchor": 1, "section": 1, "term": 1, "score": 1,
            "category": 1,
        }}

    def hybrid_search(
        self, query: str, course_id: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        from rag.search_v3 import _atlas_filter as _v3_atlas_filter
        from rag.search_v3 import _build_vector_stage

        db = self._get_db()
        emb = embedding or self._embed(query)

        # $vectorSearch with MQL-style filter (same as search_v3._build_vector_stage)
        atlas_f = _v3_atlas_filter(course_id, None)
        vector_stage = _build_vector_stage(emb, retrieve_k, atlas_f)
        vector_pipeline = [
            vector_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            self._projection(),
        ]

        # $search compound with Atlas Search equals filter (filter can't go inline in text)
        compound: dict = {"must": [{"text": {"query": query, "path": "content"}}]}
        if course_id:
            compound["filter"] = [{"equals": {"path": "course_id", "value": course_id}}]
        text_pipeline_stages = [
            {"$search": {"index": TEXT_INDEX, "compound": compound}},
            {"$limit": retrieve_k},
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            self._projection(),
        ]

        parent = _current_span()
        search_span = None
        t0 = time.perf_counter()
        if parent is not None:
            try:
                search_span = parent.span(
                    name="search.mongo_hybrid",
                    input={"course_id": course_id, "retrieve_k": retrieve_k},
                )
            except Exception:
                search_span = None

        try:
            vector_results = list(db[CHUNKS_COLLECTION].aggregate(vector_pipeline))
            _vector_err = None
        except Exception as e:
            vector_results = []
            _vector_err = str(e)

        try:
            text_results = list(db[CHUNKS_COLLECTION].aggregate(text_pipeline_stages))
            _text_err = None
        except Exception as e:
            text_results = []
            _text_err = str(e)

        if not vector_results and not text_results:
            if search_span is not None:
                try:
                    search_span.end(metadata={
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "error": True,
                    })
                except Exception:
                    pass
            errors = [e for e in [_vector_err, _text_err] if e]
            raise RuntimeError(f"hybrid_search returned no results: {'; '.join(errors)}")

        from rag.search_v3 import _rrf_fuse  # reuse existing RRF fusion
        results = _rrf_fuse([vector_results, text_results])[:retrieve_k]
        # Strip ObjectId — not msgpack-serializable by LangGraph checkpointer
        for r in results:
            r.pop("_id", None)

        if search_span is not None:
            try:
                search_span.end(
                    output={
                        "n_vector": len(vector_results),
                        "n_text": len(text_results),
                        "n_fused": len(results),
                    },
                    metadata={"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)},
                )
            except Exception:
                pass

        return results

    def semantic_search(
        self, query: str, retrieve_k: int, embedding: Optional[list[float]] = None
    ) -> list[dict]:
        from rag.search_v3 import _build_vector_stage

        db = self._get_db()
        emb = embedding or self._embed(query)

        pipeline = [
            _build_vector_stage(emb, retrieve_k, filters=None),
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            self._projection(),
        ]

        parent = _current_span()
        search_span = None
        t0 = time.perf_counter()
        if parent is not None:
            try:
                search_span = parent.span(
                    name="search.mongo_semantic",
                    input={"retrieve_k": retrieve_k},
                )
            except Exception:
                search_span = None

        results = list(db[CHUNKS_COLLECTION].aggregate(pipeline))
        for r in results:
            r.pop("_id", None)

        if search_span is not None:
            try:
                search_span.end(
                    output={"n_results": len(results)},
                    metadata={"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)},
                )
            except Exception:
                pass

        return results

    def fetch_anchor_chunks(
        self, course_ids: list[str], categories: list[str]
    ) -> tuple[list[dict], list[tuple[str, str]], bool]:
        from rag.search_v3 import fetch_anchor_chunks as v3_fetch
        chunks, gaps, integrity = v3_fetch(course_ids, categories)
        for c in chunks:
            c.pop("_id", None)
        return chunks, gaps, integrity

    def get_meeting_times(self, course_ids: list[str]) -> dict[str, Any]:
        from rag.search_v3 import get_meeting_times as v3_get_mt
        return v3_get_mt(course_ids)

    def get_syllabus_urls(self, course_ids: list[str]) -> dict[str, str]:
        from rag.search_v3 import get_syllabus_urls as v3_get_urls
        return v3_get_urls(course_ids)
