# Public API — evals and app.py import from here, not from submodules
from rag.models import ChunkDoc, CourseDoc, PolicyDoc, VALID_CATEGORIES
from rag.router import (
    route_retrieve_rerank,
    classify_query,
    RouterResult,
    compute_dynamic_k,
    deduplicate_chunks,
    FUNCTION_CATEGORY_STRATEGIES,
)
from rag.generator import generate, generate_stream, generate_comparison
from rag.search import hybrid_search, search_semantic, search_by_course_categories, get_missing_sections, fetch_anchor_chunks
from rag.reranker import rerank, rerank_multi_course
from rag.observability import get_langfuse, run_ragas_background, compute_ragas_metrics
# router must be fully initialized before pipeline imports
from rag.pipeline import run_pipeline, router_order, db_order, generator_order

__all__ = [
    "ChunkDoc", "CourseDoc", "PolicyDoc", "VALID_CATEGORIES",
    "route_retrieve_rerank", "classify_query", "RouterResult",
    "compute_dynamic_k", "deduplicate_chunks", "FUNCTION_CATEGORY_STRATEGIES",
    "generate", "generate_stream", "generate_comparison",
    "hybrid_search", "search_semantic", "search_by_course_categories", "get_missing_sections", "fetch_anchor_chunks",
    "rerank", "rerank_multi_course",
    "get_langfuse", "run_ragas_background", "compute_ragas_metrics",
    "run_pipeline", "router_order", "db_order", "generator_order",
]
