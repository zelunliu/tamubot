# Public API — evals and app.py import from here, not from submodules
from rag.generator import generate, generate_comparison, generate_stream, generator_order
from rag.models import VALID_CATEGORIES, ChunkDoc, CourseDoc, PolicyDoc
from rag.observability import compute_ragas_metrics, get_langfuse, run_ragas_background
from rag.reranker import rerank, rerank_multi_course, stratified_select
from rag.router import (
    FUNCTION_CATEGORY_STRATEGIES,
    RouterResult,
    classify_query,
    compute_dynamic_k,
    deduplicate_chunks,
    route_retrieve_rerank,
    router_order,
)
from rag.search import (
    fetch_anchor_chunks,
    get_missing_sections,
    hybrid_search,
    search_by_course_categories,
    search_semantic,
)
from rag.v4.pipeline_v4 import run_pipeline_v4 as run_pipeline  # noqa: F401

__all__ = [
    "ChunkDoc", "CourseDoc", "PolicyDoc", "VALID_CATEGORIES",
    "route_retrieve_rerank", "classify_query", "RouterResult",
    "compute_dynamic_k", "deduplicate_chunks", "FUNCTION_CATEGORY_STRATEGIES",
    "generate", "generate_stream", "generate_comparison",
    "hybrid_search", "search_semantic", "search_by_course_categories", "get_missing_sections", "fetch_anchor_chunks",
    "rerank", "rerank_multi_course", "stratified_select",
    "get_langfuse", "run_ragas_background", "compute_ragas_metrics",
    "run_pipeline", "router_order", "generator_order",
]
