# Public API — evals and app.py import from here, not from submodules
from rag.generator import generate, generate_comparison, generate_stream
from rag.models import VALID_CATEGORIES, ChunkDoc, CourseDoc, PolicyDoc
from rag.observability import compute_ragas_metrics, get_langfuse, run_ragas_background

# v4 is now the default pipeline — import run_pipeline from v4
from rag.v4.pipeline_v4 import run_pipeline_v4 as run_pipeline  # noqa: F401
# v3 orchestration helpers kept for rollback compat (still used by app.py Vertex path)
from rag.v3_legacy.pipeline import db_order, generator_order, router_order  # noqa: F401
from rag.reranker import rerank, rerank_multi_course, stratified_select
from rag.router import (
    FUNCTION_CATEGORY_STRATEGIES,
    RouterResult,
    classify_query,
    compute_dynamic_k,
    deduplicate_chunks,
    route_retrieve_rerank,
)
from rag.search import (
    fetch_anchor_chunks,
    get_missing_sections,
    hybrid_search,
    search_by_course_categories,
    search_semantic,
)

__all__ = [
    "ChunkDoc", "CourseDoc", "PolicyDoc", "VALID_CATEGORIES",
    "route_retrieve_rerank", "classify_query", "RouterResult",
    "compute_dynamic_k", "deduplicate_chunks", "FUNCTION_CATEGORY_STRATEGIES",
    "generate", "generate_stream", "generate_comparison",
    "hybrid_search", "search_semantic", "search_by_course_categories", "get_missing_sections", "fetch_anchor_chunks",
    "rerank", "rerank_multi_course", "stratified_select",
    "get_langfuse", "run_ragas_background", "compute_ragas_metrics",
    "run_pipeline", "router_order", "db_order", "generator_order",
]
