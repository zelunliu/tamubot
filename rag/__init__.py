"""rag — RAG pipeline public API.

Import only from this module, not from submodules.
"""
from rag.graph.pipeline import get_current_state, run_pipeline, run_pipeline_with_memory
from rag.models import VALID_CATEGORIES, ChunkDoc, CourseDoc, PolicyDoc
from rag.state.pipeline_state import ConversationState, PipelineState, RouterResult
from rag.tools.langfuse import compute_ragas_metrics, get_langfuse, run_ragas_background

__all__ = [
    # Pipeline entry points
    "run_pipeline",
    "run_pipeline_with_memory",
    "get_current_state",
    # Data types
    "ChunkDoc", "CourseDoc", "PolicyDoc", "VALID_CATEGORIES",
    "PipelineState", "ConversationState", "RouterResult",
    # Observability
    "get_langfuse", "run_ragas_background", "compute_ragas_metrics",
]
