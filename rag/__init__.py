"""rag — RAG pipeline public API.

Import only from this module, not from submodules.
"""
from rag.graph.pipeline import get_current_state, run_pipeline, run_pipeline_with_memory
from rag.models import VALID_CATEGORIES, ChunkDoc, CourseDoc, PolicyDoc
from rag.observability import get_langfuse
from rag.router import RouterResult
from rag.state.pipeline_state import ConversationState, PipelineState

__all__ = [
    # Pipeline entry points
    "run_pipeline",
    "run_pipeline_with_memory",
    "get_current_state",
    # Data types
    "ChunkDoc", "CourseDoc", "PolicyDoc", "VALID_CATEGORIES",
    "PipelineState", "ConversationState", "RouterResult",
    # Observability
    "get_langfuse",
]
