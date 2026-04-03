"""Backward-compat shim — import from rag.tools.langfuse instead."""
from rag.tools.langfuse import *  # noqa: F401, F403
from rag.tools.langfuse import (
    get_langfuse, compute_ragas_metrics, run_ragas_background,
    MinimalLangfuseClient, LFGeneration, LFSpan,
)
