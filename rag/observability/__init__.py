"""rag.observability — unified observability package.

Replaces rag/tools/langfuse.py with a config-driven approach.
"""

from .config import (
    ObservabilityConfig,
    benchmark_config,
    chunking_config,
    probe_config,
    prod_config,
)
from .evals import EvalInputs, run_evals
from .tracing import create_trace, finalize_trace, get_langfuse

__all__ = [
    "ObservabilityConfig",
    "prod_config",
    "probe_config",
    "benchmark_config",
    "chunking_config",
    "create_trace",
    "finalize_trace",
    "get_langfuse",
    "EvalInputs",
    "run_evals",
]
