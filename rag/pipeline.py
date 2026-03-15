"""v3 pipeline stub — real code has moved to rag.v3_legacy.pipeline.

This stub re-exports everything from rag.v3_legacy.pipeline for rollback
compatibility. New code should call rag.v4.pipeline_v4.run_pipeline_v4 directly.
"""
import warnings

warnings.warn(
    "rag.pipeline is deprecated. Use rag.v4.pipeline_v4.run_pipeline_v4 instead. "
    "This module will be moved to rag.v3_legacy in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from rag.v3_legacy.pipeline import *  # noqa: F401, F403
from rag.v3_legacy.pipeline import (  # noqa: F401
    _apply_schedule_filter,
    _cap_discovery_courses,
    db_order,
    generator_order,
    router_order,
    run_pipeline,
)
