"""Langfuse singleton + trace lifecycle helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import ObservabilityConfig

logger = logging.getLogger("tamubot.observability")

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_langfuse_client = None


def get_langfuse():
    """Lazy singleton. Returns None if Langfuse credentials are not configured."""
    global _langfuse_client
    if _langfuse_client is None:
        import config

        if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
            return None
        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_BASE_URL,
                flush_interval=0.5,
            )
            logger.info("Langfuse SDK client initialised.")
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e}")
            return None
    return _langfuse_client


# ---------------------------------------------------------------------------
# Trace lifecycle
# ---------------------------------------------------------------------------


def create_trace(
    obs_config: ObservabilityConfig,
    query: str,
) -> tuple[Optional[object], Optional[str]]:
    """Create a Langfuse root observation (trace). Returns (span, trace_id) or (None, None).

    Uses start_observation() which is the v4 SDK's way to create root traces.
    Tags and session_id are stored in metadata since the v4 SDK does not expose
    trace-level session_id on start_observation.
    """
    lf = get_langfuse()
    if lf is None:
        return None, None
    try:
        merged_meta = {
            **(obs_config.metadata or {}),
            "session_id": obs_config.session_id,
        }
        span = lf.start_observation(
            name=obs_config.trace_name,
            input=query,
            metadata=merged_meta,
        )
        trace_id = span.trace_id
        # Set tags on the trace via SDK internal API (only public way in v4)
        if obs_config.tags:
            try:
                lf._create_trace_tags_via_ingestion(
                    trace_id=trace_id, tags=obs_config.tags,
                )
            except Exception:
                pass  # tags are nice-to-have, not critical
        return span, trace_id
    except Exception as e:
        logger.warning(f"Langfuse trace creation failed: {e}")
        return None, None


def finalize_trace(trace, output: str) -> None:
    """Update trace output, end span, and flush. No-op if trace is None."""
    if trace is None:
        return
    try:
        trace.update(output=output)
        trace.end()
    except Exception:
        pass
    lf = get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass
