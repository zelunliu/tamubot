"""V4Tracer — wraps MinimalLangfuseClient for per-node span emission.

Uses rag.observability.MinimalLangfuseClient but never replaces it.
If lf_client=None, all methods are no-ops (context manager returns None).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Optional

from rag.v4.state import PipelineState


class _NullSpan:
    """No-op span for when lf_client is None."""
    def end(self, **kwargs):
        pass


class V4Tracer:
    """Emits Langfuse spans for each v4 node transition.

    Wraps MinimalLangfuseClient. lf_client=None → all methods are no-ops.
    """

    def __init__(self, lf_client: Optional[Any] = None):
        self._lf = lf_client

    @contextmanager
    def node_span(self, node_name: str, state: PipelineState):
        """Context manager: opens a Langfuse span for node_name, ends it on exit."""
        span = None
        if self._lf is not None:
            try:
                span = self._lf.span(
                    name=f"v4_{node_name}",
                    input={
                        "function": state.get("function"),
                        "course_ids": state.get("course_ids", []),
                        "query": state.get("query", "")[:100],
                    },
                )
            except Exception:
                span = None

        t_start = time.perf_counter()
        try:
            yield span
        finally:
            elapsed = round((time.perf_counter() - t_start) * 1000, 1)
            if span is not None:
                try:
                    span.end(metadata={"elapsed_ms": elapsed})
                except Exception:
                    pass

    def trace_node_transition(
        self, from_node: str, to_node: str, state: PipelineState
    ) -> None:
        """Record a node transition in Langfuse metadata (best-effort)."""
        if self._lf is None:
            return
        try:
            # Record as a span event on the trace
            span = self._lf.span(
                name=f"v4_transition_{from_node}_to_{to_node}",
                input={"node_trace": state.get("node_trace", [])},
            )
            span.end()
        except Exception:
            pass
