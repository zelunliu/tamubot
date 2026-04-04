"""Custom exceptions for the v4 pipeline.

Canonical location: rag/graph/exceptions.py
"""


class V4PipelineError(Exception):
    """Base exception for v4 pipeline errors."""


class V4RouterError(V4PipelineError):
    """Raised when the router node fails to classify a query."""


class V4RetrievalError(V4PipelineError):
    """Raised when retrieval fails."""


class V4GenerationError(V4PipelineError):
    """Raised when generation fails."""
