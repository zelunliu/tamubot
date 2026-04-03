"""Backward-compat shim — import from rag.tools.context instead."""
from rag.tools.context import *  # noqa: F401, F403
from rag.tools.context import format_context_xml, collapse_whitespace, strip_thinking_blocks
