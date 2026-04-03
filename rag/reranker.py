"""Backward-compat shim — import from rag.tools.voyage instead."""
from rag.tools.voyage import rerank, stratified_select


def rerank_multi_course(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Multi-course rerank — delegates to rerank()."""
    return rerank(query, chunks, top_k)
