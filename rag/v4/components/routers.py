"""Haystack-compatible router component."""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from haystack import component

import rag.router as _router_mod
from rag.router import classify_query

_call_llm_lock = threading.Lock()


@component
class LLMRouterComponent:
    """Classifies queries using a configurable LLM function.

    The llm_fn defaults to rag.llm_client.call_llm but can be any callable
    matching the signature — this is the flexibility unlock: swap Gemini for
    any LLM with one line.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        prompt_template: Optional[str] = None,
    ):
        self._llm_fn = llm_fn
        self._prompt_template = prompt_template

    def _get_llm_fn(self):
        if self._llm_fn is None:
            from rag.llm_client import call_llm
            return call_llm
        return self._llm_fn

    @component.output_types(router_result=object)
    def run(self, query: str, trace: Optional[Any] = None) -> dict:
        if self._llm_fn is not None:
            # Injected llm_fn — temporarily replace call_llm in rag.router so
            # classify_query uses it. Protected by a module-level lock to prevent
            # race conditions under concurrent requests.
            with _call_llm_lock:
                _original = _router_mod.call_llm
                try:
                    _router_mod.call_llm = self._llm_fn
                    router_result = classify_query(query, router_span=trace)
                finally:
                    _router_mod.call_llm = _original
        else:
            router_result = classify_query(query, router_span=trace)
        return {"router_result": router_result}

    def classify(self, query: str, trace: Optional[Any] = None) -> Any:
        """Satisfy RouterLLMComponent protocol."""
        return self.run(query=query, trace=trace)["router_result"]
