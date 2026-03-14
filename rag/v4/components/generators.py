"""Haystack-compatible generator component."""
from __future__ import annotations
from typing import Any, Callable, Iterator, Optional
from haystack import component


@component
class LLMGeneratorComponent:
    """Generates answers using a configurable streaming function.

    stream_fn defaults to rag.generator.generate_stream.
    prompt_builder defaults to rag.generator.build_system_prompt (if it exists).
    """

    def __init__(
        self,
        stream_fn: Optional[Callable] = None,
        prompt_builder: Optional[Callable] = None,
    ):
        self._stream_fn = stream_fn
        self._prompt_builder = prompt_builder

    def _get_stream_fn(self):
        if self._stream_fn is None:
            from rag.generator import generate_stream
            return generate_stream
        return self._stream_fn

    def _get_eval_fn(self):
        from rag.generator import generate_eval_search_string
        return generate_eval_search_string

    @component.output_types(answer_stream=object)
    def run(self, state: Any) -> dict:
        stream = self.generate_stream(state)
        return {"answer_stream": stream}

    def generate_stream(self, state: Any) -> Iterator[str]:
        """Satisfy GeneratorLLMComponent protocol."""
        stream_fn = self._get_stream_fn()
        router_result = state.get("router_result")
        return stream_fn(
            results=state.get("retrieved_chunks", []),
            question=state.get("rewritten_query") or state.get("query", ""),
            function=state.get("function", "semantic_general"),
            course_ids=state.get("course_ids", []),
            intent_type=state.get("intent_type"),
            specific_categories=state.get("specific_categories", []),
            specific_only=router_result.specific_only if router_result else False,
            data_gaps=state.get("data_gaps", []),
            data_integrity=state.get("data_integrity", True),
            conflicted_course_ids=state.get("conflicted_course_ids", []),
            trace=state.get("trace"),
        )

    def generate_eval_query(
        self, query: str, anchor_chunks: list[dict], trace: Optional[Any] = None
    ) -> str:
        """Satisfy GeneratorLLMComponent protocol."""
        eval_fn = self._get_eval_fn()
        return eval_fn(anchor_chunks, query, "GENERAL", parent_span=trace)
