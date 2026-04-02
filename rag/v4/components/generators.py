"""Haystack-compatible generator component."""
from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

from haystack import component

from rag.llm_client import call_llm


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
        from rag.v4.trace_registry import current_span as _current_span, get as _get_trace
        stream_fn = self._get_stream_fn()
        router_result = state.get("router_result")
        # Prefer current node span (for nesting), fall back to root trace or state trace
        trace = _current_span() or _get_trace(state.get("session_id", "")) or state.get("trace")
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
            trace=trace,
            history_context=state.get("history_context"),
        )

    def generate_eval_query(
        self, query: str, anchor_chunks: list[dict], trace: Optional[Any] = None
    ) -> str:
        """Satisfy GeneratorLLMComponent protocol."""
        eval_fn = self._get_eval_fn()
        return eval_fn(anchor_chunks, query, "GENERAL", parent_span=trace)

    def summarize_history(self, turns_to_compress: list[dict], existing_summary: str) -> str:
        """Compress evicted turns + existing summary into a rolling summary."""
        transcript_lines = []
        for msg in turns_to_compress:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if content:
                transcript_lines.append(f"{role}: {content[:500]}")
        transcript = "\n".join(transcript_lines)

        prior_block = f"Prior summary:\n{existing_summary}\n\n" if existing_summary else ""
        prompt = (
            "You are maintaining a running summary of a student's conversation "
            "with the Texas A&M University course assistant chatbot.\n\n"
            f"{prior_block}"
            "New conversation turns to incorporate:\n"
            f"{transcript}\n\n"
            "Write a concise summary (3-6 sentences) covering all topics, courses, "
            "and questions discussed. Preserve any course IDs or specific details "
            "mentioned. Output ONLY the summary, no other text."
        )
        try:
            result = call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,  # TAMU gateway requires min 4096
            )
            return result.text.strip() or existing_summary
        except Exception:
            return existing_summary
