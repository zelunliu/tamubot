"""Generator (Outlet LLM) — Stage 3 of the 3-stage RAG pipeline.

Takes reranked retrieval results and generates a grounded, cited response
using Gemini Flash with function-adaptive system prompts.
"""

import config
from rag.context_builder import collapse_whitespace, format_context_xml
from rag.gates import validate_citations_with_trace
from rag.llm_client import call_llm, stream_llm
from rag.prompts import (
    _BASE_SYSTEM,
    _FUNCTION_PROMPTS,
    _FUNCTION_TEMPERATURES,
    _HYBRID_COURSE_COMBINED,
    _HYBRID_COURSE_DEFAULT,
    _HYBRID_COURSE_SPECIFIC,
    _SEMANTIC_TYPE_PROMPTS,
    COMPARISON_SYSTEM,
    UNCERTAINTY_INJECTION,
)

# ---------------------------------------------------------------------------
# Eval Pass — context-aware search string for recurrent discovery
# ---------------------------------------------------------------------------

def generate_eval_search_string(
    anchor_chunks: list[dict],
    original_query: str,
    intent_type: str,
    parent_span=None,
) -> str:
    """Recurrent Eval Pass: generate a context-aware vector search string.

    Given anchor chunks (already retrieved) and the user's original query,
    produce a concise 1-2 sentence search string for the corpus-wide
    discovery step. Uses temperature=0 for determinism.

    Args:
        anchor_chunks:  Chunks from the anchor course(s) metadata fetch.
        original_query: The user's original question (or rewritten_query).
        intent_type:    Advisory dimension (e.g. "PLANNING", "ACADEMIC").
        parent_span:    Optional Langfuse trace/span; creates EvalSearch_Stage generation.

    Returns:
        A concise search string for the hybrid discovery step.
    """
    # Summarize anchor content, capped to avoid token bloat
    anchor_text = " ".join(
        f"{c.get('header_text', '')} {c.get('content', '')}" for c in anchor_chunks
    )[:2000]

    eval_prompt = (
        f"You are helping a student find related courses at Texas A&M University.\n\n"
        f"The student asked: {original_query}\n\n"
        f"Intent dimension: {intent_type}\n\n"
        f"Here is content from the anchor course(s) they mentioned:\n{anchor_text}\n\n"
        f"Based on the anchor course content and the student's question, write a concise "
        f"search string (1-2 sentences) that captures what additional courses or content "
        f"to search for. Output ONLY the search string, no other text."
    )

    eval_gen = None
    if parent_span is not None:
        try:
            eval_gen = parent_span.generation(
                name="EvalSearch_Stage",
                model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
                input=[{"role": "user", "content": eval_prompt}],
                metadata={"intent_type": intent_type, "n_anchor_chunks": len(anchor_chunks)},
            )
        except Exception:
            eval_gen = None

    llm_result = None
    try:
        llm_result = call_llm(
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
            max_tokens=4096,  # TAMU gateway requires min 4096 or response is empty
        )
        text = llm_result.text.strip() or original_query
    except Exception as e:
        if eval_gen is not None:
            try:
                eval_gen.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        return original_query

    if eval_gen is not None:
        try:
            eval_gen.end(
                output=text,
                usage=(
                    {"input": llm_result.input_tokens, "output": llm_result.output_tokens}
                    if llm_result and llm_result.input_tokens is not None
                    else None
                ),
            )
        except Exception:
            pass

    return text


# ---------------------------------------------------------------------------
# Function-adaptive system prompt assembly
# ---------------------------------------------------------------------------

def build_system_prompt(
    function: str,
    course_ids: list[str] | None = None,
    intent_type: str | None = None,
    category_confidence: float | None = None,
    specific_categories: list[str] | None = None,
    specific_only: bool = False,
) -> str:
    """Build a function-adaptive system prompt.

    Args:
        function: Router function type (e.g., "hybrid_course", "recurrent").
        course_ids: List of course IDs referenced in the query.
        intent_type: Advisory intent type from the router (e.g., "CAREER").
        category_confidence: Router's confidence in category extraction (0.0-1.0).
                            If < 0.7, injects Verbal Uncertainty Calibration (VUC).
        specific_categories: Categories the user asked about (from router).
        specific_only: True if the user asked ONLY about those categories.
    """
    parts = [_BASE_SYSTEM]

    if function == "hybrid_course":
        cats = specific_categories or []
        if not cats:
            function_instruction = _HYBRID_COURSE_DEFAULT
        elif specific_only:
            function_instruction = _HYBRID_COURSE_SPECIFIC
        else:
            function_instruction = _HYBRID_COURSE_COMBINED
    else:
        function_instruction = _FUNCTION_PROMPTS.get(function, _FUNCTION_PROMPTS["semantic_general"])
    parts.append(function_instruction)

    # Verbal Uncertainty Calibration: inject uncertainty language if confidence is low
    if category_confidence is not None and category_confidence < 0.7:
        parts.append(UNCERTAINTY_INJECTION)

    # Multi-course comparison overlay
    if course_ids and len(course_ids) > 1:
        parts.append(
            "The user is comparing multiple courses. "
            "Present the comparison in a Markdown table. "
            "IMMEDIATELY ADD ' |' AFTER EACH HEADING — do not add extra spaces for column alignment. "
            "Ensure you cover each course mentioned and highlight key differences. "
            "Use [Source N] citations for each piece of information."
        )

    # Advisory overlay for hybrid/semantic functions
    if intent_type and intent_type in _SEMANTIC_TYPE_PROMPTS:
        parts.append(_SEMANTIC_TYPE_PROMPTS[intent_type])

    if course_ids:
        parts.append(f"Courses referenced: {', '.join(course_ids)}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    results: list[dict],
    question: str,
    function: str = "semantic_general",
    course_ids: list[str] | None = None,
    intent_type: str | None = None,
    specific_categories: list[str] | None = None,
    specific_only: bool = False,
    data_gaps: list[tuple[str, str]] | None = None,
    data_integrity: bool = True,
    trace=None,
    history_context: str | None = None,
) -> str:
    """Generate a grounded response with citations using Gemini 2.0 Flash.

    Args:
        results:             Reranked retrieval results (list of chunk dicts).
        question:            The user's original question.
        function:            Retrieval function from the router (e.g. "hybrid_course").
        course_ids:          Extracted course IDs for context.
        intent_type:         Advisory intent type from the router (e.g. "CAREER").
        specific_categories: Categories the user asked about (router extraction).
        specific_only:       True if the user asked ONLY about those categories.
        data_gaps:           [(course_id, category)] pairs missing from DB (recurrent only).
        data_integrity:      False if any data gaps were found; triggers disclaimer.
        trace:               Optional Langfuse trace; creates a Generator_Stage span.

    Returns:
        Generated answer string with [Source N] citations.
    """
    # out_of_scope gets a canned response without an LLM call
    if function == "out_of_scope":
        return (
            "Howdy! I'm TamuBot, your Texas A&M academic assistant. "
            "I can help you with questions about courses, syllabi, grading policies, "
            "schedules, and university policies. What would you like to know?"
        )

    # Route multi-course known-course queries to streaming comparison.
    # Recurrent queries may have multiple anchor course IDs but need pairing framing, not comparison.
    if course_ids and len(course_ids) > 1 and function != "recurrent":
        return "".join(generate_comparison(results, question, course_ids, trace, specific_categories))

    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(
        function, course_ids, intent_type,
        specific_categories=specific_categories, specific_only=specific_only,
    )
    if history_context:
        user_message = (
            f"{context_xml}\n\n"
            f"<conversation_history>\n{history_context}\n</conversation_history>\n\n"
            f"Question: {question}"
        )
    else:
        user_message = f"{context_xml}\n\nQuestion: {question}"

    # Generator_Stage generation span
    generation_span = None
    if trace is not None:
        try:
            generation_span = trace.generation(
                name="Generator_Stage",
                model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                metadata={
                    "function": function,
                    "intent_type": intent_type,
                    "course_ids": course_ids or [],
                    "n_sources": len(results),
                    "system_prompt_length": len(system_prompt),
                    "data_integrity": data_integrity,
                    "n_data_gaps": len(data_gaps) if data_gaps else 0,
                },
            )
        except Exception:
            generation_span = None

    # Determine thinking budget based on function type and intent.
    # Advisory queries (intent_type set) on hybrid_course also benefit from thinking.
    thinking_budget = (
        config.THINKING_BUDGET_SEMANTIC
        if function in ["recurrent", "semantic_general"] or intent_type is not None
        else config.THINKING_BUDGET_METADATA
    )

    llm_result = None
    try:
        llm_result = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=_FUNCTION_TEMPERATURES.get(function, 0.1),
            max_tokens=4096,
            thinking_budget=thinking_budget,
        )
        text = llm_result.text
    except Exception as e:
        if generation_span is not None:
            try:
                generation_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise
    text = collapse_whitespace(text)

    # Prepend data integrity disclaimer when DB chunks are missing
    if not data_integrity and data_gaps:
        gap_lines = "\n".join(f"- {cid} / {cat}" for cid, cat in data_gaps)
        disclaimer = (
            f"⚠️ Note: The following data was not found in the syllabus database:\n{gap_lines}\n\n"
        )
        text = disclaimer + text

    # Gate 1: Validate citations in response
    validate_citations_with_trace(text, function, trace)

    # Gate 2 (groundedness scoring) intentionally disabled — uses LLM on every query.

    if generation_span is not None:
        try:
            if llm_result is not None and llm_result.input_tokens is not None:
                generation_span.end(
                    output=text,
                    usage={
                        "input": llm_result.input_tokens,
                        "output": llm_result.output_tokens,
                    },
                    metadata={"thinking_tokens": llm_result.thinking_tokens},
                )
            else:
                generation_span.end(output=text)
        except Exception:
            pass

    return text


def generate_comparison(
    results: list[dict],
    question: str,
    course_ids: list[str],
    trace=None,
    specific_categories: list[str] | None = None,
):
    """Stream a multi-course comparison as free-form markdown.

    Uses stream_llm (no constrained decoding) so tokens arrive immediately.

    Args:
        results:             Reranked retrieval results (list of chunk dicts).
        question:            The user's original question.
        course_ids:          List of course IDs being compared (len > 1).
        trace:               Optional Langfuse trace; creates a Comparison_Generation span.
        specific_categories: If set, constrain comparison to these syllabus categories.

    Yields:
        str: Text tokens as they arrive.
    """
    context_xml = format_context_xml(results)
    courses_list = ", ".join(course_ids)
    focus = (
        f" Focus only on: {', '.join(specific_categories)}."
        if specific_categories else ""
    )
    user_message = (
        f"{context_xml}\n\n"
        f"Question: {question}\n\n"
        f"Compare the following courses: {courses_list}.{focus}"
    )

    generation_span = None
    if trace is not None:
        try:
            generation_span = trace.generation(
                name="Comparison_Generation",
                model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
                input=[
                    {"role": "system", "content": COMPARISON_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                metadata={
                    "function": "hybrid_course",
                    "course_ids": course_ids,
                    "n_sources": len(results),
                    "streaming": True,
                },
            )
        except Exception:
            generation_span = None

    full_text_parts: list[str] = []
    usage_out: list = []

    for token in stream_llm(
        messages=[
            {"role": "system", "content": COMPARISON_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=4096,
        usage_out=usage_out,
    ):
        full_text_parts.append(token)
        yield token

    if generation_span is not None:
        try:
            full_text = "".join(full_text_parts)
            if usage_out:
                generation_span.end(
                    output=full_text,
                    usage={"input": usage_out[0], "output": usage_out[1]},
                    metadata={"thinking_tokens": usage_out[2] if len(usage_out) > 2 else 0},
                )
            else:
                generation_span.end(output=full_text)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_RESPONSE = (
    "Howdy! I'm TamuBot, your Texas A&M academic assistant. "
    "I can help you with questions about courses, syllabi, grading policies, "
    "schedules, and university policies. What would you like to know?"
)


def generate_stream(
    results: list[dict],
    question: str,
    function: str = "semantic_general",
    course_ids: list[str] | None = None,
    intent_type: str | None = None,
    specific_categories: list[str] | None = None,
    specific_only: bool = False,
    data_gaps: list[tuple[str, str]] | None = None,
    data_integrity: bool = True,
    conflicted_course_ids: list[str] | None = None,
    trace=None,
    history_context: str | None = None,
):
    """Streaming variant of generate(). Yields text chunks as they arrive.

    Single-course queries stream tokens directly from Gemini using
    generate_content_stream(). Multi-course comparison queries also stream
    via generate_comparison() using free-form markdown (no constrained decoding).

    Args:
        results:        Reranked retrieval results (list of chunk dicts).
        question:       The user's original question.
        function:       Retrieval function from the router.
        course_ids:     Extracted course IDs for context.
        intent_type:    Advisory intent type from the router (e.g. "CAREER").
        data_gaps:      [(course_id, category)] pairs missing from DB (recurrent only).
        data_integrity: False if any data gaps were found; triggers disclaimer.
        trace:          Optional Langfuse trace.

    Yields:
        str: Text chunks as they arrive from the model.
    """
    # out_of_scope: yield canned response immediately
    if function == "out_of_scope":
        yield _OUT_OF_SCOPE_RESPONSE
        return

    # Multi-course known-course comparison: stream free-form markdown directly.
    # Recurrent queries with multiple anchors stream normally using their own prompts.
    if course_ids and len(course_ids) > 1 and function != "recurrent":
        yield from generate_comparison(results, question, course_ids, trace, specific_categories)
        return

    # Schedule conflict notice (recurrent path only)
    if conflicted_course_ids:
        names = ", ".join(conflicted_course_ids)
        plural = len(conflicted_course_ids) > 1
        yield (
            f"_Note: {names} {'were' if plural else 'was'} excluded — "
            f"{'their' if plural else 'its'} meeting time conflicts with your anchor course._\n\n"
        )

    # Prepend data integrity disclaimer before streaming begins
    if not data_integrity and data_gaps:
        gap_lines = "\n".join(f"- {cid} / {cat}" for cid, cat in data_gaps)
        disclaimer = (
            f"⚠️ Note: The following data was not found in the syllabus database:\n{gap_lines}\n\n"
        )
        yield disclaimer

    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(
        function, course_ids, intent_type,
        specific_categories=specific_categories, specific_only=specific_only,
    )
    if history_context:
        user_message = (
            f"{context_xml}\n\n"
            f"<conversation_history>\n{history_context}\n</conversation_history>\n\n"
            f"Question: {question}"
        )
    else:
        user_message = f"{context_xml}\n\nQuestion: {question}"

    thinking_budget = (
        config.THINKING_BUDGET_SEMANTIC
        if function in ["recurrent", "semantic_general"] or intent_type is not None
        else config.THINKING_BUDGET_METADATA
    )

    # Generator_Stage generation span
    generation_span = None
    if trace is not None:
        try:
            generation_span = trace.generation(
                name="Generator_Stage",
                model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                metadata={
                    "function": function,
                    "intent_type": intent_type,
                    "course_ids": course_ids or [],
                    "n_sources": len(results),
                    "system_prompt_length": len(system_prompt),
                    "data_integrity": data_integrity,
                    "n_data_gaps": len(data_gaps) if data_gaps else 0,
                    "streaming": True,
                    "thinking_budget": thinking_budget,
                },
            )
        except Exception:
            generation_span = None

    full_text_parts: list[str] = []
    usage_out: list = []

    for token in stream_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=_FUNCTION_TEMPERATURES.get(function, 0.1),
        max_tokens=4096,
        thinking_budget=thinking_budget,
        usage_out=usage_out,
    ):
        full_text_parts.append(token)
        yield token

    # Post-stream: run Gate 1 citation check + async Gate 2 groundedness scoring
    complete_text = "".join(full_text_parts)
    validate_citations_with_trace(complete_text, function, trace)

    # Gate 2 (groundedness scoring) intentionally disabled — uses LLM on every query.

    if generation_span is not None:
        try:
            usage = (
                {"input": usage_out[0], "output": usage_out[1]}
                if len(usage_out) >= 2 and usage_out[0] is not None
                else None
            )
            generation_span.end(
                output=complete_text,
                usage=usage,
                metadata={"thinking_tokens": usage_out[2] if len(usage_out) >= 3 else None},
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# generator_order — thin orchestration wrapper (moved from pipeline.py)
# ---------------------------------------------------------------------------

def generator_order(
    recurrent: bool,
    chunks: list[dict],
    query: str,
    router_result,
    data_gaps=None,
    data_integrity: bool = True,
    conflicted_course_ids=None,
    trace=None,
):
    """Generator_Stage: eval search string (recurrent) or answer stream (final).

    Returns:
        str (recurrent=True) or Iterator[str] (recurrent=False)
    """
    if recurrent:
        return generate_eval_search_string(
            chunks,
            router_result.rewritten_query or query,
            router_result.intent_type or "GENERAL",
        )
    return generate_stream(
        results=chunks,
        question=query,
        function=router_result.function,
        course_ids=router_result.course_ids,
        intent_type=router_result.intent_type,
        specific_categories=router_result.specific_categories,
        specific_only=router_result.specific_only,
        data_gaps=data_gaps or [],
        data_integrity=data_integrity,
        conflicted_course_ids=conflicted_course_ids or [],
        trace=trace,
    )
