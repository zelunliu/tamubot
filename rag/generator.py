"""Generator (Outlet LLM) — Stage 3 of the 3-stage RAG pipeline.

Takes reranked retrieval results and generates a grounded, cited response
using Gemini Flash with function-adaptive system prompts.
"""

import json
import re

import config
from rag.context_builder import format_context_xml, collapse_whitespace
from rag.gates import validate_citations_with_trace
from rag.llm_client import call_llm, stream_llm
from rag.prompts import (
    _BASE_SYSTEM,
    _FUNCTION_PROMPTS,
    _SEMANTIC_TYPE_PROMPTS,
    _FUNCTION_TEMPERATURES,
    UNCERTAINTY_INJECTION,
)


# ---------------------------------------------------------------------------
# Function-adaptive system prompt assembly
# ---------------------------------------------------------------------------

def build_system_prompt(
    function: str,
    course_ids: list[str] | None = None,
    semantic_type: str | None = None,
    category_confidence: float | None = None,
) -> str:
    """Build a function-adaptive system prompt.

    Args:
        function: Router function type (e.g., "metadata_specific").
        course_ids: List of course IDs referenced in the query.
        semantic_type: Advisory semantic type (e.g., "CAREER").
        category_confidence: Router's confidence in category extraction (0.0-1.0).
                            If < 0.7, injects Verbal Uncertainty Calibration (VUC).
    """
    parts = [_BASE_SYSTEM]

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
    if semantic_type and semantic_type in _SEMANTIC_TYPE_PROMPTS:
        parts.append(_SEMANTIC_TYPE_PROMPTS[semantic_type])

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
    semantic_type: str | None = None,
    trace=None,
) -> str:
    """Generate a grounded response with citations using Gemini 2.0 Flash.

    Args:
        results:       Reranked retrieval results (list of chunk dicts).
        question:      The user's original question.
        function:      Retrieval function from the router (e.g. "metadata_specific").
        course_ids:    Extracted course IDs for context.
        semantic_type: Advisory semantic type from the router (e.g. "CAREER").
        trace:         Optional Langfuse trace; creates a Generator_Stage span.

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

    # Route multi-course queries to single-call comparison architecture
    if course_ids and len(course_ids) > 1:
        return generate_comparison(results, question, course_ids, trace)

    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(function, course_ids, semantic_type)
    user_message = f"{context_xml}\n\nQuestion: {question}"

    # Generator_Stage generation span
    generation_span = None
    if trace is not None:
        try:
            generation_span = trace.generation(
                name="Generator_Stage",
                model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
                input=user_message,
                metadata={
                    "function": function,
                    "semantic_type": semantic_type,
                    "course_ids": course_ids or [],
                    "n_sources": len(results),
                    "system_prompt_length": len(system_prompt),
                },
            )
        except Exception:
            generation_span = None

    # Determine thinking budget based on function type
    thinking_budget = (
        config.THINKING_BUDGET_SEMANTIC
        if function in ["hybrid_default", "hybrid_specific", "hybrid_combined", "semantic_general"]
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
    text = collapse_whitespace(text).strip()

    # Gate 1: Validate citations in response
    validate_citations_with_trace(text, function, trace)

    # Gate 2: Launch async groundedness scoring (fire-and-forget)
    contexts = [doc.get("content", "") for doc in results]
    trace_id = trace.id if trace is not None else None
    if trace_id:
        from rag.observability import run_groundedness_scoring_background
        run_groundedness_scoring_background(question, contexts, text, trace_id)

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


def _render_comparison_markdown(courses: list[dict]) -> str:
    """Render a Markdown comparison table + Detailed Comparison from structured course data.

    Builds the output entirely in Python from the structured extraction so we
    don't rely on Gemini to produce Markdown inside a JSON string field.
    """
    _EMPTY = {"", "n/a", "not found in original syllabus"}

    def _cell(val: str) -> str:
        """Escape pipe chars and collapse long whitespace for table cells."""
        return re.sub(r'\s+', ' ', val.replace("|", "\\|")).strip() or "N/A"

    lines: list[str] = []

    # 4-column summary table
    lines += [
        "| Course | Grading | Workload | Prerequisites |",
        "|--------|---------|----------|---------------|",
    ]
    for c in courses:
        lines.append(
            f"| {_cell(c.get('course_id', 'Unknown'))} "
            f"| {_cell(c.get('grading', ''))} "
            f"| {_cell(c.get('workload', ''))} "
            f"| {_cell(c.get('prerequisites', ''))} |"
        )

    # Detailed Comparison prose
    lines += ["", "## Detailed Comparison"]
    subsections = [
        ("### Course Overview",    "course_overview"),
        ("### Learning Outcomes",  "learning_outcomes"),
        ("### Topic Complexity",   "topics_complexity"),
        ("### Materials",          "materials"),
    ]
    for heading, field in subsections:
        parts = []
        for c in courses:
            val = c.get(field, "").strip()
            if val.lower() not in _EMPTY:
                parts.append(f"**{c.get('course_id', 'Unknown')}**: {val}")
        if parts:
            lines += ["", heading, ""] + ["\n\n".join(parts)]

    return "\n".join(lines)


def generate_comparison(
    results: list[dict],
    question: str,
    course_ids: list[str],
    trace=None,
) -> str:
    """Generate a multi-course comparison using a single-call architecture.

    One Gemini call extracts per-course structured data via CourseComparisonTable.
    The Markdown table + Detailed Comparison are then rendered in Python from
    that structured data — no second LLM call needed.

    Args:
        results:    Reranked retrieval results (list of chunk dicts).
        question:   The user's original question.
        course_ids: List of course IDs being compared (len > 1).
        trace:      Optional Langfuse trace; creates a Comparison_Extraction span.

    Returns:
        Markdown comparison table + detailed comparison string.
    """
    from rag.comparison_schemas import CourseComparisonTable

    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(
        function="hybrid_combined",
        course_ids=course_ids,
        semantic_type="GENERAL",
    )

    extraction_span = None
    if trace is not None:
        try:
            extraction_span = trace.generation(
                name="Comparison_Extraction",
                model=config.TAMU_MODEL if config.USE_TAMU_API else config.GENERATION_MODEL,
                input=question,
                metadata={
                    "function": "hybrid_combined",
                    "course_ids": course_ids,
                    "n_sources": len(results),
                    "call": "single_call",
                },
            )
        except Exception:
            extraction_span = None

    # Look up which categories are missing from each course's original syllabus
    from rag.search import get_missing_sections
    missing_per_course = {cid: get_missing_sections(cid) for cid in course_ids}
    missing_note_lines = []
    for cid, missing in missing_per_course.items():
        if missing:
            missing_note_lines.append(f"  {cid}: {', '.join(missing)}")
    missing_note = (
        "Sections confirmed missing from the original syllabi (use \"Not found in original syllabus\" for these fields):\n"
        + "\n".join(missing_note_lines)
        if missing_note_lines else ""
    )

    courses_list = ", ".join(course_ids)
    extraction_prompt = f"""{context_xml}

Question: {question}

You MUST extract data for ALL of the following courses: {courses_list}.
Produce one entry per course — do not skip any course.
For each course provide:
- grading: full grading breakdown
- workload: workload/difficulty assessment
- prerequisites: required background
- course_overview: description, scope, and general difficulty from the syllabus
- learning_outcomes: key learning objectives students are expected to achieve
- topics_complexity: key topics and their technical depth or difficulty level
- materials: required textbooks/software that signal course level (use empty string if not found)

{missing_note}""".strip()

    llm_result = None
    try:
        llm_result = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": extraction_prompt},
            ],
            temperature=_FUNCTION_TEMPERATURES.get("hybrid_combined", 0.2),
            max_tokens=4096,
            json_schema={
                "name": "CourseComparisonTable",
                "schema": CourseComparisonTable.model_json_schema(),
                "strict": True,
            },
            response_schema=CourseComparisonTable,
        )
        extraction_text = llm_result.text
    except Exception as e:
        if extraction_span is not None:
            try:
                extraction_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    # Parse structured data and render Markdown in Python
    try:
        parsed = json.loads(extraction_text)
        courses_data = parsed.get("courses", [])
        table_text = _render_comparison_markdown(courses_data)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: return raw extraction text so the user sees something
        table_text = extraction_text

    if extraction_span is not None:
        try:
            if llm_result is not None and llm_result.input_tokens is not None:
                extraction_span.end(
                    output=table_text,
                    usage={
                        "input": llm_result.input_tokens,
                        "output": llm_result.output_tokens,
                    },
                )
            else:
                extraction_span.end(output=table_text)
        except Exception:
            pass

    return table_text


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
    semantic_type: str | None = None,
    trace=None,
):
    """Streaming variant of generate(). Yields text chunks as they arrive.

    Single-course queries stream tokens directly from Gemini using
    generate_content_stream(). Multi-course comparison queries fall back to
    blocking generate_comparison() (streaming is unavailable when the JSON
    response must be fully received before Markdown can be rendered).

    Args:
        results:       Reranked retrieval results (list of chunk dicts).
        question:      The user's original question.
        function:      Retrieval function from the router.
        course_ids:    Extracted course IDs for context.
        semantic_type: Advisory semantic type from the router.
        trace:         Optional Langfuse trace.

    Yields:
        str: Text chunks as they arrive from the model.
    """
    # out_of_scope: yield canned response immediately
    if function == "out_of_scope":
        yield _OUT_OF_SCOPE_RESPONSE
        return

    # Multi-course comparison: fall back to blocking generate_comparison()
    # (streaming unsupported until single-call JSON response is fully received)
    if course_ids and len(course_ids) > 1:
        yield generate_comparison(results, question, course_ids, trace)
        return

    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(function, course_ids, semantic_type)
    user_message = f"{context_xml}\n\nQuestion: {question}"

    thinking_budget = (
        config.THINKING_BUDGET_SEMANTIC
        if function in ["hybrid_default", "hybrid_specific", "hybrid_combined", "semantic_general"]
        else config.THINKING_BUDGET_METADATA
    )

    full_text_parts: list[str] = []

    for token in stream_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=_FUNCTION_TEMPERATURES.get(function, 0.1),
        max_tokens=4096,
        thinking_budget=thinking_budget,
    ):
        full_text_parts.append(token)
        yield token

    # Post-stream: run Gate 1 citation check + async Gate 2 groundedness scoring
    complete_text = "".join(full_text_parts)
    validate_citations_with_trace(complete_text, function, trace)

    if trace is not None:
        contexts = [doc.get("content", "") for doc in results]
        trace_id = trace.id
        if trace_id:
            from rag.observability import run_groundedness_scoring_background
            run_groundedness_scoring_background(question, contexts, complete_text, trace_id)
