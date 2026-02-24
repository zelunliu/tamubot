"""Generator (Outlet LLM) — Stage 3 of the 3-stage RAG pipeline.

Takes reranked retrieval results and generates a grounded, cited response
using Gemini Flash with function-adaptive system prompts.
"""

import html
import re

from google.genai import types

import config

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context_xml(results: list[dict]) -> str:
    """Format retrieval results as XML-tagged chunks for the generator.

    Implements primacy-recency bracketing to combat Lost-in-the-Middle attention degradation:
    - Rank 1 chunk → Context start (primacy position)
    - Rank 2 chunk → Context end (recency/nearest query position)
    - Ranks 3–N → Middle (descending rank order)

    Each chunk gets metadata attributes so the LLM can cite sources precisely.
    """
    if not results:
        return "<context>\nNo relevant documents found.\n</context>"

    # Apply primacy-recency reordering: [rank_1, ranks_3_to_N, rank_2]
    if len(results) == 1:
        ordered_results = results
        rank_mapping = [1]
    elif len(results) == 2:
        ordered_results = [results[0], results[1]]
        rank_mapping = [1, 2]
    else:
        # Rank 1 at start, Rank 2 at end, Ranks 3-N in middle (descending order)
        ordered_results = [results[0]] + results[2:] + [results[1]]
        rank_mapping = [1] + list(range(3, len(results) + 1)) + [2]

    parts = ["<context>"]
    for position, (rank, doc) in enumerate(zip(rank_mapping, ordered_results), 1):
        # source= attribute uses original rank for citation purposes
        attrs = [f'source="{rank}"', f'id="{rank}"']
        if doc.get("course_id"):
            attrs.append(f'course="{doc["course_id"]}"')
        if doc.get("section"):
            attrs.append(f'section="{doc["section"]}"')
        if doc.get("category"):
            attrs.append(f'category="{doc["category"]}"')
        if doc.get("instructor_name"):
            attrs.append(f'instructor="{doc["instructor_name"]}"')
        if doc.get("term"):
            attrs.append(f'term="{doc["term"]}"')

        attr_str = " ".join(attrs)
        title = doc.get("title", "")
        content = doc.get("content", doc.get("policy_name", ""))

        # XML escape special characters in content
        content_escaped = html.escape(content)
        title_escaped = html.escape(title) if title else ""

        parts.append(f"<chunk {attr_str}>")
        if title_escaped:
            parts.append(f"<title>{title_escaped}</title>")
        parts.append(f"<content>{content_escaped}</content>")
        parts.append("</chunk>")
    parts.append("</context>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Function-adaptive system prompts
# ---------------------------------------------------------------------------

_BASE_SYSTEM = """\
You are TamuBot, an academic assistant for Texas A&M University.
You help students find information about courses, syllabi, policies, and schedules.

RULES:
1. Answer ONLY based on the provided <context>. Never invent information.
2. Cite your sources using [Source N] notation matching the source numbers in the context.
3. Chain-of-Verification (Quote-then-Paraphrase): Before answering, extract a verbatim quote \
from the most relevant chunk into a <thinking> block. Then paraphrase that quote into your \
student-facing answer with [Source N] citation. This ensures all claims are grounded in the provided text.
4. Verification: Before answering, identify which chunk contains the answer. \
If no chunk contains it, state "I cannot find that information in the provided context" \
and do NOT use training data to fill the gap.
5. Do NOT answer questions outside TAMU academics — politely decline.
6. Be concise but thorough. Use markdown formatting for readability.
7. When using markdown tables, do NOT pad cells with extra spaces. Keep columns compact.
"""

# Primary prompt per function — describes the factual framing of the response.
_FUNCTION_PROMPTS: dict[str, str] = {
    "metadata_default": (
        "The user is asking for a general overview of a course. "
        "Provide key facts about course overview, prerequisites, and learning outcomes. "
        "Include the course ID and section. Label information by section where multiple sections are present."
    ),
    "metadata_specific": (
        "The user is asking about specific course details. "
        "Focus on the requested categories and be precise and complete. "
        "Include the course ID and section. Name the instructor where relevant."
    ),
    "metadata_combined": (
        "The user is asking about specific course details in the context of a broader overview. "
        "Cover both the requested categories and the general course overview. "
        "Include the course ID and section."
    ),
    "semantic_general": (
        "The user has a broad question not tied to a specific course. "
        "First define the relevant principle or framework underlying the question, "
        "then apply that principle to the specific question using available context. "
        "Provide a helpful answer based only on the available context. "
        "If the evidence is insufficient to answer fully, state: "
        "'I don't have enough data to answer this accurately based on the available syllabi.'"
    ),
    "hybrid_default": (
        "The user is asking about a course with an advisory or subjective component. "
        "First define the relevant principle or framework, then apply it to the course. "
        "Provide factual information from the course content and address the advisory aspect "
        "using only evidence from the context. "
        "Limit all advisory statements to those grounded in specific course details."
    ),
    "hybrid_specific": (
        "The user is asking about specific course categories with an advisory component. "
        "First define the relevant principle or framework, then apply it to the specific categories. "
        "Focus on the requested categories and use that evidence to address the advisory dimension. "
        "Ground all advisory statements in specific facts from the context."
    ),
    "hybrid_combined": (
        "The user is asking about specific course details and a broader overview with an advisory component. "
        "First define the relevant principle or framework, then apply it to the courses. "
        "Cover all relevant categories and use the evidence to address the advisory aspect. "
        "Ground all advisory statements in specific facts from the context."
    ),
}

# Advisory overlay appended when semantic_type is present (hybrid_* and semantic_general).
_SEMANTIC_TYPE_PROMPTS: dict[str, str] = {
    "ACADEMIC": (
        "Address the academic dimension: discuss learning outcomes, topics covered, and academic content."
    ),
    "CAREER": (
        "Address the career relevance dimension: discuss how the course content relates to "
        "industry applications and career paths."
    ),
    "DIFFICULTY": (
        "Address the difficulty/workload dimension: use grading weights, prerequisites, and "
        "attendance requirements as evidence of course rigor."
    ),
    "PLANNING": (
        "Address the planning dimension: help the student understand how this course fits into "
        "their academic progression."
    ),
    "GENERAL": (
        "Address the advisory aspect of the question using evidence from the course context."
    ),
}

# Per-function generation temperature.
# metadata_* and semantic_general: 0.0 (maximum fidelity to context).
# hybrid_*: 0.1 (slight flexibility for advisory synthesis).
_FUNCTION_TEMPERATURES: dict[str, float] = {
    "metadata_default":  0.0,
    "metadata_specific": 0.0,
    "metadata_combined": 0.0,
    "semantic_general":  0.0,
    "hybrid_default":    0.1,
    "hybrid_specific":   0.1,
    "hybrid_combined":   0.1,
}


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
        parts.append(
            "NOTE: The extracted category is not high-confidence. "
            "Based on the available, but potentially incomplete, data in the provided context, "
            "provide your best answer. If the evidence is ambiguous or insufficient, acknowledge this limitation."
        )

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

    # format_context_xml handles primacy-recency bracketing:
    # Rank 1 → start, Rank 2 → end, Ranks 3-N → middle
    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(function, course_ids, semantic_type)
    user_message = f"{context_xml}\n\nQuestion: {question}"

    # Generator_Stage generation span
    generation_span = None
    if trace is not None:
        try:
            generation_span = trace.generation(
                name="Generator_Stage",
                model=config.GENERATION_MODEL,
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

    client = config.get_genai_client()
    try:
        response = client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=_FUNCTION_TEMPERATURES.get(function, 0.1),
                max_output_tokens=4096,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    except Exception as e:
        if generation_span is not None:
            try:
                generation_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    text = response.text or ""
    # Gemini sometimes pads markdown table cells with excessive whitespace
    text = re.sub(r' {3,}', ' ', text)
    text = text.strip()

    if generation_span is not None:
        try:
            usage = response.usage_metadata
            thinking_tokens = getattr(usage, "thoughts_token_count", None) or 0
            generation_span.end(
                output=text,
                usage={
                    "input": getattr(usage, "prompt_token_count", None),
                    "output": getattr(usage, "candidates_token_count", None),
                },
                metadata={"thinking_tokens": thinking_tokens},
            )
        except Exception:
            pass

    return text
