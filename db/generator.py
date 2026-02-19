"""Generator (Outlet LLM) — Stage 3 of the 3-stage RAG pipeline.

Takes reranked retrieval results and generates a grounded, cited response
using Gemini Flash with intent-adaptive system prompts.
"""

import re

from google.genai import types

import config

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context_xml(results: list[dict]) -> str:
    """Format retrieval results as XML-tagged chunks for the generator.

    Each chunk gets metadata attributes so the LLM can cite sources precisely.
    """
    if not results:
        return "<context>\nNo relevant documents found.\n</context>"

    parts = ["<context>"]
    for i, doc in enumerate(results, 1):
        attrs = [f'source="{i}"']
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

        parts.append(f"<chunk {attr_str}>")
        if title:
            parts.append(f"<title>{title}</title>")
        parts.append(f"<content>{content}</content>")
        parts.append("</chunk>")
    parts.append("</context>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Intent-adaptive system prompts
# ---------------------------------------------------------------------------

_BASE_SYSTEM = """\
You are TamuBot, an academic assistant for Texas A&M University.
You help students find information about courses, syllabi, policies, and schedules.

RULES:
1. Answer ONLY based on the provided <context>. Never invent information.
2. Cite your sources using [Source N] notation matching the source numbers in the context.
3. If the context does not contain enough information to answer, say so clearly.
4. Do NOT answer questions outside TAMU academics — politely decline.
5. Be concise but thorough. Use markdown formatting for readability.
6. When using markdown tables, do NOT pad cells with extra spaces. Keep columns compact.
"""

_INTENT_PROMPTS = {
    "single_course_lookup": (
        "The user is asking about a specific course. "
        "Provide a clear, detailed answer based on the syllabus information. "
        "Include the course ID and section in your response."
    ),
    "multi_course_comparison": (
        "The user is comparing multiple courses. "
        "Present the comparison in a clear markdown table when possible. "
        "Ensure you cover each course mentioned and highlight key differences. "
        "Use [Source N] citations for each piece of information."
    ),
    "aggregation_query": (
        "The user is asking for counts, lists, or summaries. "
        "Present numerical data clearly. List all relevant items."
    ),
    "policy_lookup": (
        "The user is asking about a university policy. "
        "Provide the policy information accurately and completely."
    ),
    "schedule_query": (
        "The user is asking about course meeting times or schedule. "
        "Be precise about days, times, and locations."
    ),
    "instructor_query": (
        "The user is asking about instructors. "
        "Include name, office hours, email, and office location if available."
    ),
    "general_academic": (
        "The user has a general academic question. "
        "Provide a helpful answer based on the available context. "
        "If multiple courses are relevant, mention each."
    ),
}


def build_system_prompt(intent: str, course_ids: list[str] | None = None) -> str:
    """Build an intent-adaptive system prompt."""
    parts = [_BASE_SYSTEM]

    intent_instruction = _INTENT_PROMPTS.get(intent, _INTENT_PROMPTS["general_academic"])
    parts.append(intent_instruction)

    if course_ids:
        parts.append(f"Courses referenced: {', '.join(course_ids)}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    results: list[dict],
    question: str,
    intent: str = "general_academic",
    course_ids: list[str] | None = None,
) -> str:
    """Generate a grounded response with citations using Gemini 2.5 Flash.

    Args:
        results: reranked retrieval results (list of chunk dicts)
        question: the user's original question
        intent: classified intent from the router
        course_ids: extracted course IDs for context

    Returns:
        Generated answer string with [Source N] citations.
    """
    # Out-of-scope gets a canned response without an LLM call
    if intent == "out_of_scope":
        return (
            "Howdy! I'm TamuBot, your Texas A&M academic assistant. "
            "I can help you with questions about courses, syllabi, grading policies, "
            "schedules, and university policies. What would you like to know?"
        )

    context_xml = format_context_xml(results)
    system_prompt = build_system_prompt(intent, course_ids)

    user_message = f"{context_xml}\n\nQuestion: {question}"

    client = config.get_genai_client()
    response = client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=4096,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    text = response.text or ""
    # Gemini sometimes pads markdown table cells with excessive whitespace
    text = re.sub(r' {3,}', ' ', text)
    return text.strip()
