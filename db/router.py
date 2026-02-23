"""Query intent router — classifies user questions and orchestrates retrieval + reranking.

Stage 1 of the 3-stage RAG pipeline: Router/Inlet → Retrieval+Rerank → Generator/Outlet.

Uses Gemini 2.5 Flash for intent classification with 8 intent types, multi-course entity
extraction, and query rewriting/expansion.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from google.genai import types

import config
from db import search, reranker


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

VALID_INTENTS = [
    "single_course_lookup",
    "multi_course_comparison",
    "aggregation_query",
    "policy_lookup",
    "schedule_query",
    "instructor_query",
    "general_academic",
    "out_of_scope",
]

ROUTER_PROMPT = """\
You are a query classifier for a Texas A&M University course assistant.
Given the user's question, extract structured JSON with these fields:

{{
  "intent": one of {intents},
  "course_ids": list of course IDs mentioned (e.g. ["CSCE 120", "CSCE 221"]), or [],
  "section": section number if mentioned, or null,
  "category": one of ["COURSE_OVERVIEW", "INSTRUCTOR", "PREREQUISITES", "LEARNING_OUTCOMES", "MATERIALS", "GRADING", "SCHEDULE", "ATTENDANCE_AND_MAKEUP", "AI_POLICY", "UNIVERSITY_POLICIES", "SUPPORT_SERVICES"] if relevant, or null,
  "policy_name": boilerplate policy name if asking about a specific policy, or null,
  "rewritten_query": the user's question rewritten for optimal search retrieval (expand slang, add synonyms),
  "confidence": float 0-1 indicating how confident you are in the classification
}}

Intent definitions:
- "single_course_lookup": asking about a specific course's details (grading, schedule, instructor, materials, AI policy, etc.)
- "multi_course_comparison": comparing two or more courses or sections (e.g. "compare CSCE 120 and 221", "differences between...")
- "aggregation_query": asking for counts, lists, or summaries across sections (e.g. "how many sections of CSCE 120?", "which instructors teach...")
- "policy_lookup": asking about a university-wide boilerplate policy (academic integrity, disability, Title IX, etc.)
- "schedule_query": asking when a course meets, meeting times, days, room locations
- "instructor_query": asking who teaches a course, office hours, contact info
- "general_academic": broad questions about course topics, prerequisites, or recommendations not tied to a specific section
- "out_of_scope": greetings, weather, unrelated questions — not about TAMU academics

Query rewriting rules:
- "late work" → "attendance makeup deadline extensions late submission"
- "ChatGPT" / "AI tools" → "AI policy artificial intelligence generative AI tools"
- "prereqs" → "prerequisites required courses corequisites"
- "prof" / "teacher" → "instructor professor"
- "grade breakdown" → "grading policy grade distribution weight percentage"
- Keep the rewrite concise but include key synonyms for better retrieval.

Course ID normalization:
- "csce120" → "CSCE 120", "CSCE-120" → "CSCE 120", "csce 120" → "CSCE 120"
- Always output uppercase department + space + number

Respond with ONLY valid JSON, no other text.

User question: {query}
"""


# ---------------------------------------------------------------------------
# Router result
# ---------------------------------------------------------------------------

@dataclass
class RouterResult:
    """Structured output from the query router."""
    intent: str = "general_academic"
    course_ids: list[str] = field(default_factory=list)
    section: Optional[str] = None
    category: Optional[str] = None
    policy_name: Optional[str] = None
    rewritten_query: str = ""
    confidence: float = 0.0

    @property
    def requires_retrieval(self) -> bool:
        return self.intent != "out_of_scope"

    @property
    def is_comparison(self) -> bool:
        return self.intent == "multi_course_comparison"


def _normalize_course_id(raw: str) -> str:
    """Normalize a course ID like 'csce120' → 'CSCE 120'."""
    raw = raw.strip().upper().replace("-", " ")
    # If no space between letters and digits, insert one
    match = re.match(r"^([A-Z]+)\s*(\d+.*)$", raw)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return raw


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_query(query: str, router_span=None) -> "RouterResult":
    """Classify a user query into intent + entities using Gemini Flash.

    Args:
        query:       The raw user question.
        router_span: Optional Langfuse span to record router metadata into.
    """
    client = config.get_genai_client()

    prompt = ROUTER_PROMPT.format(
        query=query,
        intents=json.dumps(VALID_INTENTS),
    )

    try:
        response = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                max_output_tokens=512,
                thinking_config=types.ThinkingConfig(thinking_budget=512),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    except Exception as e:
        if router_span is not None:
            try:
                router_span.update(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    try:
        data = json.loads(response.text)
    except (json.JSONDecodeError, ValueError, AttributeError):
        result = RouterResult(intent="general_academic", rewritten_query=query, confidence=0.0)
        if router_span is not None:
            try:
                router_span.update(metadata={
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "course_ids": [],
                    "parse_error": True,
                })
            except Exception:
                pass
        return result

    # Normalize course IDs
    raw_ids = data.get("course_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not raw_ids and data.get("course_id"):
        raw_ids = [data["course_id"]]
    course_ids = [_normalize_course_id(c) for c in raw_ids]

    intent = data.get("intent", "general_academic")
    if intent not in VALID_INTENTS:
        intent = "general_academic"

    result = RouterResult(
        intent=intent,
        course_ids=course_ids,
        section=data.get("section"),
        category=data.get("category"),
        policy_name=data.get("policy_name"),
        rewritten_query=data.get("rewritten_query", query),
        confidence=data.get("confidence", 0.0),
    )

    # Record router metadata + token usage into the span
    if router_span is not None:
        try:
            usage = response.usage_metadata
            thinking_tokens = getattr(usage, "thoughts_token_count", None) or 0
            router_span.update(
                output=data,
                metadata={
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "course_ids": result.course_ids,
                    "rewritten_query": result.rewritten_query,
                    "input_tokens": getattr(usage, "prompt_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None),
                    "thinking_tokens": thinking_tokens,
                },
            )
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Orchestrator: route → retrieve → rerank
# ---------------------------------------------------------------------------

def route_retrieve_rerank(query: str, trace=None) -> tuple[list[dict], "RouterResult"]:
    """Full Stage 1+2 pipeline: classify → retrieve → rerank.

    Args:
        query: The raw user question.
        trace: Optional Langfuse trace to attach Router_Stage and Retrieval_Stage spans to.

    Returns:
        (reranked_results, router_result) — list of chunk dicts and the router classification.
    """
    # --- Router_Stage span ---
    router_span = None
    if trace is not None:
        try:
            router_span = trace.span(
                name="Router_Stage",
                input={"query": query},
            )
        except Exception:
            router_span = None

    router_result = classify_query(query, router_span=router_span)

    if router_span is not None:
        try:
            router_span.end()
        except Exception:
            pass

    search_query = router_result.rewritten_query or query

    # Out-of-scope: no retrieval
    if not router_result.requires_retrieval:
        return [], router_result

    # --- Retrieval_Stage span ---
    retrieval_span = None
    if trace is not None:
        try:
            retrieval_span = trace.span(
                name="Retrieval_Stage",
                input={
                    "query": search_query,
                    "intent": router_result.intent,
                    "course_ids": router_result.course_ids,
                },
            )
        except Exception:
            retrieval_span = None

    try:
        reranked = _retrieve_and_rerank(query, search_query, router_result, retrieval_span)
    except Exception as e:
        if retrieval_span is not None:
            try:
                retrieval_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    if retrieval_span is not None:
        try:
            retrieval_span.end(
                output={"n_results": len(reranked)},
            )
        except Exception:
            pass

    return reranked, router_result


def _retrieve_and_rerank(
    query: str,
    search_query: str,
    router_result: "RouterResult",
    retrieval_span=None,
) -> list[dict]:
    """Internal helper: run the intent-specific retrieval + reranking."""

    # Low confidence → fall back to broad hybrid search
    if router_result.confidence < 0.5:
        results = search.hybrid_search(search_query, k=config.RETRIEVAL_TOP_K, parent_span=retrieval_span)
        return reranker.rerank(search_query, results, parent_span=retrieval_span)

    # --- Intent-specific retrieval ---

    if router_result.intent == "policy_lookup" and router_result.policy_name:
        policy = search.get_policy(router_result.policy_name)
        if policy:
            policy.pop("_id", None)
            return [policy]
        # Fall through to hybrid if not found

    if router_result.intent == "aggregation_query":
        course_id = router_result.course_ids[0] if router_result.course_ids else None
        agg_results = search.aggregate_query(
            category=router_result.category or "",
            course_id=course_id,
        )
        if agg_results:
            return agg_results

    if router_result.intent == "multi_course_comparison" and len(router_result.course_ids) >= 2:
        results = search.multi_course_retrieve(
            query=search_query,
            course_ids=router_result.course_ids,
            category=router_result.category,
            k_per_course=config.RETRIEVAL_TOP_K // len(router_result.course_ids),
            parent_span=retrieval_span,
        )
        course_groups: dict[str, list[dict]] = {}
        for doc in results:
            cid = doc.get("course_id", "unknown")
            course_groups.setdefault(cid, []).append(doc)
        return reranker.rerank_multi_course(
            search_query, course_groups, top_k_per_course=3, parent_span=retrieval_span
        )

    # Single-course intents: schedule, instructor, single_course_lookup
    if router_result.intent in ("single_course_lookup", "schedule_query", "instructor_query"):
        filters: dict = {}
        if router_result.course_ids:
            filters["course_id"] = router_result.course_ids[0]
        if router_result.category:
            filters["category"] = router_result.category
        if router_result.intent == "schedule_query" and not router_result.category:
            filters["category"] = "SCHEDULE"
        elif router_result.intent == "instructor_query" and not router_result.category:
            filters["category"] = "INSTRUCTOR"

        results = search.hybrid_search(
            search_query,
            filters=filters if filters else None,
            k=config.RETRIEVAL_TOP_K,
            parent_span=retrieval_span,
        )
        return reranker.rerank(search_query, results, parent_span=retrieval_span)

    # general_academic or any fallthrough
    filters = {}
    if router_result.course_ids:
        filters["course_id"] = router_result.course_ids[0]
    results = search.hybrid_search(
        search_query,
        filters=filters if filters else None,
        k=config.RETRIEVAL_TOP_K,
        parent_span=retrieval_span,
    )
    return reranker.rerank(search_query, results, parent_span=retrieval_span)
