"""Query router — extracts structured variables from user questions and orchestrates retrieval + reranking.

Stage 1 of the 3-stage RAG pipeline: Router/Inlet → Retrieval+Rerank → Generator/Outlet.

Uses Gemini 2.5 Flash for structured variable extraction (course IDs, categories,
semantic intent) with query rewriting/expansion.  The retrieval function is derived
mechanically from the extracted variables — there is no intent classification step.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from google.genai import types

import config
from db import search, reranker


# ---------------------------------------------------------------------------
# Router prompt — structured variable extraction
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """\
You are a query parser for a Texas A&M University course assistant.
Extract structured variables from the user's question and emit JSON.

COURSE IDs
Identify all course IDs mentioned (e.g. "CSCE 638", "CSCE 670").
Normalize: uppercase department + space + number ("csce638" → "CSCE 638", "CSCE-670" → "CSCE 670").

CATEGORIES
Identify which syllabus categories the question is asking about.
Valid categories: COURSE_OVERVIEW, INSTRUCTOR, PREREQUISITES, LEARNING_OUTCOMES, MATERIALS,
GRADING, SCHEDULE, ATTENDANCE_AND_MAKEUP, AI_POLICY, UNIVERSITY_POLICIES, SUPPORT_SERVICES

- specific_categories: list of relevant categories (or [] if none are clearly targeted)
- specific_only: true if the question asks ONLY about those categories (not a general overview).
  False if the question is broad or requests a general overview with a category emphasis.
- category_confidence: 0.0–1.0 for confidence in the extracted categories.

Examples:
- "What is the grading breakdown for CSCE 638?"
  → specific_categories=["GRADING"], specific_only=true, category_confidence=0.95
- "Tell me about CSCE 670"
  → specific_categories=[], specific_only=false, category_confidence=1.0
- "Tell me about CSCE 638, especially the grading"
  → specific_categories=["GRADING"], specific_only=false, category_confidence=0.85
- "Can I use ChatGPT in CSCE 638?"
  → specific_categories=["AI_POLICY"], specific_only=true, category_confidence=0.95
- "What materials and grading does CSCE 638 require?"
  → specific_categories=["MATERIALS","GRADING"], specific_only=true, category_confidence=0.9

SEMANTIC INTENT
Determine if the question has a subjective, advisory, or opinion component.
IMPORTANT: semantic_intent only applies to TAMU academic questions. Non-TAMU questions
(weather, restaurants, cover letters, coding tasks unrelated to courses) must use
semantic_intent = false regardless of phrasing.

Set semantic_intent = true ONLY when the question is about TAMU academics AND:
- Asks for opinions, evaluations, or difficulty comparisons about specific courses
- Asks about career relevance or skill building ("good for ML career?", "worth taking?")
- Uses clearly evaluative language about a course: "hard", "strict", "fair", "useful", "worth it"
- Is a TAMU academic discovery query with NO specific course ID
  (e.g. "what courses cover ML?", "what is the TAMU academic integrity policy?",
   "what campus resources are available?")

Set semantic_intent = false for:
- Purely factual questions (what, when, who, list, how many) — even when comparing two courses
- Factual side-by-side comparisons of course policies, schedules, or grading structures
- Questions NOT about TAMU academics (weather, restaurants, non-academic tasks)
- Greetings and off-topic requests

Examples:
- "Compare the grading of CSCE 638 and CSCE 670" → semantic_intent=false (factual comparison)
- "Is CSCE 638 harder than CSCE 670?" → semantic_intent=true (evaluative/opinion)
- "What is the TAMU academic integrity policy?" → semantic_intent=true (TAMU discovery, no course_id)
- "What are the best restaurants near TAMU?" → semantic_intent=false (NOT TAMU academic)
- "Can you write a cover letter?" → semantic_intent=false (NOT TAMU academic)

If semantic_intent = true, set semantic_type to one of:
- ACADEMIC: Learning outcomes, topics covered, academic content, policies, campus resources
- CAREER: Job relevance, skill building, industry applications
- DIFFICULTY: Workload, how hard is it, grading rigor
- PLANNING: Which course to take, course sequence, scheduling
- GENERAL: Any other advisory/subjective component about a TAMU course

If semantic_intent = false, set semantic_type = null.

QUERY REWRITING
Rewrite the query with expanded synonyms for optimal retrieval:
- "late work" → "attendance makeup deadline extensions late submission"
- "ChatGPT" / "AI tools" → "AI policy artificial intelligence generative AI tools"
- "prereqs" → "prerequisites required courses corequisites"
- "prof" / "teacher" → "instructor professor"
- "grade breakdown" → "grading policy grade distribution weight percentage"
Keep the rewrite concise but include key synonyms.

Output ONLY a JSON object with these fields:
{{
  "course_ids": list of normalized course IDs or [],
  "section": section number string if mentioned, or null,
  "specific_categories": list of category strings or [],
  "specific_only": true or false,
  "category_confidence": float 0.0–1.0,
  "semantic_intent": true or false,
  "semantic_type": "ACADEMIC"|"CAREER"|"DIFFICULTY"|"PLANNING"|"GENERAL" or null,
  "rewritten_query": "expanded query string for retrieval"
}}

Respond with ONLY valid JSON, no other text.

User question: {query}
"""


# ---------------------------------------------------------------------------
# Derivation helpers (pure Python, no LLM)
# ---------------------------------------------------------------------------

def _derive_retrieval_mode(course_ids: list[str], category_confidence: float) -> str:
    """Derive retrieval_mode from course presence and category confidence.

    - No course IDs → "semantic" (full-corpus vector search)
    - course IDs + high confidence → "metadata" (exact index lookup)
    - course IDs + low confidence → "hybrid" (RRF of vector + text)
    """
    if not course_ids:
        return "semantic"
    if category_confidence >= config.CATEGORY_CONFIDENCE_THRESHOLD:
        return "metadata"
    return "hybrid"


def _derive_function(
    course_ids: list[str],
    semantic_intent: bool,
    specific_categories: list[str],
    specific_only: bool,
) -> str:
    """Derive the retrieval function name from extracted variables.

    Function matrix:
    course_ids  semantic_intent  specific_categories  specific_only  → function
    empty       True             any                  any            → semantic_general
    empty       False            any                  any            → out_of_scope
    present     False            empty                —              → metadata_default
    present     False            populated            True           → metadata_specific
    present     False            populated            False          → metadata_combined
    present     True             empty                —              → hybrid_default
    present     True             populated            True           → hybrid_specific
    present     True             populated            False          → hybrid_combined
    """
    if not course_ids:
        return "semantic_general" if semantic_intent else "out_of_scope"

    if not semantic_intent:
        if not specific_categories:
            return "metadata_default"
        return "metadata_specific" if specific_only else "metadata_combined"
    else:
        if not specific_categories:
            return "hybrid_default"
        return "hybrid_specific" if specific_only else "hybrid_combined"


# ---------------------------------------------------------------------------
# RouterResult — extracted variables + derived fields
# ---------------------------------------------------------------------------

@dataclass
class RouterResult:
    """Structured output from the query router."""

    # Extracted by router LLM
    course_ids: list[str] = field(default_factory=list)
    specific_categories: list[str] = field(default_factory=list)
    specific_only: bool = False
    semantic_intent: bool = False
    semantic_type: Optional[str] = None
    category_confidence: float = 0.0
    rewritten_query: str = ""
    section: Optional[str] = None

    # Derived in Python — auto-computed in __post_init__ if left empty
    retrieval_mode: str = ""
    function: str = ""

    def __post_init__(self):
        if not self.retrieval_mode:
            self.retrieval_mode = _derive_retrieval_mode(
                self.course_ids, self.category_confidence
            )
        if not self.function:
            self.function = _derive_function(
                self.course_ids,
                self.semantic_intent,
                self.specific_categories,
                self.specific_only,
            )

    @property
    def requires_retrieval(self) -> bool:
        return bool(self.course_ids) or self.semantic_intent


def _normalize_course_id(raw: str) -> str:
    """Normalize a course ID like 'csce638' → 'CSCE 638'."""
    raw = raw.strip().upper().replace("-", " ")
    match = re.match(r"^([A-Z]+)\s*(\d+.*)$", raw)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return raw


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_query(query: str, router_span=None) -> "RouterResult":
    """Extract structured variables from a user query using Gemini Flash.

    Args:
        query:       The raw user question.
        router_span: Optional Langfuse span to record router metadata into.
    """
    client = config.get_genai_client()

    prompt = ROUTER_PROMPT.format(query=query)

    try:
        response = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                max_output_tokens=1024,
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
        result = RouterResult(rewritten_query=query, category_confidence=0.0)
        if router_span is not None:
            try:
                router_span.update(metadata={
                    "function": result.function,
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
    course_ids = [_normalize_course_id(c) for c in raw_ids if c]

    # Validate specific_categories against known values
    valid_categories = {
        "COURSE_OVERVIEW", "INSTRUCTOR", "PREREQUISITES", "LEARNING_OUTCOMES",
        "MATERIALS", "GRADING", "SCHEDULE", "ATTENDANCE_AND_MAKEUP",
        "AI_POLICY", "UNIVERSITY_POLICIES", "SUPPORT_SERVICES",
    }
    specific_categories = [
        c for c in (data.get("specific_categories") or [])
        if c in valid_categories
    ]

    valid_semantic_types = {"ACADEMIC", "CAREER", "DIFFICULTY", "PLANNING", "GENERAL"}
    semantic_type = data.get("semantic_type")
    if semantic_type not in valid_semantic_types:
        semantic_type = None

    result = RouterResult(
        course_ids=course_ids,
        specific_categories=specific_categories,
        specific_only=bool(data.get("specific_only", False)),
        semantic_intent=bool(data.get("semantic_intent", False)),
        semantic_type=semantic_type,
        category_confidence=float(data.get("category_confidence", 0.0)),
        rewritten_query=data.get("rewritten_query", query),
        section=data.get("section"),
        # function and retrieval_mode auto-derived in __post_init__
    )

    # Record router metadata into the span
    if router_span is not None:
        try:
            usage = response.usage_metadata
            thinking_tokens = getattr(usage, "thoughts_token_count", None) or 0
            router_span.update(
                output=data,
                metadata={
                    "function": result.function,
                    "retrieval_mode": result.retrieval_mode,
                    "course_ids": result.course_ids,
                    "specific_categories": result.specific_categories,
                    "semantic_intent": result.semantic_intent,
                    "semantic_type": result.semantic_type,
                    "category_confidence": result.category_confidence,
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

    # No retrieval needed (out_of_scope or no course + no semantic intent)
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
                    "function": router_result.function,
                    "retrieval_mode": router_result.retrieval_mode,
                    "course_ids": router_result.course_ids,
                },
            )
        except Exception:
            retrieval_span = None

    try:
        reranked = _deduplicate_chunks(
            _retrieve_and_rerank(query, search_query, router_result, retrieval_span)
        )
    except Exception as e:
        if retrieval_span is not None:
            try:
                retrieval_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    if retrieval_span is not None:
        try:
            retrieval_span.end(output={"n_results": len(reranked)})
        except Exception:
            pass

    return reranked, router_result


def _deduplicate_chunks(results: list[dict]) -> list[dict]:
    """Keep only the highest-scored chunk per (course_id, category) pair.

    The reranker returns results sorted best-first, so the first occurrence
    of each pair is the most relevant.  Deduplication prevents near-duplicate
    chunks (e.g. same GRADING content across 3 sections) from inflating the
    context and triggering Gemini's whitespace padding bug.
    """
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for doc in results:
        key = (doc.get("course_id", ""), doc.get("category", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped


def _retrieve_and_rerank(
    query: str,
    search_query: str,
    router_result: "RouterResult",
    retrieval_span=None,
) -> list[dict]:
    """Internal helper: run the function-specific retrieval + optional reranking.

    Retrieval strategy:
    - metadata path (retrieval_mode="metadata"): search_by_course_categories, no reranking
    - hybrid path (retrieval_mode="hybrid"):     hybrid_search per course + reranking
    - semantic path (retrieval_mode="semantic"): search_semantic + reranking
    """
    fn = router_result.function
    mode = router_result.retrieval_mode
    cfg = config.FUNCTION_RETRIEVAL_CONFIG.get(fn, {})
    retrieve_k = cfg.get("retrieve_k", config.RETRIEVAL_TOP_K)
    rerank_k = cfg.get("rerank_k", config.RERANK_TOP_K)

    course_ids = router_result.course_ids
    specific_cats = router_result.specific_categories

    # ── semantic_general: no course ID, pure corpus-wide vector search ──────
    if fn == "semantic_general":
        results = search.search_semantic(search_query, top_k=retrieve_k)
        return reranker.rerank(search_query, results, top_k=rerank_k, parent_span=retrieval_span)

    # ── out_of_scope: should have been filtered upstream ────────────────────
    if fn == "out_of_scope" or not course_ids:
        return []

    # ── Determine which categories to fetch ─────────────────────────────────
    if fn in ("metadata_default", "hybrid_default"):
        categories = config.DEFAULT_SUMMARY_CATEGORIES
    elif fn in ("metadata_specific", "hybrid_specific"):
        categories = specific_cats or config.DEFAULT_SUMMARY_CATEGORIES
    elif fn in ("metadata_combined", "hybrid_combined"):
        # DEFAULT_SUMMARY + specific, deduped, order preserved
        seen_cats: set[str] = set()
        categories = []
        for c in (config.DEFAULT_SUMMARY_CATEGORIES + specific_cats):
            if c not in seen_cats:
                seen_cats.add(c)
                categories.append(c)
    else:
        categories = config.DEFAULT_SUMMARY_CATEGORIES

    # ── Per-course fetch helper ──────────────────────────────────────────────
    def fetch_course(cid: str) -> list[dict]:
        if mode == "metadata":
            return search.search_by_course_categories(cid, categories)
        # hybrid path
        filters: dict = {"course_id": cid}
        return search.hybrid_search(
            search_query, filters=filters, k=retrieve_k, parent_span=retrieval_span
        )

    # ── Single-course path ───────────────────────────────────────────────────
    if len(course_ids) == 1:
        results = fetch_course(course_ids[0])
        if mode == "metadata":
            return results  # exact lookup — no reranking needed
        return reranker.rerank(search_query, results, top_k=rerank_k, parent_span=retrieval_span)

    # ── Multi-course path ────────────────────────────────────────────────────
    course_groups: dict[str, list[dict]] = {cid: fetch_course(cid) for cid in course_ids}

    if mode == "metadata":
        # No reranking — interleave results across courses
        combined: list[dict] = []
        for cid in course_ids:
            combined.extend(course_groups[cid])
        return combined

    return reranker.rerank_multi_course(
        search_query, course_groups, top_k_per_course=rerank_k, parent_span=retrieval_span
    )
