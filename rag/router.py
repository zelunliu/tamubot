"""Query router — extracts structured variables from user questions and orchestrates retrieval + reranking.

Stage 1 of the 3-stage RAG pipeline: Router/Inlet → Retrieval+Rerank → Generator/Outlet.

Uses Gemini 2.5 Flash for structured variable extraction (course IDs, categories,
semantic intent) with query rewriting/expansion.  The retrieval function is derived
mechanically from the extracted variables — there is no intent classification step.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import config
from rag import search, reranker
from rag.llm_client import call_llm
from rag.prompts import ROUTER_PROMPT


# ---------------------------------------------------------------------------
# Dynamic-k helper (pure Python, no LLM)
# ---------------------------------------------------------------------------

def _compute_dynamic_k(function: str, n_courses: int) -> dict[str, int]:
    """Compute retrieve_k and rerank_k scaled by the number of courses in the query.

    semantic_general is corpus-wide — do not scale by course count.
    All other functions multiply their per-course base by n_courses, capped at the
    global maximums to avoid over-retrieving.
    """
    base = config.PER_COURSE_K[function]
    if function == "semantic_general":
        return dict(base)  # fixed, not scaled
    n = max(1, n_courses)
    return {
        "retrieve_k": min(base["retrieve_k"] * n, config.MAX_RETRIEVE_K),
        "rerank_k": min(base["rerank_k"] * n, config.MAX_RERANK_K),
    }


# ---------------------------------------------------------------------------
# Derivation helpers (pure Python, no LLM)
# ---------------------------------------------------------------------------

def _derive_retrieval_mode(
    course_ids: list[str],
    category_confidence: float,
    function: str = "",
) -> str:
    """Derive retrieval_mode from course presence, category confidence, and function.

    - No course IDs → "semantic" (full-corpus vector search)
    - *_combined functions → always "hybrid" regardless of confidence.
      Broad framing ("what should I know about X, especially Y") benefits from
      the semantic component even when the named category has high confidence.
    - course IDs + high confidence → "metadata" (exact index lookup)
    - course IDs + low confidence → "hybrid" (RRF of vector + text)
    """
    if not course_ids:
        return "semantic"
    if function.endswith("_combined"):
        return "hybrid"
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
        if not self.function:
            self.function = _derive_function(
                self.course_ids,
                self.semantic_intent,
                self.specific_categories,
                self.specific_only,
            )
        if not self.retrieval_mode:
            self.retrieval_mode = _derive_retrieval_mode(
                self.course_ids, self.category_confidence, self.function
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
# Retrieval category registry (pure Python, no LLM)
# ---------------------------------------------------------------------------

def _get_combined_categories(router_result: "RouterResult") -> list[str]:
    """DEFAULT_SUMMARY + specific categories, deduped, order preserved."""
    seen: set[str] = set()
    cats: list[str] = []
    for c in config.DEFAULT_SUMMARY_CATEGORIES + router_result.specific_categories:
        if c not in seen:
            seen.add(c)
            cats.append(c)
    return cats


# Maps each retrieval function to a callable that returns the categories to fetch.
# Adding a new retrieval function = adding one entry here; no logic change elsewhere.
_FUNCTION_CATEGORY_STRATEGIES: dict[str, Callable[["RouterResult"], list[str]]] = {
    "metadata_default":  lambda r: list(config.DEFAULT_SUMMARY_CATEGORIES),
    "metadata_specific": lambda r: r.specific_categories or list(config.DEFAULT_SUMMARY_CATEGORIES),
    "metadata_combined": _get_combined_categories,
    "hybrid_default":    lambda r: list(config.DEFAULT_SUMMARY_CATEGORIES),
    "hybrid_specific":   lambda r: r.specific_categories or list(config.DEFAULT_SUMMARY_CATEGORIES),
    "hybrid_combined":   _get_combined_categories,
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_query(query: str, router_span=None) -> "RouterResult":
    """Extract structured variables from a user query using Gemini Flash.

    Args:
        query:       The raw user question.
        router_span: Optional Langfuse span to record router metadata into.
    """
    prompt = ROUTER_PROMPT.format(query=query)

    llm_result = None
    try:
        llm_result = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
            json_mode=True,
            thinking_budget=512,
        )
        raw_text = llm_result.text
    except Exception as e:
        if router_span is not None:
            try:
                router_span.update(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    try:
        data = json.loads(raw_text)
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

    valid_semantic_types = {"ACADEMIC", "CAREER", "DIFFICULTY", "PLANNING", "ADMINISTRATIVE", "GENERAL"}
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
                    "input_tokens": llm_result.input_tokens if llm_result else None,
                    "output_tokens": llm_result.output_tokens if llm_result else None,
                    "thinking_tokens": llm_result.thinking_tokens if llm_result else None,
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
    dk = _compute_dynamic_k(fn, len(router_result.course_ids))
    retrieve_k = dk["retrieve_k"]
    rerank_k = dk["rerank_k"]

    course_ids = router_result.course_ids

    # ── semantic_general: no course ID, pure corpus-wide vector search ──────
    if fn == "semantic_general":
        results = search.search_semantic(search_query, top_k=retrieve_k)
        return reranker.rerank(search_query, results, top_k=rerank_k, parent_span=retrieval_span)

    # ── out_of_scope: should have been filtered upstream ────────────────────
    if fn == "out_of_scope" or not course_ids:
        return []

    # ── Determine which categories to fetch (registry lookup) ───────────────
    strategy = _FUNCTION_CATEGORY_STRATEGIES.get(fn)
    categories = strategy(router_result) if strategy else list(config.DEFAULT_SUMMARY_CATEGORIES)

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
