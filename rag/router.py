"""Query router — extracts structured variables from user questions and orchestrates retrieval + reranking.

Stage 1 of the 3-stage RAG pipeline: Router/Inlet → Retrieval+Rerank → Generator/Outlet.

Uses Gemini 2.5 Flash for structured variable extraction (course IDs, categories,
intent type) with query rewriting/expansion.  The retrieval function is derived
mechanically from the extracted variables — there is no intent classification step.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import config
from rag.llm_client import call_llm
from rag.prompts import ROUTER_PROMPT

# ---------------------------------------------------------------------------
# Dynamic-k helper (pure Python, no LLM)
# ---------------------------------------------------------------------------

def compute_dynamic_k(function: str, n_courses: int) -> dict[str, int]:
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
    recurrent_search: bool,
) -> str:
    """Derive retrieval_mode from course presence and recurrent_search flag.

    - No course IDs → "semantic" (full-corpus vector search, no anchor)
    - recurrent_search=True → "hybrid" (two-stage: anchor metadata fetch + corpus hybrid)
    - course IDs, no recurrent_search → "metadata" (exact index lookup, bypass vector)
    """
    if not course_ids:
        return "semantic"
    if recurrent_search:
        return "hybrid"
    return "metadata"


def _derive_function(
    course_ids: list[str],
    recurrent_search: bool,
    intent_type: Optional[str],
    specific_categories: list[str],
    specific_only: bool,
) -> str:
    """Derive the retrieval function name from extracted variables.

    Function matrix:
    course_ids  recurrent_search  intent_type  specific_categories  specific_only  → function
    empty       any               not None     any                  any            → semantic_general
    empty       any               None         any                  any            → out_of_scope
    present     True              any          empty                —              → recurrent_default
    present     True              any          populated            True           → recurrent_specific
    present     True              any          populated            False          → recurrent_combined
    present     False             any          empty                —              → metadata_default
    present     False             any          populated            True           → metadata_specific
    present     False             any          populated            False          → metadata_combined
    """
    if not course_ids:
        return "semantic_general" if intent_type is not None else "out_of_scope"

    if recurrent_search:
        if not specific_categories:
            return "recurrent_default"
        return "recurrent_specific" if specific_only else "recurrent_combined"
    else:
        if not specific_categories:
            return "metadata_default"
        return "metadata_specific" if specific_only else "metadata_combined"


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
    intent_type: Optional[str] = None  # None = out_of_scope only
    category_confidence: float = 0.0
    recurrent_search: bool = False
    rewritten_query: str = ""
    section: Optional[str] = None

    # Derived in Python — auto-computed in __post_init__ if left empty
    retrieval_mode: str = ""
    function: str = ""

    def __post_init__(self):
        if not self.function:
            self.function = _derive_function(
                self.course_ids,
                self.recurrent_search,
                self.intent_type,
                self.specific_categories,
                self.specific_only,
            )
        if not self.retrieval_mode:
            self.retrieval_mode = _derive_retrieval_mode(
                self.course_ids, self.recurrent_search
            )

    @property
    def requires_retrieval(self) -> bool:
        return bool(self.course_ids) or self.intent_type is not None


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
# For recurrent_* functions, categories are used for the anchor course metadata fetch (Stage 1).
# Adding a new retrieval function = adding one entry here; no logic change elsewhere.
FUNCTION_CATEGORY_STRATEGIES: dict[str, Callable[["RouterResult"], list[str]]] = {
    "metadata_default":   lambda r: list(config.DEFAULT_SUMMARY_CATEGORIES),
    "metadata_specific":  lambda r: r.specific_categories or list(config.DEFAULT_SUMMARY_CATEGORIES),
    "metadata_combined":  _get_combined_categories,
    "recurrent_default":  lambda r: list(config.DEFAULT_SUMMARY_CATEGORIES),
    "recurrent_specific": lambda r: r.specific_categories or list(config.DEFAULT_SUMMARY_CATEGORIES),
    "recurrent_combined": _get_combined_categories,
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
                router_span.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise

    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError, AttributeError):
        result = RouterResult(rewritten_query=query, category_confidence=0.0)
        if router_span is not None:
            try:
                router_span.end(
                    usage=(
                        {"input": llm_result.input_tokens, "output": llm_result.output_tokens}
                        if llm_result and llm_result.input_tokens is not None
                        else None
                    ),
                    metadata={
                        "function": result.function,
                        "course_ids": [],
                        "parse_error": True,
                    },
                )
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

    valid_intent_types = {"ACADEMIC", "CAREER", "DIFFICULTY", "PLANNING", "ADMINISTRATIVE", "GENERAL"}
    intent_type = data.get("intent_type")
    if intent_type not in valid_intent_types:
        intent_type = None

    result = RouterResult(
        course_ids=course_ids,
        specific_categories=specific_categories,
        specific_only=bool(data.get("specific_only", False)),
        intent_type=intent_type,
        category_confidence=float(data.get("category_confidence", 0.0)),
        recurrent_search=bool(data.get("recurrent_search", False)),
        rewritten_query=data.get("rewritten_query", query),
        section=data.get("section"),
        # function and retrieval_mode auto-derived in __post_init__
    )

    # End the router observation with usage counts and parsed output
    if router_span is not None:
        try:
            router_span.end(
                output=data,
                usage=(
                    {"input": llm_result.input_tokens, "output": llm_result.output_tokens}
                    if llm_result and llm_result.input_tokens is not None
                    else None
                ),
                metadata={
                    "function": result.function,
                    "retrieval_mode": result.retrieval_mode,
                    "course_ids": result.course_ids,
                    "specific_categories": result.specific_categories,
                    "intent_type": result.intent_type,
                    "category_confidence": result.category_confidence,
                    "recurrent_search": result.recurrent_search,
                    "rewritten_query": result.rewritten_query,
                    "thinking_tokens": llm_result.thinking_tokens if llm_result else None,
                },
            )
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Orchestrator: route → retrieve → rerank
# ---------------------------------------------------------------------------

def route_retrieve_rerank(
    query: str,
    trace=None,
) -> tuple[list[dict], "RouterResult", list[tuple[str, str]], bool]:
    """Backward-compat wrapper — delegates to pipeline.run_pipeline()."""
    from rag.pipeline import run_pipeline  # lazy import to avoid circular
    return run_pipeline(query, trace=trace)


def deduplicate_chunks(results: list[dict]) -> list[dict]:
    """Keep only the highest-scored chunk per (course_id, category) pair.

    The reranker returns results sorted best-first, so the first occurrence
    of each pair is the most relevant.  Deduplication prevents near-duplicate
    chunks (e.g. same GRADING content across 3 sections) from inflating the
    context and triggering Gemini's whitespace padding bug.
    """
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for doc in results:
        key = (doc.get("course_id", ""), doc.get("chunk_index", -1))
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped
