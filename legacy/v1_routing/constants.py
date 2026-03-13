"""V1 routing constants — preserved as reference for the pre-v3-rework architecture.

The v1 system used 8 routing functions derived from course_ids, recurrent_search,
specific_categories, and specific_only.  The live pipeline was updated to 4 functions
(hybrid_course, recurrent, semantic_general, out_of_scope) where metadata_* paths now
use per-course filtered hybrid search instead of fetch_anchor_chunks.

Git reference: commit 29eb405 (last state before the v3 routing rework)

To restore: git checkout 29eb405 -- rag/router.py rag/pipeline.py rag/prompts.py
            rag/generator.py rag/gates.py config.py
"""

# ---------------------------------------------------------------------------
# V1 retrieval config (8 functions)
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY_CATEGORIES: list[str] = [
    "COURSE_OVERVIEW", "PREREQUISITES", "LEARNING_OUTCOMES"
]

CATEGORY_CONFIDENCE_THRESHOLD: float = 0.7

# rerank_k=0 on metadata_* = no reranking (fetch_anchor_chunks returns exact results)
FUNCTION_RETRIEVAL_CONFIG: dict[str, dict[str, int]] = {
    "metadata_default":    {"retrieve_k": 10, "rerank_k": 0},
    "metadata_specific":   {"retrieve_k": 10, "rerank_k": 0},
    "metadata_combined":   {"retrieve_k": 10, "rerank_k": 0},
    "semantic_general":    {"retrieve_k": 30, "rerank_k": 10},
    "recurrent_default":   {"retrieve_k": 15, "rerank_k": 5},
    "recurrent_specific":  {"retrieve_k": 12, "rerank_k": 4},
    "recurrent_combined":  {"retrieve_k": 18, "rerank_k": 6},
}

MAX_RETRIEVE_K: int = 60
MAX_RERANK_K: int = 20

# ---------------------------------------------------------------------------
# V1 function derivation matrix (8 functions)
# ---------------------------------------------------------------------------
#
# course_ids  recurrent_search  intent_type  specific_categories  specific_only  → function
# empty       any               not None     any                  any            → semantic_general
# empty       any               None         any                  any            → out_of_scope
# present     True              any          empty                —              → recurrent_default
# present     True              any          populated            True           → recurrent_specific
# present     True              any          populated            False          → recurrent_combined
# present     False             any          empty                —              → metadata_default
# present     False             any          populated            True           → metadata_specific
# present     False             any          populated            False          → metadata_combined

def derive_function_v1(
    course_ids: list[str],
    recurrent_search: bool,
    intent_type,
    specific_categories: list[str],
    specific_only: bool,
) -> str:
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
# V1 FUNCTION_CATEGORY_STRATEGIES
#
# Maps each retrieval function to the categories fetched during the anchor pass.
# fetch_anchor_chunks(course_ids, categories) pulled only the named categories
# from the v1 chunks collection (which had a 'category' field per chunk).
# ---------------------------------------------------------------------------

def _get_combined_categories_v1(specific_categories: list[str]) -> list[str]:
    seen: set[str] = set()
    cats: list[str] = []
    for c in DEFAULT_SUMMARY_CATEGORIES + specific_categories:
        if c not in seen:
            seen.add(c)
            cats.append(c)
    return cats


FUNCTION_CATEGORY_STRATEGIES_V1: dict[str, list[str]] = {
    # metadata_* — anchor pass categories
    "metadata_default":   DEFAULT_SUMMARY_CATEGORIES,
    # metadata_specific / metadata_combined: dynamic (depends on specific_categories)
    # recurrent_* — same strategy, used for Step 2 anchor fetch
    "recurrent_default":  DEFAULT_SUMMARY_CATEGORIES,
}

# ---------------------------------------------------------------------------
# V1 generator temperatures
# ---------------------------------------------------------------------------

FUNCTION_TEMPERATURES_V1: dict[str, float] = {
    "metadata_default":   0.0,
    "metadata_specific":  0.0,
    "metadata_combined":  0.0,
    "semantic_general":   0.2,
    "recurrent_default":  0.2,
    "recurrent_specific": 0.2,
    "recurrent_combined": 0.2,
    "out_of_scope":       0.0,
}

# ---------------------------------------------------------------------------
# V1 thinking budgets
# ---------------------------------------------------------------------------

THINKING_BUDGET_METADATA_V1 = 0     # metadata_* — deterministic, no thinking
THINKING_BUDGET_SEMANTIC_V1  = 1024  # recurrent_*, semantic_general

SEMANTIC_THINKING_FUNCTIONS_V1 = frozenset([
    "recurrent_default", "recurrent_specific", "recurrent_combined", "semantic_general"
])

# ---------------------------------------------------------------------------
# V1 pipeline flow summary
# ---------------------------------------------------------------------------
#
# metadata_default / metadata_specific / metadata_combined:
#   1. fetch_anchor_chunks(course_ids, categories)  ← MongoDB exact category filter
#   2. reranker.rerank(query, anchor, top_k=rerank_k)  ← rerank_k=0 means skip rerank
#   3. deduplicate_chunks(reranked)
#
# recurrent_default / recurrent_specific / recurrent_combined:
#   1. Anchor_Pass: fetch_anchor_chunks(course_ids, categories)
#   2. EvalSearch_Stage: generate_eval_search_string(anchor, query, intent)
#   3. Discover_Pass: search_semantic(eval_query) → filter out anchor courses → rerank
#   4. deduplicate_chunks(anchor + discovery)
#
# semantic_general:
#   1. search_semantic(query, top_k=retrieve_k)
#   2. reranker.rerank(query, results, top_k=rerank_k)
#
# out_of_scope:
#   → canned response, no retrieval
