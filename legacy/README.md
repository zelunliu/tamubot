# Legacy Routing — V1 Architecture Reference

Preserved snapshot of the pre-v3-rework routing system.

## Git reference

```bash
# Last commit with v1 routing intact
git show 29eb405 --stat

# Restore individual files to v1 state
git checkout 29eb405 -- rag/router.py rag/pipeline.py rag/prompts.py rag/generator.py rag/gates.py config.py
git checkout 29eb405 -- evals/generate_golden_set.py evals/eval_pipeline.py
```

## What changed in the v3 rework

| Aspect | V1 | V3 (current) |
|---|---|---|
| Functions | 8 (metadata_default/specific/combined, recurrent_default/specific/combined, semantic_general, out_of_scope) | 4 (hybrid_course, recurrent, semantic_general, out_of_scope) |
| Course query retrieval | `fetch_anchor_chunks(course_ids, categories)` — fetches all chunks matching a category list | `hybrid_search_v3(query, course_id=cid, k=retrieve_k)` — vector + BM25 filtered by course, top-k by relevance |
| Category role | Drove both function selection AND retrieval filtering | Router still extracts categories for generator prompt framing only |
| Reranking on course path | Skipped (`rerank_k=0`) | Always applied (cross-course rerank after per-course hybrid search) |
| "Is CSCE 638 worth taking?" | Routed to `metadata_default` (no vector search) | Routes to `hybrid_course` with `intent_type=DIFFICULTY` + thinking budget |

## Files in this folder

- `v1_routing/constants.py` — FUNCTION_RETRIEVAL_CONFIG, FUNCTION_CATEGORY_STRATEGIES, function derivation logic, pipeline flow comments
- `v1_routing/prompts_v1.py` — FUNCTION_PROMPTS_V1 (8 entries), STRATUM_MAP_V1 (eval strata)
