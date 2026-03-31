# LightRAG Iter 6-8 Findings
Date: 2026-03-28

## Comparison Table

| Metric | Iter4 (baseline) | Iter6 | Iter7 | Iter8 (multi) |
|--------|-----------------|-------|-------|---------------|
| Total entities | 94 | 96 | 95 | 542 |
| Total relationships | 83 | 48 | 48 | 421 |
| Fact coverage | 18/18 | 18/18 | 18/18 | 18/18 |
| "The Course" entity present | yes | no | no | no |
| Name-anchored % | ? | 11.5% | 11.6% | 10.9% |
| Case-drift dups merged | N/A | N/A | 1 ("Learning To Rank") | 2 ("Learning To Rank", "CSCE 638 NLP title") |
| Avg context tokens (5 queries) | ~7000 | ~6854 | ~6841 | ~16170 (cross-course) |

## Iter6 Observations

- "The Course" entity ABSENT — suppression rule worked perfectly.
- Fact coverage 18/18 — all key course facts preserved.
- CourseTopic anchoring mixed: some topics well-anchored (e.g., "CSCE 670 Boolean Retrieval") but many remain unanchored (e.g., "BM25", "TF-IDF", "Link Analysis"). Name-anchored % = 11.5%, below the >80% target.
- Case-drift dups remain: "Learning To Rank" vs "Learning to Rank" — Title Case rule not consistently obeyed.
- Dual-anchor issue: Some topics appear both anchored ("Information Retrieval Techniques Course Boolean Retrieval") and unanchored ("Boolean Retrieval") — LLM anchors using the header phrase rather than the CSCE 670 ID.
- Relationship count dropped: 83 -> 48 vs iter4 baseline.
- Novel entity types introduced: 7 new types (person, location, programmingconcept, tool, method, programmingartifact, academicdeliverable).
- Context tokens: ~6,854 avg (5 queries), close to iter4 baseline.

## Iter7 Observations

- gleaning=3 hit LightRAG LLM response cache — chunk hashes matched iter6, so gleaning=3 produced identical extraction to gleaning=2. No new entities/relations.
- Normalization merged 1 entity: "Learning To Rank" -> "Learning to Rank" (lower-degree into higher-degree).
- Post-normalization: 95 entities (down 1 from 96), 48 relationships unchanged.
- Fact coverage 18/18 maintained.
- Context tokens: ~6,841 avg (virtually identical to iter6).

## Iter8 Observations

- 542 entities, 421 relationships across 3 courses. Scale is ~5.7x single-course (CSCE 605 alone produced ~355 entities from 14 chunks / 24,707 chars).
- 2 case-drift dups normalized: "Learning To Rank" and "CSCE 638 Natural Language Processing: Foundations And Techniques".
- Name-anchored %: 10.9% — consistently low across all iters.
- academicpolicy is the dominant type (109 entities) due to CSCE 605 extensive policy content.
- 44 novel entity types introduced across 3 courses.

Cross-course query results (top_k=15, related_chunks=2):
- "What is the grading policy for CSCE 670?" — ~18,368t — mostly CSCE 670 entities
- "What is the grading policy for CSCE 638?" — ~17,029t — CSCE 638 entities (shows CSCE 638 Quizzes)
- "Which courses cover neural retrieval or language models?" — ~11,254t — cross-course (shows "Large Language Models")
- "Who teaches CSCE 605?" — ~17,884t — partial bleed (CSCE 638 entities in preview)
- "What topics does CSCE 638 cover each week?" — ~16,315t — partial bleed (CSCE 670 Course Schedule in preview)

Cross-course discovery works for topic queries. Some bleed on instructor/topics queries.

## Recommended next steps

1. Fix anchoring compliance — LLM only anchors a subset of CourseTopic entities. Consider post-processing to prefix unanchored entities with course ID from their source chunk metadata.
2. Address LLM cache collision — gleaning=3 did not fire because chunk hashes matched iter6. To test, clear cache or add prompt variation before re-running iter7.
3. Reduce context tokens for multi-course — 18K tokens per query. Reduce top_k to 5-7 or implement per-course subgraph routing.
4. Bleed prevention — Per-course subgraph routing (filter by course ID prefix in entity names) would improve course-specific query precision.
5. CSCE 605 entity explosion — 355 entities from one course. Consider capping novel type introduction or adding a consolidation prompt pass.
