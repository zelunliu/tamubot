# Benchmark Report: sample10_v4_ragas_20260326b

**Date:** 2026-03-26 07:29  
**Git commit:** 2acdf5d  
**Questions:** 10

## Summary

| Metric | Value |
|--------|-------|
| Router accuracy | 100.0% (10/10) |
| Citation pass rate | 100.0% (10/10) |
| Mean RAGAS faithfulness | 0.97 |
| Mean RAGAS relevancy | 0.55 |
| Mean chunks retrieved | 5 |
| Mean est. input tokens | 3200 |
| Mean est. output tokens | 298 |
| Mean total latency (ms) | 14628 |
| Mean pipeline latency (ms) | 10571 |
| Mean generator latency (ms) | 4056 |
| Mean router latency (ms) | 3705 |
| Mean retrieval latency (ms) | 422 |
| Errors | 0 |

## Router Accuracy by Stratum

| Stratum | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| metadata_combined | 1 | 1 | 100% |
| metadata_default | 5 | 5 | 100% |
| metadata_default_advisory | 2 | 2 | 100% |
| metadata_specific_evaluative | 1 | 1 | 100% |
| semantic_general | 1 | 1 | 100% |