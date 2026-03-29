# Benchmark Report: sample10_v4_ragas_20260318

**Date:** 2026-03-18 07:05  
**Git commit:** 1fac367  
**Questions:** 10

## Summary

| Metric | Value |
|--------|-------|
| Router accuracy | 100.0% (10/10) |
| Citation pass rate | 100.0% (10/10) |
| Mean RAGAS faithfulness | 0.96 |
| Mean RAGAS relevancy | 0.50 |
| Mean chunks retrieved | 5 |
| Mean est. input tokens | 3200 |
| Mean est. output tokens | 342 |
| Mean total latency (ms) | 12458 |
| Mean pipeline latency (ms) | 8425 |
| Mean generator latency (ms) | 4033 |
| Mean router latency (ms) | 3174 |
| Mean retrieval latency (ms) | 416 |
| Errors | 0 |

## Router Accuracy by Stratum

| Stratum | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| metadata_combined | 1 | 1 | 100% |
| metadata_default | 5 | 5 | 100% |
| metadata_default_advisory | 2 | 2 | 100% |
| metadata_specific_evaluative | 1 | 1 | 100% |
| semantic_general | 1 | 1 | 100% |