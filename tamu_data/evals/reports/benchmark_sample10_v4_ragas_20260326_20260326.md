# Benchmark Report: sample10_v4_ragas_20260326

**Date:** 2026-03-26 07:01  
**Git commit:** 2acdf5d  
**Questions:** 10

## Summary

| Metric | Value |
|--------|-------|
| Router accuracy | 100.0% (10/10) |
| Citation pass rate | 100.0% (10/10) |
| Mean RAGAS faithfulness | 0.87 |
| Mean RAGAS relevancy | 0.50 |
| Mean chunks retrieved | 5 |
| Mean est. input tokens | 3200 |
| Mean est. output tokens | 186 |
| Mean total latency (ms) | 11441 |
| Mean pipeline latency (ms) | 8659 |
| Mean generator latency (ms) | 2782 |
| Mean router latency (ms) | 3914 |
| Mean retrieval latency (ms) | 576 |
| Errors | 0 |

## Router Accuracy by Stratum

| Stratum | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| metadata_combined | 1 | 1 | 100% |
| metadata_default | 5 | 5 | 100% |
| metadata_default_advisory | 2 | 2 | 100% |
| metadata_specific_evaluative | 1 | 1 | 100% |
| semantic_general | 1 | 1 | 100% |