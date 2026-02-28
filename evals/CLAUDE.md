# evals/

## Probe (ad-hoc iteration)

```bash
python evals/run_probe.py --query "What is the grading for CSCE 638?"
python evals/run_probe.py --test-ids 1 3 7
python evals/run_probe.py --suite all --tag "generator_v2"
python evals/run_probe.py --test-ids 1 --ragas   # adds RAGAS faithfulness+relevancy (~30s)
```

Prints trace URL. Calls `lf.flush()` after each query. Gate 2 groundedness fires automatically.

## Eval Workflow (run in order)

```bash
python evals/generate_golden_set.py
python evals/eval_router_metrics.py --golden-set tamu_data/logs/golden_set.jsonl
python evals/eval_pipeline.py --golden-set tamu_data/logs/golden_set.jsonl [--ragas]
python evals/adjudicate_golden_set.py --golden-set tamu_data/logs/golden_set.jsonl --router-results tamu_data/logs/router_metrics.json --output tamu_data/logs/golden_set_v2.jsonl
python evals/eval_router_metrics.py --golden-set tamu_data/logs/golden_set_v2.jsonl   # true accuracy
python evals/eval_retrieval_metrics.py --golden-set tamu_data/logs/golden_set.jsonl [--rrf-sweep]
python evals/eval_generator_tiered.py --question "..." --answer "..." --reference "..." --n-sources 3
```

## Key Exports

```python
from evals.eval_pipeline import TEST_SUITE, TestCase   # NOT TEST_CASES — that name does not exist
# TestCase: query, function_expected, expected_course_ids, expected_specific_categories,
#           expected_semantic_intent, notes, source_crn, source_category, reference_answer

from evals.eval_statistics import adjusted_wald_ci, mcnemar_exact, wilcoxon_test, eval_summary_table
run_probe(query, tag, session_id, ragas, index, total) -> dict   # from run_probe.py
```

## Known Issues

- **Golden set ~10 label errors**: run adjudication before trusting router accuracy (74% raw → ~90% est.)
- **Recall@k 36%**: CRN-exact match counts cross-section hits as misses → redefine as `course_id + category`
