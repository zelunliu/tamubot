# evals/ — Evaluation Framework

> **Maintenance**: Update this file when eval scripts, metrics, or known issues change.

## Eval Workflow

```
1. generate_golden_set.py    → tamu_data/logs/golden_set.jsonl
2. eval_router_metrics.py    → tamu_data/logs/router_metrics.json    (router accuracy)
3. eval_pipeline.py          → tamu_data/evals/reports/              (end-to-end)
4. adjudicate_golden_set.py  → tamu_data/logs/golden_set_v2.jsonl    (fix label errors)
5. eval_router_metrics.py    (re-run on golden_set_v2 for true accuracy)
```

## Running Each Script

```bash
# Generate 50 stratified questions from live MongoDB chunks
python evals/generate_golden_set.py

# Evaluate router accuracy vs golden set
python evals/eval_router_metrics.py \
  --golden-set tamu_data/logs/golden_set.jsonl

# End-to-end pipeline eval (router → retrieval → generator)
python evals/eval_pipeline.py \
  --golden-set tamu_data/logs/golden_set.jsonl \
  [--ragas]   # add RAGAS faithfulness scoring (slower)

# Fix ~10 wrong golden set labels via LLM adjudication
python evals/adjudicate_golden_set.py \
  --golden-set tamu_data/logs/golden_set.jsonl \
  --router-results tamu_data/logs/router_metrics.json \
  --output tamu_data/logs/golden_set_v2.jsonl
```

## Metrics Reference

| Metric | What it measures |
|--------|-----------------|
| **ECE** | Expected Calibration Error — how well router confidence matches accuracy |
| **Intent F1** | Per-function F1 score across all 8 router functions |
| **Recall@k** | Whether the correct source chunk appears in the top-k retrieved results |
| **RAGAS Faithfulness** | Whether the generated answer is grounded in retrieved context |
| **RAGAS AnswerRelevancy** | Whether the answer addresses the question |

## Known Issue: ~10 Golden Set Label Errors

`generate_golden_set.py` assigns router function labels mechanically from question strata.
Synthesized questions that name a specific category get wrong function labels (~10 cases).

**Current reported accuracy: ~74%** — estimated true accuracy after adjudication: ~90%.

Run `adjudicate_golden_set.py` before trusting accuracy numbers.

## Recall@k Known Issue

Recall@k is 36% on first run due to CRN-exact matching. A correct answer from a different
section of the same course (same `course_id` + `category`) counts as a MISS. Needs
per-function analysis with looser match criterion (`course_id` + `category`, not CRN-exact).
