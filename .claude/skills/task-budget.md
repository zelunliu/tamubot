---
name: task-budget
description: Use before any task involving TAMU API, Voyage AI, or Google AI calls — RAG queries, probes, ingestion, benchmarks, or eval generation
---

# Task Budget Skill

**Skip for:** file edits, git ops, refactors, mocked tests, docs-only changes.

## Step 1 — Announce estimate before starting

```
API Budget Estimate
  TAMU API:    ~N calls  (generation/routing)
  Voyage AI:   ~N calls  (embeddings/reranking)
  Google AI:   ~N calls  (PDF parsing — ingestion only)
  Total cost:  low / medium / high
```

**Reference estimates:**

| Task | TAMU | Voyage | Google |
|---|---|---|---|
| Single RAG query | ~2 | ~2 | 0 |
| `make probe` (smoke, ~5 q) | ~10 | ~10 | 0 |
| `make probe-full` (all q) | ~50 | ~50 | 0 |
| Ingestion (1 syllabus) | ~5 | ~1 | ~3 |
| Benchmark (50 q) | ~150 | ~50 | 0 |
| Eval draft (`--n 60`) | ~120 | 0 | 0 |

## Step 2 — Stop if 2× exceeded

If actual calls for any service exceed **2× the estimate**:

```
Budget exceeded: TAMU API used ~N calls (estimated ~E, limit 2×E = L)
Pausing. Confirm to continue or abort.
```

Do not proceed until user confirms.
