---
name: task-budget
description: Before any non-trivial task, announce estimated API call counts per service and stop if actuals exceed 2x estimate. Triggered automatically for tasks involving RAG queries, ingestion, benchmarks, or any operation that calls TAMU/Voyage/Google APIs.
---

# Task Budget Skill

**Trigger:** Any non-trivial task involving TAMU API, Voyage AI, or Google AI calls.

**No overhead for:** Simple file edits, git operations, refactors, docs-only changes.

## Step 1 — Announce estimate before starting

At the start of any API-touching task, output this block:

```
📊 API Budget Estimate
  TAMU API:    ~N calls  (generation/routing)
  Voyage AI:   ~N calls  (embeddings/reranking)
  Google AI:   ~N calls  (PDF parsing — ingestion only)
  Total cost:  low / medium / high
```

Use 0 for services not involved in the task.

**Estimation guidelines:**
- `make probe` (smoke suite, ~5 queries): TAMU ~10, Voyage ~10, Google 0
- `make probe-full` (all queries): TAMU ~50, Voyage ~50, Google 0
- Single RAG query (via app or run_probe): TAMU ~2, Voyage ~2, Google 0
- Ingestion (1 syllabus): TAMU ~5, Voyage ~1, Google ~3
- Benchmark run (50 questions): TAMU ~150, Voyage ~50, Google 0
- Eval generation (`eval-draft --n 60`): TAMU ~120, Voyage 0, Google 0

## Step 2 — Track actuals

Count API calls as the task executes. You don't need exact numbers — round to nearest 5.

## Step 3 — Stop if 2× exceeded

If actual calls for any service exceed **2× the estimate**, stop immediately and report:

```
⚠️  Budget exceeded: TAMU API used ~N calls (estimated ~E, limit 2×E = L)
Pausing. Confirm to continue or abort.
```

Do not proceed until the user confirms.

## When NOT to use this skill

- Git operations (commit, push, branch)
- File edits, refactors, formatting
- Tests that mock API calls
- Documentation changes
- Any task where you can confirm zero API calls will be made
