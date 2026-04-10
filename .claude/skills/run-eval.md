---
description: Guided eval runner — discovers golden sets, confirms settings, runs eval
triggers: ["run eval", "run benchmark", "run evals", "benchmark the pipeline", "benchmark rag", "run chunking eval"]
---

# Run Eval Skill

Announce: "Using run-eval skill."

## Step 1 — Discover golden sets

```bash
ls tamu_data/evals/golden_sets/*.jsonl
```

Display numbered with question counts (`wc -l`).

## Step 2 — Confirm settings

Propose in one block. Default golden set = most recently modified. Default experiment name = `{stem}_{YYYYMMDD}`.

```
Ready to run eval:

  Golden set:    [1] golden_20260313_draft_v1_sample10.jsonl (10 q)
  Eval type:     chunking          # chunking | full-pipeline
  Experiment:    chunk_600ov100_k7_20260410
  Description:   <goal or notes for this run, e.g. "test smaller chunks for precision">
  Metadata:      chunk_size=600, chunk_overlap=100, top_k=7, threshold=0.35
  Outputs:       precision_at_k, hit_rate_at_k, recall_at_k, retrieved_tokens
  RAGAS:         no

Confirm or edit.
```

- **Metadata** = all input variables (settings being varied). Ask user to fill if not provided.
- **Outputs** = metrics that will be measured (propose based on eval type; user can edit).
- **Description** = free-text goal/notes. Ask user to provide.

Wait for confirmation before proceeding.

## Step 3 — Run

**Chunking eval:**
```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && make eval-chunking \
  GOLDEN=<golden_file> EXP=<experiment> \
  CHUNK_SIZE=<n> CHUNK_OVERLAP=<n> TOP_K=<n> THRESHOLD=<n> \
  DESC='<description>' $(if ragas: RAGAS=1)"
```

**Full-pipeline benchmark:**
```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && make bench \
  GOLDEN=<golden_file> EXP=<experiment> $(if ragas: --ragas)"
```

Stream output live.

## Step 4 — Report

Print: router accuracy, error count, Langfuse run URL if available, report paths.
