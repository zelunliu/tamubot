---
description: Guided benchmark runner — lists golden sets, proposes experiment name, clears sandbox cache, runs benchmark
triggers: ["run eval", "run benchmark", "run evals", "benchmark the pipeline", "benchmark rag"]
---

# Run Eval Skill

Announce: "Using run-eval skill to set up and run the benchmark."

## Step 1 — Discover golden sets

List all `.jsonl` files in `tamu_data/evals/golden_sets/`:

```bash
ls tamu_data/evals/golden_sets/*.jsonl
```

Display them numbered. Show question count for each with `wc -l <file>`:

```
Available golden sets:
  [1] golden_20260313_draft_v1_sample10.jsonl   (10 questions)
  [2] golden_20260313_draft_v1.jsonl            (60 questions)
```

## Step 2 — Present all settings in one block

Propose settings together. Use today's date (YYYYMMDD). Default golden set = most recently modified file. Default experiment name = `{stem}_v4_{YYYYMMDD}`.

```
Ready to run benchmark:

  Golden set:       [1] golden_20260313_draft_v1_sample10.jsonl
  Experiment name:  golden_20260313_draft_v1_sample10_v4_20260318
  RAGAS:            no
  Cache clear:      yes (sandbox-down + sandbox-up, ~25s)

Confirm, or tell me what to change.
```

Wait for user confirmation. If user edits any field, update and confirm again before proceeding.

## Step 3 — Clear sandbox cache (always, no prompt)

```bash
make sandbox-down && make sandbox-up
```

Wait for both commands to complete before proceeding.

## Step 4 — Run benchmark

```bash
docker exec tamubot-claude-1 bash -c "cd /workspace && python evals/run_benchmark.py \
  --golden-set tamu_data/evals/golden_sets/<golden_file> \
  --experiment-name <experiment_name> [--ragas]"
```

Stream output to user as it runs.

## Step 5 — Show results

After completion, print:
- Router accuracy
- Error count (if any)
- Paths to `.xlsx` and `.md` reports

If RAGAS was **not** run, remind:
```
To add RAGAS scores later:
  python evals/validate_ragas.py --benchmark <xlsx_path>
```
