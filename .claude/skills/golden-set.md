---
name: golden-set
description: Create, refine, or export a TamuBot golden evaluation set (JSONL + Excel).
triggers: ["create golden set", "generate golden set", "make eval set", "refine golden set", "refine reference answers", "build eval set", "update golden set", "export golden set"]
---

# Golden Set Skill

Announce: "Using golden-set skill."

## Determine task

| Task | Command |
|---|---|
| Generate draft questions from MongoDB | `python evals/generate_eval_draft.py --n 60` → `tamu_data/evals/drafts/eval_draft_YYYYMMDD.xlsx` |
| Import human-approved draft → JSONL | `make import-draft DRAFT=<xlsx> TAG=v1` → `tamu_data/evals/golden_sets/golden_*.jsonl` |
| **Refine reference answers + export** | `python evals/refine_golden_references.py --input <jsonl> --output <jsonl> --excel` |
| Export existing JSONL → Excel only | `python evals/refine_golden_references.py --input <jsonl> --excel-only` |

The most common task is **refine**: it loads each course's full syllabus markdown, calls TAMU API to synthesize a proper answer, then removes duplicates + empty answers and assigns sequential `id`s.

## Refine flags

| Flag | Effect |
|---|---|
| `--excel` | Also write `.xlsx` alongside the JSONL |
| `--excel-only` | Clean + number + export, no API calls |
| `--dry-run` | Build prompts, no API calls |
| `--overwrite` | Write output to same path as input |

## File naming

`tamu_data/evals/golden_sets/golden_YYYYMMDD_v<N>.jsonl` — today's date, increment version.

## What refine does automatically

- Removes duplicate questions (keeps first)
- Removes entries where the answer is empty, `(out of scope)`, or contains no-info patterns
- Assigns sequential `id` field to remaining entries
- Syllabus source: `tamu_data/processed/v3_step1_markdown/`
