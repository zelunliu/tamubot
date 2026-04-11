---
name: golden-set
description: Use when creating, refining, or exporting a TamuBot golden evaluation set (JSONL + Excel)
triggers: ["create golden set", "generate golden set", "make eval set", "refine golden set", "refine reference answers", "build eval set", "update golden set", "export golden set"]
---

# Golden Set Skill

Announce: "Using golden-set skill."

## Tasks

| Task | Command |
|---|---|
| Generate draft questions from MongoDB | `python evals/generate_eval_draft.py --n 60` → `tamu_data/evals/drafts/eval_draft_YYYYMMDD.xlsx` |
| Import human-approved draft → JSONL | `make import-draft DRAFT=<xlsx> TAG=v1` → `tamu_data/evals/golden_sets/golden_*.jsonl` |
| Refine reference answers + export | `python evals/refine_golden_references.py --input <jsonl> --output <jsonl> --excel` |
| Export existing JSONL → Excel only | `python evals/refine_golden_references.py --input <jsonl> --excel-only` |

Most common task: **refine** — loads full syllabus markdown per course, calls TAMU API for answers, deduplicates, assigns sequential IDs.

## Refine flags

| Flag | Effect |
|---|---|
| `--excel` | Also write `.xlsx` alongside JSONL |
| `--excel-only` | Clean + number + export, no API calls |
| `--dry-run` | Build prompts only, no API calls |
| `--overwrite` | Write output to same path as input |

## File naming

`tamu_data/evals/golden_sets/golden_YYYYMMDD_v<N>.jsonl` — today's date, increment version.

## What refine removes automatically

- Duplicate questions (keeps first)
- Empty answers, `(out of scope)`, or no-info patterns
- Syllabus source: `tamu_data/processed/v3_step1_markdown/`
