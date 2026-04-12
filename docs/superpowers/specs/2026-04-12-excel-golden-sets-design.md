# Excel-Native Golden Sets

**Date:** 2026-04-12
**Status:** Approved

## Problem

Golden sets are currently stored as JSONL files. The user edits them in Excel, which requires a manual conversion step (`import_eval_draft.py` Excel → JSONL) before running evals. This is unnecessary friction. Excel should be the single source of truth with no intermediate format.

## Goal

Read `.xlsx` directly in all eval scripts. Append per-run results as columns into the same Excel file. Remove all JSONL golden set files.

---

## Schema

Each golden set is a flat `.xlsx` file with one row per question. No list fields.

| Column | Type | Notes |
|---|---|---|
| `id` | int | Unique row identifier |
| `question` | str | Must end with `?` |
| `reference_answer` | str | Human-written ground truth |
| `expected_function` | str | `hybrid_course`, `semantic_general`, `recursive`, `out_of_scope` |
| `human_notes` | str / null | Free-text reviewer notes (replaces `human_judgment`) |
| `run:<experiment>` | str / null | Appended per eval run (see below) |

Run columns are appended automatically by eval scripts. Column name format: `run:<experiment_name>`.

**Chunking run value:** `CSCE 670 §SCHEDULE 0.87, CSCE 638 §SCHEDULE 0.71, ...`
**Full pipeline run value:** generated answer text (plain string)

---

## New Module: `evals/golden_set.py`

Three public functions:

### `load(path: Path) -> list[dict]`
Reads `.xlsx`, returns list of dicts with keys matching the schema above. Skips rows where `question` is empty. Handles null cells (returns `None`). Ignores `run:*` columns — those are for human review only, not used by eval logic.

### `save(items: list[dict], path: Path)`
Writes list of dicts to a new `.xlsx`. Columns written in schema order. Creates parent directories if needed. Used by `import_eval_draft.py` and migration script.

### `append_run_column(path: Path, experiment: str, results: dict[int, str])`
Opens existing `.xlsx`, adds or overwrites column `run:<experiment>`, matches rows by `id`. Saves in place. `results` is a dict mapping question `id` → value string.

---

## Changes to Existing Scripts

### `eval_chunking.py`
- Replace `load_golden_set()` with `golden_set.load()`
- After eval completes, call `golden_set.append_run_column()` with chunk ID + score strings
- Update `--golden-set` help text and docstring to reference `.xlsx`

### `run_benchmark.py`
- Replace JSONL loader with `golden_set.load()`
- After benchmark completes, call `golden_set.append_run_column()` with generated answer text

### `import_eval_draft.py`
- Change output from `.jsonl` to `.xlsx` using `golden_set.save()`
- Keep existing validation logic (required fields, question ends with `?`)
- Remove `stratum`, `category`, `source_crn`, `source_course_id`, `source_category`, `expected_course_ids`, `expected_specific_categories`, `expected_semantic_intent` from output schema (dropped fields)

---

## Migration

A one-off migration script converts existing JSONL → Excel and deletes the JSONL files:

- `golden_20260313_draft_v1.jsonl` → `.xlsx`
- `golden_20260313_draft_v1_sample10.jsonl` → `.xlsx`
- `golden_20260410_v2.jsonl` → `.xlsx`
- `golden_20260411_v1.jsonl` → `.xlsx` (user has already reworked this in Excel — copy their file, do not regenerate from JSONL)
- `CSCE_25gradcourses_nonrecursive.jsonl` → `.xlsx`
- `smoke_v1.jsonl` → `.xlsx`
- Delete throwaway subsets: `smoke_semantic_1q.jsonl`, `golden_20260411_v1_reworked.jsonl`

Migration maps `human_judgment` → `human_notes`. Dropped fields are silently ignored.

---

## Skill Update

`run-eval` skill: update `ls` command from `*.jsonl` to `*.xlsx`, update the confirmation block example metadata to reference `.xlsx`.

---

## Testing

- Unit tests for `golden_set.load()`: reads schema correctly, skips empty rows, ignores `run:*` columns
- Unit tests for `golden_set.save()`: round-trip (save then load returns same data)
- Unit tests for `golden_set.append_run_column()`: adds column, overwrites on re-run, matches by id
- Existing eval tests updated to use `.xlsx` fixtures instead of JSONL
