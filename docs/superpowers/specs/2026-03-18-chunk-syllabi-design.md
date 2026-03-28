# Design: chunk-syllabi script + skill

**Date:** 2026-03-18
**Status:** Approved

---

## Overview

A standalone chunking script and companion Claude skill for re-chunking v3 syllabus pipeline output at configurable token sizes. Operates purely on already-processed `_stripped.md` files — no LLM calls, no API cost. Output is for **experimentation and inspection only**, not for direct ingestion into the vector store.

---

## Background

The v3 pipeline currently produces `v3_step3_flat/` with `chunk_size=600, overlap=100` via an LLM-enriched step 3. We need a lightweight way to experiment with different chunk sizes (e.g., 300/50) without re-running the expensive LLM enrichment. The existing `chunker_v3.py` already implements the core algorithm; this work adds a scriptable entry point and skill wrapper.

The app reads chunks from MongoDB Atlas (ingested separately) — renaming `v3_step3_flat/` on disk does not affect the running application.

---

## Data Layout

### Folder naming convention
All chunked output folders use the pattern:
```
tamu_data/processed/v3_result_<size>t_<overlap>o_<YYYYMMDD>/
```

Examples:
- `v3_result_600t_100o_20260311/` — existing production chunks (renamed from `v3_step3_flat/`)
- `v3_result_300t_50o_20260318/` — new 300-token experiment

### One-time migration (manual, pre-requisite — not part of the skill)

**Step A — Archive legacy formats:**
```
tamu_data/processed/gem_parsed_20260304/  →  tamu_data/processed/legacy/
tamu_data/processed/gem_parsed_20260306/  →  tamu_data/processed/legacy/
tamu_data/processed/gemini_parsed/        →  tamu_data/processed/legacy/
tamu_data/processed/v2_chunked/           →  tamu_data/processed/legacy/
tamu_data/processed/v2_filtered/          →  tamu_data/processed/legacy/
```

**Step B — Rename existing step 3 output to standardized name:**
```
tamu_data/processed/v3_step3_flat/  →  tamu_data/processed/v3_result_600t_100o_20260311/
```
Date `20260311` comes from `_run_meta_v010.json` → `created_at: 2026-03-11`.

**Note:** `pipeline_logger.py` hardcodes `STEP3_V3_ROOT = Path("tamu_data/processed/v3_step3_flat")`. After migration, the full pipeline (`process_syllabi_v3.py`) still writes to that path (new runs re-create the folder). The `chunk_syllabi.py` script writes to the new `v3_result_*` convention exclusively.

---

## Script: `ingestion_pipeline/chunk_syllabi.py`

### Input
- Source: `tamu_data/processed/v3_step2_boilerplate/*_stripped.md`
- File selection order: **alphabetical** (ensures reproducibility)

### CLI
| Arg | Default | Description |
|---|---|---|
| `--chunk-size N` | 300 | Chunk size in approximate tokens |
| `--overlap N` | 50 | Overlap in approximate tokens |
| `--files N` | 5 | Process first N files (alphabetical); mutually exclusive with `--all` |
| `--all` | — | Process all `_stripped.md` files |
| `--force` | — | Overwrite all existing output files; without this, existing files are skipped |

### Output folder
`tamu_data/processed/v3_result_{chunk_size}t_{overlap}o_{YYYYMMDD}/`
Created automatically on first run.

### Output JSON per file
Filename: `{stem_without_stripped}.json` (e.g., `202611_CSCE_670_600_46627_v010.json`)

```json
{
  "source_file": "202611_CSCE_670_600_46627_v010_stripped.md",
  "course_id": "CSCE_670",
  "semester": "202611",
  "chunk_size": 300,
  "overlap": 50,
  "total_chunks": 6,
  "chunks": [
    {
      "chunk_index": 0,
      "content": "...",
      "has_table": false,
      "token_count": 287
    }
  ]
}
```

`course_id` and `semester` are parsed directly from the filename stem (pattern: `{semester}_{dept}_{num}_{section}_{crn}_vNNN`).
`token_count` is approximate: `round(len(content) / 4)` — same estimator used by `chunker_v3.py`.
`has_table` is `True` if the chunk content contains `|...|` (Markdown table rows).

### Console output
Per file: `{stem}: {N} chunks, avg {X} tok (min {m}, max {M})`
Final summary: total files processed, total chunks, skipped count.

---

## Skill: `/chunk-syllabi`

**File:** `.claude/skills/chunk-syllabi.md`

### Trigger
User runs `/chunk-syllabi [args]` or asks to re-chunk syllabi with different settings.

### Arguments
| Arg | Default |
|---|---|
| `--chunk-size N` | 300 |
| `--overlap N` | 50 |
| `--files N` | 5 |
| `--all` | — |

### Flow
1. Parse args from invocation (use defaults if not specified)
2. Run in Docker: `docker exec tamubot-claude-1 python ingestion_pipeline/chunk_syllabi.py ...`
3. Print per-file summary from stdout
4. Inspect 2 output files: prefer one with `has_table=true` chunks and one without; show `chunk_index`, `token_count`, and full `content` for 2 chunks each
5. If test mode (`--files N`), offer to run `--all`

### Examples
```
/chunk-syllabi
/chunk-syllabi --chunk-size 500 --overlap 75
/chunk-syllabi --all --chunk-size 300 --overlap 50
```

---

## Implementation Steps

1. **Pre-requisite (manual):** migrate legacy folders and rename `v3_step3_flat/`
2. Write `ingestion_pipeline/chunk_syllabi.py`
3. Write `.claude/skills/chunk-syllabi.md`
4. Test run: `/chunk-syllabi` (5 files, 300/50)
5. Inspect output
