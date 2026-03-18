# Design: chunk-syllabi script + skill

**Date:** 2026-03-18
**Status:** Approved

---

## Overview

A standalone chunking script and companion Claude skill for re-chunking v3 syllabus pipeline output at configurable token sizes. Operates purely on already-processed `_stripped.md` files — no LLM calls, no API cost.

---

## Background

The v3 pipeline currently produces `v3_step3_flat/` with `chunk_size=600, overlap=100` via an LLM-enriched step 3. We need a lightweight way to experiment with different chunk sizes (e.g., 300/50) without re-running the expensive LLM enrichment. The existing `chunker_v3.py` already implements the core algorithm; this work adds a scriptable entry point and skill wrapper.

---

## Data Layout

### Folder naming convention
All chunked output folders use the pattern:
```
tamu_data/processed/v3_result_<size>t_<overlap>o_<YYYYMMDD>/
```

Examples:
- `v3_result_600t_100o_20260310/` — existing production chunks (renamed from `v3_step3_flat/`)
- `v3_result_300t_50o_20260318/` — new 300-token experiment

### One-time migration (manual, not in skill)
- `gem_parsed_20260304/`, `gem_parsed_20260306/`, `gemini_parsed/`, `v2_chunked/`, `v2_filtered/` → `tamu_data/processed/legacy/`
- `v3_step3_flat/` → `v3_result_600t_100o_<mtime_date>/`

---

## Script: `ingestion_pipeline/chunk_syllabi.py`

### Input
- Source: `tamu_data/processed/v3_step2_boilerplate/*_stripped.md`
- CLI args:
  - `--chunk-size INT` (default: 300)
  - `--overlap INT` (default: 50)
  - `--files N` — process first N files alphabetically (default: 5, test mode)
  - `--all` — process all files

### Output folder
`tamu_data/processed/v3_result_{chunk_size}t_{overlap}o_{YYYYMMDD}/`

Created automatically. Skips existing files unless `--force` is passed.

### Output JSON per file
Filename: `{stem}.json` (stem without `_stripped` suffix)

```json
{
  "source_file": "202611_CSCE_670_600_46627_v010_stripped.md",
  "chunk_size": 300,
  "overlap": 50,
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

### Console output
Per file: `{stem}: {N} chunks, avg {X} tokens (min {m}, max {M})`
Final summary: total files processed, total chunks produced.

---

## Skill: `/chunk-syllabi`

**File:** `.claude/skills/chunk-syllabi.md`

### Trigger
User runs `/chunk-syllabi [args]` or asks to re-chunk syllabi with different settings.

### Arguments
| Arg | Default | Description |
|---|---|---|
| `--chunk-size N` | 300 | Chunk size in tokens |
| `--overlap N` | 50 | Overlap in tokens |
| `--files N` | 5 | Number of test files |
| `--all` | — | Process all 336 files |

### Flow
1. Parse args from invocation (use defaults if not specified)
2. Run script in Docker sandbox: `docker exec tamubot-claude-1 python ingestion_pipeline/chunk_syllabi.py ...`
3. Print per-file summary from stdout
4. Inspect 2 random chunks from 2 different output files — show `chunk_index`, `token_count`, and full `content` inline
5. Offer to run `--all` if in test mode

### Examples
```
/chunk-syllabi
/chunk-syllabi --chunk-size 500 --overlap 75
/chunk-syllabi --all --chunk-size 300 --overlap 50
```

---

## Implementation Checklist

1. One-time manual migration: legacy folders + rename `v3_step3_flat/`
2. Write `ingestion_pipeline/chunk_syllabi.py`
3. Write `.claude/skills/chunk-syllabi.md`
4. Test run: `/chunk-syllabi` (5 files, 300/50)
5. Inspect output
