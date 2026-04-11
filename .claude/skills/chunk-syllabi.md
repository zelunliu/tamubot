---
name: chunk-syllabi
description: Use when re-chunking v3 syllabus stripped markdown files at custom token sizes, running chunking experiments, or inspecting token-level chunk output
---

# /chunk-syllabi — Re-chunk v3 syllabi

Pure token chunking from `v3_step2_boilerplate/*_stripped.md`. No LLM, no API cost.
Output: `tamu_data/processed/v3_result_<size>t_<overlap>o_<date>/`

## Args

| Arg | Default | Description |
|---|---|---|
| `--chunk-size N` | 300 | Chunk size in approximate tokens |
| `--overlap N` | 50 | Overlap in approximate tokens |
| `--files N` | 5 | First N files (test mode) |
| `--all` | — | Process all files |
| `--force` | — | Overwrite existing output |

## Run

```bash
docker exec tamubot-claude-1 python ingestion_pipeline/chunk_syllabi.py \
  --chunk-size <size> --overlap <overlap> [--files N | --all] [--force]
```

Run in background, stream stdout.

## Inspect output

After run, show 2 chunks from 2 files (prefer one with `has_table: true`):

```python
import json, random
from pathlib import Path
out_dir = sorted(Path("tamu_data/processed").glob("v3_result_<size>t_<overlap>o_*"))[-1]
files = sorted(out_dir.glob("*.json"))
for fp in random.sample(files, min(2, len(files))):
    d = json.loads(fp.read_text())
    print(f"\n=== {fp.stem} | {d['total_chunks']} chunks ===")
    for c in d["chunks"][:2]:
        print(f"[chunk {c['chunk_index']} | {c['token_count']} tok | table={c['has_table']}]")
        print(c["content"])
```

After test run, offer: "Run `--all` to process all files?"

## Notes

- Token count: `round(len(content) / 4)` (approximate)
- Production baseline: `v3_result_600t_100o_*` (chunk_size=600, overlap=100)
