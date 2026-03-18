---
name: chunk-syllabi
description: Use when re-chunking v3 syllabus stripped markdown files with configurable chunk size and overlap, running experiments without LLM calls, or inspecting token-level chunk output.
---

# /chunk-syllabi — Re-chunk v3 syllabi at custom token sizes

Pure token chunking from `v3_step2_boilerplate/*_stripped.md`. No LLM, no API cost.
Output goes to `tamu_data/processed/v3_result_<size>t_<overlap>o_<date>/`.

---

## Step 1 — Parse arguments

| Arg | Default | Description |
|---|---|---|
| `--chunk-size N` | 300 | Chunk size in approximate tokens |
| `--overlap N` | 50 | Overlap in approximate tokens |
| `--files N` | 5 | First N files (alphabetical), test mode |
| `--all` | — | Process all files |
| `--force` | — | Overwrite existing output |

Examples:
```
/chunk-syllabi                                     → 300/50, 5 files
/chunk-syllabi --chunk-size 500 --overlap 75       → 500/75, 5 files
/chunk-syllabi --all --chunk-size 300 --overlap 50 → 300/50, all files
```

---

## Step 2 — Run the script

```bash
docker exec tamubot-claude-1 python ingestion_pipeline/chunk_syllabi.py \
  --chunk-size <size> --overlap <overlap> [--files N | --all] [--force]
```

Run in background. Stream stdout for live per-file progress.

---

## Step 3 — Inspect output

After the run completes, inspect 2 output files:
- Prefer one file with `has_table: true` chunks and one without.
- Show `chunk_index`, `token_count`, and full `content` for 2 chunks from each file.

```bash
docker exec tamubot-claude-1 python - <<'EOF'
import json, random
from pathlib import Path

out_dir = sorted(Path("tamu_data/processed").glob("v3_result_<size>t_<overlap>o_*"))[-1]
files = sorted(out_dir.glob("*.json"))

with_table = [f for f in files if any(c["has_table"] for c in json.loads(f.read_text())["chunks"])]
without_table = [f for f in files if f not in with_table]

picks = []
if with_table: picks.append(random.choice(with_table))
if without_table: picks.append(random.choice(without_table))
if len(picks) < 2: picks.append(random.choice(files))

for fp in picks[:2]:
    d = json.loads(fp.read_text())
    print(f"\n=== {fp.stem} | {d['total_chunks']} chunks | {d['course_id']} ===")
    for c in d["chunks"][:2]:
        print(f"\n[chunk {c['chunk_index']} | {c['token_count']} tok | table={c['has_table']}]")
        print(c["content"])
EOF
```

---

## Step 4 — Offer full run

If test mode (`--files N`), offer:
> "Test run complete. Run `--all` to process all files?"

---

## Notes

- Token counts are approximate: `round(len(content) / 4)`
- `has_table` detects Markdown pipe syntax (`|...|`)
- Output folder is **experiment only** — not for direct Atlas ingestion
- `v3_result_600t_100o_20260311/` = original production chunks (renamed from `v3_step3_flat/`)
