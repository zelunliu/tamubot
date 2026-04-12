"""One-off migration: convert JSONL golden sets → Excel, delete JSONL files.

Run from repo root:
    python evals/migrate_golden_sets.py

golden_20260411_v1.xlsx already exists (user-edited) — skip regenerating it,
just delete its JSONL counterpart.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.golden_set import save

GOLDEN_DIR = Path("tamu_data/evals/golden_sets")

# Files that already have a user-edited .xlsx — just delete the JSONL
SKIP_CONVERT = {"golden_20260411_v1", "golden_20260410_v2", "CSCE_25gradcourses_nonrecursive"}

# Throwaway subsets — delete only, no Excel needed
DELETE_ONLY = {"smoke_semantic_1q", "golden_20260411_v1_reworked"}


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def jsonl_to_golden_item(raw: dict, idx: int) -> dict:
    return {
        "id": raw.get("id") if raw.get("id") is not None else idx,
        "question": str(raw.get("question") or "").strip(),
        "reference_answer": str(raw.get("reference_answer") or "").strip(),
        "expected_function": str(raw.get("expected_function") or "").strip(),
        "human_notes": raw.get("human_notes") or raw.get("human_judgment"),
    }


def main():
    jsonl_files = list(GOLDEN_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found.")
        return

    for jf in sorted(jsonl_files):
        stem = jf.stem

        if stem in DELETE_ONLY:
            jf.unlink()
            print(f"  deleted (throwaway): {jf.name}")
            continue

        if stem in SKIP_CONVERT:
            jf.unlink()
            print(f"  deleted (xlsx exists): {jf.name}")
            continue

        xlsx_path = jf.with_suffix(".xlsx")
        if xlsx_path.exists():
            jf.unlink()
            print(f"  deleted (xlsx exists): {jf.name}")
            continue

        # Convert
        raw_items = load_jsonl(jf)
        items = [jsonl_to_golden_item(r, i + 1) for i, r in enumerate(raw_items)]
        save(items, xlsx_path)
        jf.unlink()
        print(f"  converted: {jf.name} → {xlsx_path.name}  ({len(items)} rows)")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
