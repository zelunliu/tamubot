"""
Test TAMU API PDF parsing on CSCE + ISEN syllabus PDFs.
Extracts PDF text with PyMuPDF, then parses via TAMU gateway.

Usage: python ingestion_pipeline/test_gemini_parse.py
"""

import json
import sys
import time
from pathlib import Path
import fitz  # PyMuPDF
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

MODEL = config.TAMU_MODEL

SYLLABI_DIR = Path("tamu_data/raw/syllabi")
OUTPUT_DIR = Path("tamu_data/processed/test_output")

TEST_FILES = [
    "202611_CSCE_481_599_42607.pdf",   # seminar, rubric tables (failed round 1 — retry)
    "202611_CSCE_482_932_37076.pdf",   # capstone, rubrics
    "202611_ISEN_210_500_53793.pdf",   # ISEN intro
    "202611_ISEN_302_500_53801.pdf",   # ISEN mid-level
]

EXPECTED_CATEGORIES = [
    "COURSE_OVERVIEW",
    "PREREQUISITES",
    "LEARNING_OUTCOMES",
    "MATERIALS",
    "GRADING",
    "SCHEDULE",
    "ATTENDANCE_AND_MAKEUP",
    "AI_POLICY",
]

PROMPT = """You are a university syllabus parser. Analyze this PDF and extract ALL content into structured JSON.

OUTPUT FORMAT — return ONLY valid JSON (no markdown fences, no commentary):
{
  "course_metadata": {
    "course_id": "CSCE XXX",
    "section": "XXX",
    "term": "Spring 2026",
    "crn": "XXXXX",
    "instructor": {
      "name": "...",
      "email": "...",
      "office": "...",
      "office_hours": "..."
    },
    "teaching_assistants": [{"name": "...", "email": "..."}],
    "meeting_times": "...",
    "location": "...",
    "credit_hours": "..."
  },
  "chunks": [
    {
      "category": "<one of the 11 categories below>",
      "title": "<section heading from the document>",
      "content": "<full text of this section, preserving all detail>",
      "has_table": true/false
    }
  ],
  "boilerplate_policies": [
    "<list NAMES of standard TAMU university policies found, e.g. 'ADA Policy', 'FERPA'>"
  ],
  "completeness_check": {
    "missing_sections": ["<category names from the 11 below that are NOT present in this syllabus>"],
    "warnings": ["<any data quality issues, e.g. 'Grade weights not specified', 'Schedule has no dates', 'Instructor email missing'>"]
  }
}

THE 11 SEMANTIC CATEGORIES (use exactly these string values for "category"):
1. COURSE_OVERVIEW — course description, catalog info, special designations, format
2. INSTRUCTOR — instructor/TA details (only if NOT already fully captured in course_metadata)
3. PREREQUISITES — required courses, corequisites, standing requirements
4. LEARNING_OUTCOMES — what students will learn, course objectives
5. MATERIALS — textbooks, required software, platforms, tech requirements
6. GRADING — grade scale, weights, component descriptions (homework, exams, labs, projects), rubrics, grade appeals
7. SCHEDULE — course calendar, weekly topics, exam dates, assignment due dates
8. ATTENDANCE_AND_MAKEUP — attendance rules, late work policy, makeup exams, excused absences
9. AI_POLICY — AI tool usage rules (permitted, required, prohibited), citation requirements
10. UNIVERSITY_POLICIES — standard institutional boilerplate (ADA, FERPA, Title IX, Honor Code, etc.)
11. SUPPORT_SERVICES — IT help, Canvas support, tutoring, writing center

RULES:
- Extract ALL course-specific content from the PDF. Do not skip or summarize anything.
- Preserve tables as Markdown tables (| col1 | col2 | format). Set has_table=true for those chunks.
- For UNIVERSITY_POLICIES: list ONLY the policy names in "boilerplate_policies". Do NOT include the full boilerplate text in any chunk. If the syllabus has course-specific modifications to a standard policy, include ONLY the modification in a chunk.
- If a section doesn't fit any category, use the closest match. Never use "MISC" or "OTHER".
- Each chunk should be a coherent section. Don't split mid-paragraph.
- For completeness_check: compare what you found against all 11 categories. List any that are completely absent from the syllabus as missing_sections. Add specific warnings for common issues like missing grade weights, missing exam dates, missing instructor contact info, etc.
- Escape all special characters in JSON strings properly. Double-check your JSON is valid before responding.
"""


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def parse_pdf(client, pdf_path: Path, max_retries: int = 2) -> dict:
    """Extract PDF text and parse via TAMU API. Retries on JSON errors."""
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name} ({pdf_path.stat().st_size/1024:.0f} KB)")
    print(f"{'='*60}")

    pdf_text = extract_pdf_text(pdf_path)
    user_message = f"{PROMPT}\n\n---\n\nSYLLABUS TEXT:\n{pdf_text}"

    for attempt in range(max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.1,
                max_tokens=65536,
                response_format={"type": "json_object"},
                stream=True,
            )
            raw = "".join(chunk.choices[0].delta.content or "" for chunk in stream).strip()
            parsed = json.loads(raw)
            if attempt > 0:
                print(f"  Succeeded on retry {attempt}")
            return parsed

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"  JSON error (attempt {attempt+1}), retrying...")
                time.sleep(2)
            else:
                print(f"  JSON parse error after {max_retries+1} attempts: {e}")
                print(f"  Raw output (first 500 chars): {raw[:500]}")
                return {"error": str(e), "raw": raw[:2000]}

        except Exception as e:
            print(f"  API error: {e}")
            if attempt < max_retries:
                time.sleep(5)
            else:
                return {"error": str(e)}


def print_summary(result: dict, pdf_name: str):
    """Print a readable summary of the parsing result."""
    if "error" in result:
        print(f"  FAILED: {result['error']}")
        return

    meta = result.get("course_metadata", {})
    chunks = result.get("chunks", [])
    policies = result.get("boilerplate_policies", [])
    completeness = result.get("completeness_check", {})

    print(f"\n  Course: {meta.get('course_id', '?')} Section {meta.get('section', '?')}")
    print(f"  Instructor: {meta.get('instructor', {}).get('name', '?')}")
    print(f"  Meeting: {meta.get('meeting_times', '?')} @ {meta.get('location', '?')}")
    print(f"  Chunks: {len(chunks)}")

    print(f"\n  Category breakdown:")
    cats = {}
    for c in chunks:
        cat = c.get("category", "?")
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        has_table = sum(1 for c in chunks if c.get("category") == cat and c.get("has_table"))
        table_str = f" ({has_table} with tables)" if has_table else ""
        print(f"    {cat:<25} {count}{table_str}")

    # Completeness check
    missing = completeness.get("missing_sections", [])
    warnings = completeness.get("warnings", [])
    if missing:
        print(f"\n  MISSING SECTIONS: {', '.join(missing)}")
    if warnings:
        print(f"  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")

    # Boilerplate
    if policies:
        print(f"\n  Boilerplate ({len(policies)}): {', '.join(policies[:4])}...")

    # Show a sample table chunk if present
    table_chunks = [c for c in chunks if c.get("has_table")]
    if table_chunks:
        sample = table_chunks[0]
        print(f"\n  Sample table ({sample['category']} — {sample.get('title', '?')}):")
        for line in sample["content"][:300].split("\n"):
            print(f"    {line}")
        if len(sample["content"]) > 300:
            print(f"    ... ({len(sample['content'])} chars total)")


def main():
    if not config.TAMU_API_KEY:
        print("ERROR: Set TAMU_API_KEY environment variable")
        sys.exit(1)

    client = config.get_tamu_client()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_summary = []

    for filename in TEST_FILES:
        pdf_path = SYLLABI_DIR / filename
        if not pdf_path.exists():
            print(f"SKIP: {pdf_path} not found")
            continue

        result = parse_pdf(client, pdf_path)
        print_summary(result, filename)

        # Save full JSON output
        out_path = OUTPUT_DIR / filename.replace(".pdf", ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

        # Track for final report
        if "error" not in result:
            cc = result.get("completeness_check", {})
            results_summary.append({
                "file": filename,
                "course": result.get("course_metadata", {}).get("course_id", "?"),
                "chunks": len(result.get("chunks", [])),
                "missing": cc.get("missing_sections", []),
                "warnings": cc.get("warnings", []),
                "boilerplate": len(result.get("boilerplate_policies", [])),
            })
        else:
            results_summary.append({"file": filename, "error": result["error"]})

        # Brief pause between API calls
        time.sleep(1)

    # Final report
    print(f"\n\n{'='*60}")
    print(f"SUMMARY — {len(results_summary)} files processed")
    print(f"{'='*60}")
    for r in results_summary:
        if "error" in r:
            print(f"  FAIL  {r['file']}: {r['error'][:80]}")
        else:
            missing_str = f" | MISSING: {', '.join(r['missing'])}" if r['missing'] else ""
            warn_str = f" | {len(r['warnings'])} warnings" if r['warnings'] else ""
            print(f"  OK    {r['course']:<12} {r['chunks']:>2} chunks, {r['boilerplate']} policies{missing_str}{warn_str}")


if __name__ == "__main__":
    main()
