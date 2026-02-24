"""
Production pipeline: Parse all CSCE + ISEN syllabus PDFs with Gemini 2.5 Flash.

Features:
  - Resumes from where it left off (skips already-processed files)
  - Logs all errors to tamu_data/ingestion_logs/errors.jsonl
  - Saves each result immediately (no batch-at-end risk)
  - Rate-limit aware with configurable delay between calls
  - Summary report at the end

Usage:
    GOOGLE_API_KEY=... python process_syllabi.py
    GOOGLE_API_KEY=... python process_syllabi.py --department CSCE   # only CSCE
    GOOGLE_API_KEY=... python process_syllabi.py --retry-errors      # retry previously failed files
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from google import genai

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL = "gemini-2.5-flash"

SYLLABI_DIR = Path("tamu_data/raw/syllabi")
OUTPUT_DIR = Path("tamu_data/processed/gemini_parsed")
LOG_DIR = Path("tamu_data/logs")

DEPARTMENTS = ["CSCE", "ISEN"]
MAX_RETRIES = 2
DELAY_BETWEEN_CALLS = 2  # seconds
DELAY_ON_RATE_LIMIT = 30  # seconds

PROMPT = """You are a university syllabus parser. Analyze this PDF and extract ALL content into structured JSON.

OUTPUT FORMAT — return ONLY valid JSON:
{
  "course_metadata": {
    "course_id": "DEPT XXX",
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
    "missing_sections": ["<category names that are NOT present in this syllabus>"],
    "warnings": ["<data quality issues, e.g. 'Grade weights not specified', 'Schedule has no dates'>"]
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
- Extract ALL course-specific content. Do not skip or summarize.
- Preserve tables as Markdown tables (| col1 | col2 | format). Set has_table=true.
- For UNIVERSITY_POLICIES: list ONLY the policy names in "boilerplate_policies". Do NOT include the full boilerplate text in chunks. Only include course-specific MODIFICATIONS to standard policies as chunks.
- Never use "MISC" or "OTHER" as a category. Use the closest match.
- Each chunk should be a coherent section. Don't split mid-paragraph.
- For completeness_check: list categories absent from the syllabus as missing_sections. Add warnings for missing grade weights, missing dates, missing contact info, etc.
- Escape all special characters properly. Output must be valid JSON.
"""


def get_pdf_list(departments: list[str]) -> list[Path]:
    """Get all PDFs for the specified departments."""
    pdfs = []
    for dept in departments:
        pdfs.extend(sorted(SYLLABI_DIR.glob(f"202611_{dept}_*.pdf")))
    return pdfs


def get_completed_files() -> set[str]:
    """Get set of filenames already successfully processed."""
    completed = set()
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.json"):
            # Verify it's a valid, non-error result
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if "error" not in data:
                    completed.add(f.stem + ".pdf")
            except (json.JSONDecodeError, IOError):
                pass
    return completed


def get_error_files() -> set[str]:
    """Get set of filenames that previously errored."""
    errors = set()
    error_log = LOG_DIR / "errors.jsonl"
    if error_log.exists():
        with open(error_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    errors.add(entry.get("file", ""))
                except json.JSONDecodeError:
                    pass
    return errors


def log_error(filename: str, error: str, attempt: int):
    """Append an error entry to the error log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "file": filename,
        "error": error,
        "attempt": attempt,
        "timestamp": datetime.now().isoformat(),
    }
    with open(LOG_DIR / "errors.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_progress(completed: int, total: int, filename: str, status: str):
    """Append a progress entry."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "file": filename,
        "status": status,
        "progress": f"{completed}/{total}",
        "timestamp": datetime.now().isoformat(),
    }
    with open(LOG_DIR / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def parse_pdf(client, pdf_path: Path) -> dict:
    """Send a PDF to Gemini and get structured JSON. Retries on failure."""
    pdf_bytes = pdf_path.read_bytes()

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    genai.types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    PROMPT,
                ],
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=65536,
                    response_mime_type="application/json",
                ),
            )

            raw = response.text.strip()
            parsed = json.loads(raw)

            # Inject source filename into metadata for traceability
            parsed["_source_file"] = pdf_path.name
            parsed["_parsed_at"] = datetime.now().isoformat()

            return parsed

        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {e}"
            log_error(pdf_path.name, error_msg, attempt)
            if attempt < MAX_RETRIES:
                time.sleep(DELAY_BETWEEN_CALLS)
            else:
                return {"error": error_msg, "_source_file": pdf_path.name}

        except Exception as e:
            error_str = str(e)
            log_error(pdf_path.name, error_str, attempt)

            # Rate limit detection
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                print(f"    Rate limited. Waiting {DELAY_ON_RATE_LIMIT}s...")
                time.sleep(DELAY_ON_RATE_LIMIT)
            elif attempt < MAX_RETRIES:
                time.sleep(DELAY_BETWEEN_CALLS * (attempt + 1))
            else:
                return {"error": error_str, "_source_file": pdf_path.name}

    return {"error": "Exhausted retries", "_source_file": pdf_path.name}


def main():
    parser = argparse.ArgumentParser(description="Parse syllabus PDFs with Gemini 2.5 Flash")
    parser.add_argument("--department", type=str, help="Process only this department (e.g., CSCE)")
    parser.add_argument("--retry-errors", action="store_true", help="Retry previously failed files")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        sys.exit(1)

    departments = [args.department.upper()] if args.department else DEPARTMENTS
    all_pdfs = get_pdf_list(departments)
    completed = get_completed_files()

    if args.retry_errors:
        error_files = get_error_files()
        to_process = [p for p in all_pdfs if p.name in error_files]
        print(f"Retrying {len(to_process)} previously failed files")
    else:
        to_process = [p for p in all_pdfs if p.name not in completed]

    total = len(all_pdfs)
    already_done = len(completed)
    remaining = len(to_process)

    print(f"{'='*60}")
    print(f"Syllabus Processing Pipeline")
    print(f"{'='*60}")
    print(f"  Departments:  {', '.join(departments)}")
    print(f"  Total PDFs:   {total}")
    print(f"  Already done: {already_done}")
    print(f"  To process:   {remaining}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print(f"  Error log:    {LOG_DIR / 'errors.jsonl'}")
    print(f"{'='*60}\n")

    if remaining == 0:
        print("Nothing to process. All files already completed.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=API_KEY)

    ok_count = 0
    fail_count = 0
    start_time = time.time()

    for i, pdf_path in enumerate(to_process):
        n = already_done + i + 1
        print(f"[{n}/{total}] {pdf_path.name} ({pdf_path.stat().st_size/1024:.0f} KB)...", end=" ", flush=True)

        result = parse_pdf(client, pdf_path)

        # Save immediately
        out_path = OUTPUT_DIR / pdf_path.name.replace(".pdf", ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if "error" in result:
            fail_count += 1
            print(f"FAIL: {result['error'][:60]}")
            log_progress(n, total, pdf_path.name, "error")
        else:
            ok_count += 1
            chunks = len(result.get("chunks", []))
            missing = result.get("completeness_check", {}).get("missing_sections", [])
            warnings = result.get("completeness_check", {}).get("warnings", [])
            status_parts = [f"{chunks} chunks"]
            if missing:
                status_parts.append(f"missing: {','.join(missing)}")
            if warnings:
                status_parts.append(f"{len(warnings)} warnings")
            print(f"OK ({', '.join(status_parts)})")
            log_progress(n, total, pdf_path.name, "ok")

        time.sleep(DELAY_BETWEEN_CALLS)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"  Succeeded: {ok_count}")
    print(f"  Failed:    {fail_count}")
    print(f"  Total processed this run: {ok_count + fail_count}")
    print(f"  Overall progress: {already_done + ok_count}/{total}")
    if fail_count > 0:
        print(f"  Run with --retry-errors to retry failed files")
    print(f"{'='*60}")

    # Write summary
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "departments": departments,
        "total_pdfs": total,
        "previously_completed": already_done,
        "processed_this_run": ok_count + fail_count,
        "succeeded": ok_count,
        "failed": fail_count,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(LOG_DIR / "last_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
