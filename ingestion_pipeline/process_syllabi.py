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

import csv
import json
import os
import re
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
OUTPUT_DIR = Path(f"tamu_data/processed/gem_parsed_{datetime.now().strftime('%Y%m%d')}")
LOG_DIR = Path("tamu_data/logs")
REPORT_DIR = Path("tamu_data/logs/per_file")
PROGRESS_CSV = OUTPUT_DIR / "parsing_progress.csv"
PROGRESS_JSONL = OUTPUT_DIR / "parsing_progress.jsonl"

DEPARTMENTS = ["CSCE", "ISEN"]
MAX_RETRIES = 2
DELAY_BETWEEN_CALLS = 2  # seconds
DELAY_ON_RATE_LIMIT = 30  # seconds

# Categories in CSV column order (COURSE_SUMMARY adjacent to OVERVIEW)
ALL_CATEGORIES = [
    "COURSE_OVERVIEW", "COURSE_SUMMARY", "INSTRUCTOR", "PREREQUISITES",
    "LEARNING_OUTCOMES", "MATERIALS", "GRADING", "SCHEDULE",
    "ATTENDANCE_AND_MAKEUP", "AI_POLICY", "UNIVERSITY_POLICIES", "SUPPORT_SERVICES",
    "SAFETY",
]

# Token-count warning thresholds per category (min, max)
CSV_FIELDS = [
    "file", "course_id", "section", "crn", "status",
    "error_type", "error_detail", "chunk_count",
    *[f"{cat}_tok" for cat in ALL_CATEGORIES],
    "course_url", "flags", "parsed_at",
]

TOKEN_THRESHOLDS: dict[str, tuple[int, int]] = {
    "COURSE_SUMMARY":        (200, 500),
    "GRADING":               (50, 5000),
    "SCHEDULE":              (50, 8000),
    "default":               (20, 3000),
}

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
    "credit_hours": "...",
    "course_url": "<Canvas course URL or official course webpage if found, else null>"
  },
  "chunks": [
    {
      "category": "<one of the 12 categories below>",
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

THE 13 SEMANTIC CATEGORIES (use exactly these string values for "category"):
1.  COURSE_OVERVIEW — course description, catalog info, special designations, format
2.  INSTRUCTOR — instructor/TA details (only if NOT already fully captured in course_metadata)
3.  PREREQUISITES — required courses, corequisites, standing requirements
4.  LEARNING_OUTCOMES — what students will learn, course objectives
5.  MATERIALS — textbooks, required software, platforms, tech requirements
6.  GRADING — grade scale, weights, component descriptions (homework, exams, labs, projects), rubrics, grade appeals
7.  SCHEDULE — course calendar, weekly topics, exam dates, assignment due dates
8.  ATTENDANCE_AND_MAKEUP — attendance rules, late work policy, makeup exams, excused absences
9.  AI_POLICY — AI tool usage rules (permitted, required, prohibited), citation requirements
10. UNIVERSITY_POLICIES — standard institutional boilerplate (ADA, FERPA, Title IX, Honor Code, etc.)
11. SUPPORT_SERVICES — IT help, Canvas support, tutoring, writing center
12. COURSE_SUMMARY — RAG-optimized keyword index (see generation rules below)
13. SAFETY — lab safety rules, protective equipment requirements, dress code, chemical/biological
    hazard handling, equipment operation rules, emergency procedures, food/drink restrictions.
    Use only for courses with a physical lab or hands-on component. Do NOT use for generic
    academic integrity or classroom conduct (those go in UNIVERSITY_POLICIES).

RULES:
- Extract ALL course-specific content. Do not skip or summarize (except COURSE_SUMMARY).
- Preserve tables as Markdown tables (| col1 | col2 | format). Set has_table=true.
- For UNIVERSITY_POLICIES: list ONLY the policy names in "boilerplate_policies". Do NOT include the full boilerplate text in chunks. Only include course-specific MODIFICATIONS to standard policies as chunks.
- Never use "MISC" or "OTHER" as a category. Use the closest match.
- Each chunk should be a coherent section. Don't split mid-paragraph.
- For completeness_check: list categories absent from the syllabus as missing_sections.
  Do NOT include COURSE_SUMMARY in missing_sections — it is always generated.
  Only include SAFETY in missing_sections if the course clearly has a lab/hands-on component.
  Add warnings for missing grade weights, missing dates, missing contact info, etc.
- Escape all special characters properly. Output must be valid JSON.
- When assigning a category, verify the CONTENT of the section (not just the header/title).
  If the header says "Assignments" but the content is about grade percentages and weights,
  assign GRADING. If a section header is ambiguous, use the content to determine the best
  matching category from the 13 options. Always prefer content semantics over header words.
- If you are unsure between two categories, choose the one that better describes what a
  student would need to know from this section's content.
- course_url: extract the Canvas course URL or official course webpage URL if present in the
  syllabus. Set to null if not found.

COURSE_SUMMARY GENERATION RULES (category 12):
Generate exactly one COURSE_SUMMARY chunk using this strict format:

  <DEPT> <NUM> [/ <crosslist>] | <Full Course Title> | <Term>
  Instructor: <Full Name> | <email>
  Meets: <days and times> | <location>
  Topics: <ALL named concepts, techniques, algorithms, models, and skills from the ENTIRE syllabus — course description, learning outcomes, AND weekly schedule — merged into one comma-separated list. Rare/specialized terms first, broader ones last. No duplicates.>
  Tools: <software, instruments, or platforms students are required or encouraged to USE for coursework (assignments, labs, projects); omit if none. Do NOT include tools merely permitted or mentioned in AI/academic integrity policies.>
  Prerequisites: <exact course codes and standing; omit if none stated>

Rules for COURSE_SUMMARY content:
- GROUNDING: Use ONLY terms and phrases that appear explicitly in the source text. Do NOT add domain knowledge or infer anything not written. If "Python" is not in the text, it is not in the summary.
- TOPICS IS EVERYTHING: Topics is the single comprehensive concept list — merge what would otherwise be topics, methods, skills, and niche into Topics. Do not create separate fields for these.
- NO DUPLICATION: No term appears twice anywhere in the summary. Tools must not repeat terms already in Topics.
- RARE TERMS FIRST: List specialized/rare terms before generic ones in Topics. Rare terms provide stronger keyword signal for search.
- BE THOROUGH: Extract ALL named algorithms, models, techniques, tools, and skills from the entire syllabus — do not truncate.
- Retain ALL proper nouns, theorem names, algorithm names, drug names, and tool names verbatim.
- NEVER write narrative: "students will learn", "designed to", "upon completion", "this course", "you will", "gain experience", or any similar framing.
- Use declarative noun phrases only.
- If a field has no data in the source text, omit that line entirely.
- Target 300–450 tokens total.
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


def sanitize_json(raw: str) -> str:
    """Attempt to clean common Gemini JSON output errors."""
    raw = raw.replace('\x00', '')  # null bytes
    # Fix invalid backslash escapes (not one of: " \ / b f n r t u 0-9)
    raw = re.sub(r'\\([^"\\/bfnrtu0-9])', r'\\\\\1', raw)
    # Strip bare control chars (0x01-0x1F except \t \n \r)
    raw = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return raw


def clean_replacement_chars(obj):
    """Recursively replace Unicode replacement character (U+FFFD) with a hyphen.

    Gemini substitutes \ufffd when it encounters bytes it can't decode from the
    PDF (e.g. Windows-1252 en-dashes, ligatures). Replacing with '-' is safe
    for office hours ranges, schedules, etc.
    """
    if isinstance(obj, str):
        return obj.replace('\ufffd', '-')
    if isinstance(obj, dict):
        return {k: clean_replacement_chars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_replacement_chars(v) for v in obj]
    return obj


def collapse_chunks_by_category(chunks: list[dict]) -> list[dict]:
    """Merge all chunks sharing the same category into a single chunk."""
    from collections import OrderedDict
    grouped: OrderedDict = OrderedDict()
    for chunk in chunks:
        cat = chunk.get("category", "COURSE_OVERVIEW")
        if cat not in grouped:
            grouped[cat] = {
                "category": cat,
                "title": chunk.get("title", ""),
                "content": chunk.get("content", ""),
                "has_table": chunk.get("has_table", False),
            }
        else:
            grouped[cat]["content"] += "\n\n" + chunk.get("content", "")
            if chunk.get("has_table"):
                grouped[cat]["has_table"] = True
            existing_title = grouped[cat]["title"]
            new_title = chunk.get("title", "")
            if new_title and new_title not in existing_title:
                grouped[cat]["title"] = existing_title + " / " + new_title
    return list(grouped.values())


def dedup_course_summary(content: str) -> str:
    """Remove duplicate terms in COURSE_SUMMARY after generation.

    - Deduplicates within Topics itself (removes repeated terms).
    - Strips any Tools terms already present in Topics.
    - Drops any other legacy fields (Skills, Methods, Niche, Schedule) if the model emitted them.
    """
    def parse_terms(s: str) -> list[str]:
        return [t.strip() for t in s.split(",") if t.strip()]

    def norm(t: str) -> str:
        return t.lower().strip()

    lines = content.split("\n")
    result_lines = []
    topics_seen: set[str] = set()

    for line in lines:
        # Deduplicate within Topics
        m = re.match(r"^(\s*)(Topics):\s*(.*)", line)
        if m:
            indent, _, terms_str = m.groups()
            unique = []
            for t in parse_terms(terms_str):
                if norm(t) not in topics_seen:
                    topics_seen.add(norm(t))
                    unique.append(t)
            result_lines.append(f"{indent}Topics: {', '.join(unique)}")
            continue

        # Strip Tools terms already in Topics
        m = re.match(r"^(\s*)(Tools):\s*(.*)", line)
        if m:
            indent, _, terms_str = m.groups()
            unique = [t for t in parse_terms(terms_str) if norm(t) not in topics_seen]
            if unique:
                result_lines.append(f"{indent}Tools: {', '.join(unique)}")
            continue

        # Drop legacy fields the model might still emit
        if re.match(r"^\s*(Skills|Methods|Tools/Platforms|Niche|Schedule):", line):
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


def write_per_file_report(pdf_path: Path, result: dict):
    """Write a human-readable .txt report for a processed PDF."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / (pdf_path.stem + ".txt")

    lines = [f"File: {pdf_path.name}"]

    if "error" in result:
        lines.append("Status: FAILED")
        lines.append(f"Error: {result['error']}")
        lines.append(f"Attempts: {result.get('_attempts', MAX_RETRIES + 1)}")
    else:
        chunks = result.get("chunks", [])
        parsed_at = result.get("_parsed_at", "")
        lines.append(f"Status: OK  |  Chunks: {len(chunks)}  |  Parsed: {parsed_at}")
        lines.append("")
        lines.append("Chunks:")
        for chunk in chunks:
            cat = chunk.get("category", "")
            title = chunk.get("title", "")
            lines.append(f"  {cat:<22} — \"{title}\"")
        completeness = result.get("completeness_check", {})
        missing = completeness.get("missing_sections", [])
        warnings = completeness.get("warnings", [])
        if missing:
            lines.append("")
            lines.append(f"Missing sections: {', '.join(missing)}")
        if warnings:
            lines.append("Completeness warnings:")
            for w in warnings:
                lines.append(f"  - {w}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def count_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars for English text)."""
    return max(0, round(len(text) / 4))


def classify_error(error_str: str) -> str:
    """Map a raw error string to a standardized error type."""
    s = error_str.lower()
    if "json parse error" in s:
        return "JSON_PARSE_ERROR"
    if "ssl" in s or "certificate" in s:
        return "SSL_ERROR"
    if "getaddrinfo" in s or "name or service not known" in s or "nodename nor servname" in s:
        return "DNS_ERROR"
    if "429" in error_str or "quota" in s or "rate" in s:
        return "RATE_LIMIT"
    if "exhausted retries" in s:
        return "EXHAUSTED_RETRIES"
    return "UNKNOWN_ERROR"


def build_progress_row(pdf_path: Path, result: dict) -> dict:
    """Build a CSV row dict for one processed PDF."""
    meta = result.get("course_metadata", {})
    chunks_by_cat = {c["category"]: c for c in result.get("chunks", [])}

    token_cols: dict[str, int] = {}
    flags: list[str] = []
    for cat in ALL_CATEGORIES:
        content = chunks_by_cat.get(cat, {}).get("content", "")
        tok = count_tokens(content)
        token_cols[f"{cat}_tok"] = tok
        if content:  # only flag present chunks
            lo, hi = TOKEN_THRESHOLDS.get(cat, TOKEN_THRESHOLDS["default"])
            if tok < lo:
                flags.append(f"{cat}:TOO_SMALL({tok})")
            elif tok > hi:
                flags.append(f"{cat}:TOO_LARGE({tok})")

    if "error" in result:
        error_type = classify_error(result["error"])
        error_detail = result["error"][:120]
    else:
        error_type = ""
        error_detail = ""

    return {
        "file": pdf_path.name,
        "course_id": meta.get("course_id", ""),
        "section": meta.get("section", ""),
        "crn": meta.get("crn", ""),
        "status": "FAILED" if "error" in result else "OK",
        "error_type": error_type,
        "error_detail": error_detail,
        "chunk_count": len(result.get("chunks", [])),
        **token_cols,
        "course_url": meta.get("course_url") or "",
        "flags": "; ".join(flags),
        "parsed_at": result.get("_parsed_at", datetime.now().isoformat()),
    }


def load_progress_csv() -> list[dict]:
    """Load existing progress CSV rows, keyed by filename for deduplication."""
    if not PROGRESS_CSV.exists():
        return []
    try:
        with open(PROGRESS_CSV, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_progress_csv(rows: list[dict]):
    """Rewrite the full progress CSV (called after every PDF)."""
    if not rows:
        return
    PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def append_progress_jsonl(row: dict):
    """Append one row to the live-tail-able JSONL sidecar (append-only, no lock conflicts)."""
    PROGRESS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


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
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = json.loads(sanitize_json(raw))

            # Clean up Unicode replacement chars (e.g. en-dashes in office hours)
            parsed = clean_replacement_chars(parsed)

            # Collapse duplicate-category chunks → at most 13 chunks per file
            parsed["chunks"] = collapse_chunks_by_category(parsed.get("chunks", []))

            # Strip has_table when false — only keep the field when a table is present
            for chunk in parsed["chunks"]:
                if not chunk.get("has_table"):
                    chunk.pop("has_table", None)

            # Deterministically remove cross-field duplicates from COURSE_SUMMARY
            for chunk in parsed["chunks"]:
                if chunk.get("category") == "COURSE_SUMMARY":
                    chunk["content"] = dedup_course_summary(chunk["content"])

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
                return {"error": error_msg, "_source_file": pdf_path.name, "_attempts": attempt + 1}

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
                return {"error": error_str, "_source_file": pdf_path.name, "_attempts": attempt + 1}

    return {"error": "Exhausted retries", "_source_file": pdf_path.name, "_attempts": MAX_RETRIES + 1}


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
    print(f"  Progress CSV: {PROGRESS_CSV}")
    print(f"  Reports dir:  {REPORT_DIR}")
    print(f"  Error log:    {LOG_DIR / 'errors.jsonl'}")
    print(f"{'='*60}\n")

    if remaining == 0:
        print("Nothing to process. All files already completed.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=API_KEY)

    # Load existing progress rows; index by filename for upsert behaviour
    progress_rows = load_progress_csv()
    progress_index = {r["file"]: i for i, r in enumerate(progress_rows)}

    ok_count = 0
    fail_count = 0
    start_time = time.time()

    for i, pdf_path in enumerate(to_process):
        n = already_done + i + 1
        print(f"[{n}/{total}] {pdf_path.name} ({pdf_path.stat().st_size/1024:.0f} KB)...", end=" ", flush=True)

        result = parse_pdf(client, pdf_path)
        write_per_file_report(pdf_path, result)

        # Update realtime progress CSV
        row = build_progress_row(pdf_path, result)
        if pdf_path.name in progress_index:
            progress_rows[progress_index[pdf_path.name]] = row
        else:
            progress_index[pdf_path.name] = len(progress_rows)
            progress_rows.append(row)
        write_progress_csv(progress_rows)
        append_progress_jsonl(row)

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
