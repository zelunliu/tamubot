"""Ingest V2 chunked JSON files into MongoDB Atlas (chunks_v2, courses_v2).

Usage:
    python -m ingestion_pipeline.ingest_v2                         # ingest all
    python -m ingestion_pipeline.ingest_v2 --department CSCE       # single dept
    python -m ingestion_pipeline.ingest_v2 --dry-run               # preview

Requires MONGODB_URI and VOYAGE_API_KEY in .env.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import voyageai
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from rag.models_v2 import ChunkDocV2, CourseDocV2
from rag.models import Instructor
from ingestion_pipeline.v2_logger import STEP3_ROOT, STEP3_V3_ROOT

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "tamubot")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# Flat structure: all versioned JSONs live directly in their step root
CHUNKED_DIR = STEP3_ROOT  # default; overridden by --version v3

EMBEDDING_MODEL = "voyage-3"
EMBEDDING_DIMS = 1024
EMBED_BATCH_SIZE = 50


def get_db():
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI not set.")
        sys.exit(1)
    return MongoClient(MONGODB_URI)[DB_NAME]


def get_voyage_client():
    if not VOYAGE_API_KEY:
        print("ERROR: VOYAGE_API_KEY not set.")
        sys.exit(1)
    return voyageai.Client(api_key=VOYAGE_API_KEY)


def build_anchor(course_id: str, section: str, term: str, header_text: str | None) -> str:
    label = header_text or "Content"
    return f"{course_id} Section {section}, {term} — {label}:"


def _pipeline_version_of(data: dict, version_flag: str) -> str:
    """Return pipeline_version from JSON data, falling back to version_flag."""
    return data.get("pipeline_version", version_flag)


def parse_json_file(filepath: Path) -> dict | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: Could not read {filepath.name}: {e}")
        return None


def build_chunk_docs(data: dict, source_file: str) -> list[dict]:
    meta = data.get("course_metadata", {})
    crn = meta.get("crn", "")
    course_id = meta.get("course_id", "")
    section = meta.get("section", "")
    term = meta.get("term", "")
    instructor = meta.get("instructor") or {}
    instructor_name = instructor.get("name") if instructor else None
    pv = _pipeline_version_of(data, "v2")

    docs = []
    for i, chunk in enumerate(data.get("chunks", [])):
        # v3 chunks carry chunk_index; v2 chunks use loop index
        idx = chunk.get("chunk_index", i)
        anchor = build_anchor(course_id, section, term, chunk.get("header_text"))
        validated = ChunkDocV2(
            crn=crn,
            chunk_index=idx,
            header_text=chunk.get("header_text"),
            header_level=chunk.get("header_level", 0),
            parent_header=chunk.get("parent_header"),
            content=chunk.get("content", ""),
            has_table=chunk.get("has_table", False),
            anchor=anchor,
            course_id=course_id,
            section=section,
            term=term,
            instructor_name=instructor_name,
            pipeline_version=pv,
            source_file=source_file,
        )
        docs.append(validated.model_dump())
    return docs


def build_course_doc(data: dict, source_file: str) -> dict:
    meta = data.get("course_metadata", {})
    chunks = data.get("chunks", [])
    pv = _pipeline_version_of(data, "v2")
    inst_raw = meta.get("instructor")
    instructor = None
    if inst_raw and isinstance(inst_raw, dict):
        try:
            instructor = Instructor(**inst_raw)
        except Exception:
            pass

    validated = CourseDocV2(
        crn=meta.get("crn", ""),
        course_id=meta.get("course_id", ""),
        section=meta.get("section", ""),
        term=meta.get("term", ""),
        instructor=instructor,
        teaching_assistants=[
            ta.get("name", "") for ta in meta.get("teaching_assistants", []) if ta.get("name")
        ],
        meeting_times=meta.get("meeting_times"),
        location=meta.get("location"),
        credit_hours=meta.get("credit_hours"),
        syllabus_url=meta.get("syllabus_url"),
        doc_id=meta.get("doc_id"),
        chunk_count=len(chunks),
        header_count=sum(1 for c in chunks if c.get("header_level", 0) > 0),
        pipeline_version=pv,
        source_file=source_file,
    )
    return validated.model_dump()


def embed_chunks(voyage: voyageai.Client, chunk_docs: list[dict]) -> list[dict]:
    texts = [doc["anchor"] + " " + doc["content"] for doc in chunk_docs]

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        retries = 0
        while True:
            try:
                result = voyage.embed(batch, model=EMBEDDING_MODEL, input_type="document")
                for j, emb in enumerate(result.embeddings):
                    chunk_docs[start + j]["embedding"] = emb
                break
            except Exception as e:
                retries += 1
                if retries > 3:
                    raise
                wait = 2 ** retries
                print(f"  WARN: Embedding error ({e}), retrying in {wait}s...")
                time.sleep(wait)

    return chunk_docs


def upsert_chunks(db, chunk_docs: list[dict]) -> int:
    if not chunk_docs:
        return 0
    ops = [
        UpdateOne(
            {"crn": doc["crn"], "chunk_index": doc["chunk_index"]},
            {"$set": doc},
            upsert=True,
        )
        for doc in chunk_docs
    ]
    result = db["chunks_v2"].bulk_write(ops, ordered=False)
    return result.upserted_count + result.modified_count


def upsert_course(db, course_doc: dict):
    db["courses_v2"].update_one(
        {"crn": course_doc["crn"]},
        {"$set": course_doc},
        upsert=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest V2/V3 chunked syllabi into MongoDB Atlas")
    parser.add_argument("--department", type=str, help="Filter by department prefix (e.g. CSCE)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument(
        "--version", choices=["v2", "v3"], default="v2",
        help="Pipeline version to ingest: v2 (header-based) or v3 (flat chunks). Default: v2",
    )
    args = parser.parse_args()

    chunked_dir = STEP3_V3_ROOT if args.version == "v3" else STEP3_ROOT
    chunk_coll = f"chunks_{args.version}"
    course_coll = f"courses_{args.version}"

    if not chunked_dir.exists():
        print(f"ERROR: Chunked directory not found: {chunked_dir}")
        sys.exit(1)

    # Exclude meta files; for multiple versions of same stem, take the latest _vNNN
    all_json = sorted(chunked_dir.glob("*.json"))
    all_json = [f for f in all_json if not f.name.startswith("_")]
    # Deduplicate by stem (strip _vNNN suffix), keeping highest version
    import re as _re
    seen_stems: dict[str, Path] = {}
    for f in all_json:
        base_stem = _re.sub(r"_v\d{3}$", "", f.stem)
        existing = seen_stems.get(base_stem)
        if existing is None or f.name > existing.name:
            seen_stems[base_stem] = f
    json_files = sorted(seen_stems.values(), key=lambda f: f.stem)
    if args.department:
        dept = args.department.upper()
        json_files = [f for f in json_files if f"_{dept}_" in f.name]

    print(f"Found {len(json_files)} {args.version.upper()} JSON files to ingest")
    if not json_files:
        return

    if args.dry_run:
        print("DRY RUN — no database writes")
        for f in json_files:
            data = parse_json_file(f)
            if data:
                chunks = build_chunk_docs(data, f.name)
                print(f"  {f.name}: {len(chunks)} chunks")
        return

    db = get_db()
    voyage = get_voyage_client()

    total_chunks = 0
    total_files = 0
    errors = []

    for i, filepath in enumerate(json_files):
        data = parse_json_file(filepath)
        if data is None:
            errors.append(filepath.name)
            continue

        try:
            chunk_docs = build_chunk_docs(data, filepath.name)
            course_doc = build_course_doc(data, filepath.name)

            if chunk_docs:
                embed_chunks(voyage, chunk_docs)

            db[course_coll].update_one(
                {"crn": course_doc["crn"]},
                {"$set": course_doc},
                upsert=True,
            )
            if chunk_docs:
                ops = [
                    UpdateOne(
                        {"crn": doc["crn"], "chunk_index": doc["chunk_index"]},
                        {"$set": doc},
                        upsert=True,
                    )
                    for doc in chunk_docs
                ]
                db[chunk_coll].bulk_write(ops, ordered=False)

            total_chunks += len(chunk_docs)
            total_files += 1
            print(f"  [{i+1}/{len(json_files)}] {filepath.name}: {len(chunk_docs)} chunks")

        except Exception as e:
            errors.append(filepath.name)
            print(f"  [{i+1}/{len(json_files)}] ERROR {filepath.name}: {e}")

    print(f"\nIngestion complete: {total_files} files, {total_chunks} chunks")
    print(f"  {course_coll}: {db[course_coll].count_documents({})}")
    print(f"  {chunk_coll}:  {db[chunk_coll].count_documents({})}")
    if errors:
        print(f"  Errors ({len(errors)}): {errors}")


if __name__ == "__main__":
    main()
