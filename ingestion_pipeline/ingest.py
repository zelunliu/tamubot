"""Ingest Gemini-parsed JSON files into MongoDB Atlas.

Usage:
    python -m ingestion_pipeline.ingest                          # ingest all
    python -m ingestion_pipeline.ingest --department CSCE         # single department
    python -m ingestion_pipeline.ingest --dry-run                 # preview without writing

Requires MONGODB_URI and VOYAGE_API_KEY in .env.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import voyageai
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from rag.models import ChunkDoc, CourseDoc

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "tamubot")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PARSED_DIR = Path("tamu_data/processed/gemini_parsed")

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


def build_anchor(course_id: str, section: str, term: str, category: str) -> str:
    """Build a contextual anchor string prepended to chunk text before embedding."""
    return f"{course_id} Section {section}, {term} — {category}:"


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()


def parse_json_file(filepath: Path) -> dict | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: Could not read {filepath.name}: {e}")
        return None


def build_chunk_docs(data: dict, source_file: str) -> list[dict]:
    """Convert parsed JSON into validated chunk documents ready for MongoDB."""
    meta = data.get("course_metadata", {})
    crn = meta.get("crn", "")
    course_id = meta.get("course_id", "")
    section = meta.get("section", "")
    term = meta.get("term", "")
    instructor = meta.get("instructor", {})
    instructor_name = instructor.get("name") if instructor else None

    docs = []
    for i, chunk in enumerate(data.get("chunks", [])):
        category = chunk.get("category", "")
        anchor = build_anchor(course_id, section, term, category)
        validated = ChunkDoc(
            crn=crn,
            chunk_index=i,
            category=category,
            title=chunk.get("title") or "",
            content=chunk.get("content") or "",
            has_table=chunk.get("has_table") or False,
            anchor=anchor,
            course_id=course_id,
            section=section,
            term=term,
            instructor_name=instructor_name,
            source_file=source_file,
        )
        docs.append(validated.model_dump())
    return docs


def build_course_doc(data: dict, source_file: str) -> dict:
    meta = data.get("course_metadata", {})
    chunks = data.get("chunks", [])
    categories = list({c.get("category", "") for c in chunks})
    completeness = data.get("completeness_check", {})
    validated = CourseDoc(
        crn=meta.get("crn", ""),
        course_id=meta.get("course_id", ""),
        section=meta.get("section", ""),
        term=meta.get("term", ""),
        instructor=meta.get("instructor"),
        teaching_assistants=meta.get("teaching_assistants", []),
        meeting_times=meta.get("meeting_times"),
        location=meta.get("location"),
        credit_hours=meta.get("credit_hours"),
        categories_present=categories,
        chunk_count=len(chunks),
        boilerplate_policies=data.get("boilerplate_policies", []),
        missing_sections=completeness.get("missing_sections", []),
        completeness_warnings=completeness.get("warnings", []),
        source_file=source_file,
    )
    return validated.model_dump()


def build_policy_ops(data: dict) -> list[UpdateOne]:
    """Build upsert operations for boilerplate policies."""
    crn = data.get("course_metadata", {}).get("crn", "")
    ops = []
    for name in data.get("boilerplate_policies", []):
        h = sha256_hash(name)
        ops.append(UpdateOne(
            {"policy_hash": h},
            {
                "$set": {"policy_name": name, "ingested_at": datetime.utcnow()},
                "$setOnInsert": {"policy_hash": h, "full_text": None},
                "$addToSet": {"courses_referencing": crn},
            },
            upsert=True,
        ))
    return ops


def embed_chunks(voyage: voyageai.Client, chunk_docs: list[dict]) -> list[dict]:
    """Embed chunk docs in batches using Voyage AI. Mutates docs in-place."""
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
                    print(f"  ERROR: Embedding failed after 3 retries: {e}")
                    raise
                wait = 2 ** retries
                print(f"  WARN: Embedding error ({e}), retrying in {wait}s...")
                time.sleep(wait)

    return chunk_docs


def upsert_chunks(db, chunk_docs: list[dict]):
    """Idempotent upsert of chunk documents keyed by (crn, chunk_index)."""
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
    result = db["chunks"].bulk_write(ops, ordered=False)
    return result.upserted_count + result.modified_count


def upsert_course(db, course_doc: dict):
    db["courses"].update_one(
        {"crn": course_doc["crn"]},
        {"$set": course_doc},
        upsert=True,
    )


def _crn_from_filename(filepath: Path) -> str | None:
    """Extract CRN from filename pattern like 202611_CSCE_670_600_46627.json."""
    parts = filepath.stem.split("_")
    if parts and parts[-1].isdigit():
        return parts[-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Ingest parsed syllabi into MongoDB Atlas")
    parser.add_argument("--department", type=str, help="Filter by department prefix (e.g. CSCE)")
    parser.add_argument("--crns-file", type=str,
                        help="JSON file with 'crns' list — ingest only those CRNs "
                             "(e.g. tamu_data/evals/eval_corpus.json)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    args = parser.parse_args()

    if not PARSED_DIR.exists():
        print(f"ERROR: Parsed directory not found: {PARSED_DIR}")
        sys.exit(1)

    json_files = sorted(PARSED_DIR.glob("*.json"))
    if args.department:
        dept = args.department.upper()
        json_files = [f for f in json_files if f"_{dept}_" in f.name]

    if args.crns_file:
        with open(args.crns_file) as f:
            corpus = json.load(f)
        target_crns = {str(c) for c in corpus.get("crns", [])}
        if not target_crns:
            print(f"ERROR: No CRNs found in {args.crns_file}")
            sys.exit(1)
        json_files = [f for f in json_files if _crn_from_filename(f) in target_crns]
        print(f"  --crns-file: filtered to {len(json_files)} files "
              f"matching {len(target_crns)} corpus CRNs")

    print(f"Found {len(json_files)} JSON files to ingest")
    if not json_files:
        return

    if args.dry_run:
        print("DRY RUN — no database writes will occur")
        for f in json_files:
            data = parse_json_file(f)
            if data:
                if "error" in data:
                    print(f"  {f.name}: SKIP (error file)")
                    continue
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

        if "error" in data:
            print(f"  SKIP: {filepath.name} — previously failed")
            continue

        try:
            # Build documents
            chunk_docs = build_chunk_docs(data, filepath.name)
            course_doc = build_course_doc(data, filepath.name)
            policy_ops = build_policy_ops(data)

            # Embed
            if chunk_docs:
                embed_chunks(voyage, chunk_docs)

            # Write to MongoDB
            upsert_course(db, course_doc)
            upsert_chunks(db, chunk_docs)
            if policy_ops:
                db["policies"].bulk_write(policy_ops, ordered=False)

            total_chunks += len(chunk_docs)
            total_files += 1
            print(f"  [{i+1}/{len(json_files)}] {filepath.name}: {len(chunk_docs)} chunks ingested")

        except Exception as e:
            errors.append(filepath.name)
            print(f"  [{i+1}/{len(json_files)}] ERROR {filepath.name}: {e}")

    print(f"\nIngestion complete: {total_files} files, {total_chunks} chunks")
    print(f"  Courses: {db['courses'].count_documents({})}")
    print(f"  Chunks:  {db['chunks'].count_documents({})}")
    print(f"  Policies: {db['policies'].count_documents({})}")
    if errors:
        print(f"  Errors ({len(errors)}): {errors}")


if __name__ == "__main__":
    main()
