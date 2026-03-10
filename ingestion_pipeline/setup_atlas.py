"""Create MongoDB Atlas indexes for the three collections.

Usage:
    python -m ingestion_pipeline.setup_atlas

Requires MONGODB_URI in .env (or environment).
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "tamubot")


def get_db():
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI not set in environment.")
        sys.exit(1)
    client = MongoClient(MONGODB_URI)
    return client[DB_NAME]


def setup_standard_indexes(db):
    """Create standard MongoDB indexes for query filtering."""
    chunks = db["chunks"]
    chunks.create_index("crn", name="idx_crn")
    chunks.create_index("course_id", name="idx_course_id")
    chunks.create_index("category", name="idx_category")
    chunks.create_index(
        [("course_id", 1), ("category", 1)],
        name="idx_course_category",
    )
    chunks.create_index(
        [("crn", 1), ("chunk_index", 1)],
        unique=True,
        name="idx_crn_chunk_unique",
    )
    print("  [chunks] standard indexes created")

    courses = db["courses"]
    courses.create_index("crn", unique=True, name="idx_crn")
    courses.create_index("course_id", name="idx_course_id")
    courses.create_index("term", name="idx_term")
    print("  [courses] standard indexes created")

    policies = db["policies"]
    policies.create_index("policy_hash", unique=True, name="idx_policy_hash")
    policies.create_index("policy_name", name="idx_policy_name")
    print("  [policies] standard indexes created")


def setup_search_indexes(db):
    """Create Atlas Search and Vector Search indexes.

    These use the Atlas Search Index API (requires M10+ cluster or Atlas free
    tier with search enabled).
    """
    chunks = db["chunks"]

    # Vector search index for embeddings (Voyage AI voyage-3: 1024 dims)
    vector_idx = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "course_id"},
                {"type": "filter", "path": "category"},
                {"type": "filter", "path": "term"},
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )

    # Full-text search index for keyword/BM25 retrieval
    text_idx = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "title": {"type": "string", "analyzer": "lucene.standard"},
                    "course_id": {"type": "token"},
                    "category": {"type": "token"},
                    "instructor_name": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name="text_index",
        type="search",
    )

    existing = [idx["name"] for idx in chunks.list_search_indexes()]

    created = []
    indexes = {"vector_index": vector_idx, "text_index": text_idx}
    for name, idx in indexes.items():
        if name in existing:
            print(f"  [chunks] search index '{name}' already exists — skipping")
        else:
            chunks.create_search_index(idx)
            created.append(name)
            print(f"  [chunks] search index '{name}' created (may take a few minutes to build)")

    return created


def setup_standard_indexes_v2(db):
    """Create standard MongoDB indexes for the V2 collections."""
    chunks_v2 = db["chunks_v2"]
    chunks_v2.create_index("crn", name="idx_crn")
    chunks_v2.create_index("course_id", name="idx_course_id")
    chunks_v2.create_index("term", name="idx_term")
    chunks_v2.create_index("header_level", name="idx_header_level")
    chunks_v2.create_index(
        [("crn", 1), ("chunk_index", 1)],
        unique=True,
        name="idx_crn_chunk_unique",
    )
    print("  [chunks_v2] standard indexes created")

    courses_v2 = db["courses_v2"]
    courses_v2.create_index("crn", unique=True, name="idx_crn")
    courses_v2.create_index("course_id", name="idx_course_id")
    courses_v2.create_index("term", name="idx_term")
    print("  [courses_v2] standard indexes created")


def setup_search_indexes_v2(db):
    """Create Atlas Search and Vector Search indexes for V2 collections."""
    chunks_v2 = db["chunks_v2"]

    vector_idx = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "course_id"},
                {"type": "filter", "path": "term"},
            ]
        },
        name="vector_index_v2",
        type="vectorSearch",
    )

    text_idx = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "header_text": {"type": "string", "analyzer": "lucene.standard"},
                    "course_id": {"type": "token"},
                    "term": {"type": "token"},
                    "instructor_name": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name="text_index_v2",
        type="search",
    )

    existing = [idx["name"] for idx in chunks_v2.list_search_indexes()]
    created = []
    indexes = {"vector_index_v2": vector_idx, "text_index_v2": text_idx}
    for name, idx in indexes.items():
        if name in existing:
            print(f"  [chunks_v2] search index '{name}' already exists — skipping")
        else:
            chunks_v2.create_search_index(idx)
            created.append(name)
            print(f"  [chunks_v2] search index '{name}' created (may take a few minutes)")
    return created


def setup_standard_indexes_v3(db):
    """Create standard MongoDB indexes for the V3 collections."""
    chunks_v3 = db["chunks_v3"]
    chunks_v3.create_index("crn", name="idx_crn")
    chunks_v3.create_index("course_id", name="idx_course_id")
    chunks_v3.create_index("term", name="idx_term")
    chunks_v3.create_index(
        [("crn", 1), ("chunk_index", 1)],
        unique=True,
        name="idx_crn_chunk_unique",
    )
    print("  [chunks_v3] standard indexes created")

    courses_v3 = db["courses_v3"]
    courses_v3.create_index("crn", unique=True, name="idx_crn")
    courses_v3.create_index("course_id", name="idx_course_id")
    courses_v3.create_index("term", name="idx_term")
    print("  [courses_v3] standard indexes created")


def setup_search_indexes_v3(db):
    """Create Atlas Search and Vector Search indexes for V3 collections."""
    chunks_v3 = db["chunks_v3"]

    vector_idx = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "course_id"},
                {"type": "filter", "path": "term"},
            ]
        },
        name="vector_index_v3",
        type="vectorSearch",
    )

    text_idx = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "course_id": {"type": "token"},
                    "term": {"type": "token"},
                    "instructor_name": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name="text_index_v3",
        type="search",
    )

    existing = [idx["name"] for idx in chunks_v3.list_search_indexes()]
    created = []
    indexes = {"vector_index_v3": vector_idx, "text_index_v3": text_idx}
    for name, idx in indexes.items():
        if name in existing:
            print(f"  [chunks_v3] search index '{name}' already exists — skipping")
        else:
            chunks_v3.create_search_index(idx)
            created.append(name)
            print(f"  [chunks_v3] search index '{name}' created (may take a few minutes)")
    return created


def main():
    parser = argparse.ArgumentParser(description="Create MongoDB Atlas indexes")
    parser.add_argument(
        "--version", choices=["v1", "v2", "v3", "all"], default="all",
        help="Which index set to create (default: all)",
    )
    args = parser.parse_args()

    db = get_db()
    print(f"Setting up indexes on database '{DB_NAME}'...\n")

    if args.version in ("v1", "all"):
        print("V1 — Standard indexes:")
        setup_standard_indexes(db)
        print("\nV1 — Atlas Search indexes:")
        search_indexes = setup_search_indexes(db)
    else:
        search_indexes = []

    if args.version in ("v2", "all"):
        print("\nV2 — Standard indexes:")
        setup_standard_indexes_v2(db)
        print("\nV2 — Atlas Search indexes:")
        search_indexes += setup_search_indexes_v2(db)

    if args.version in ("v3", "all"):
        print("\nV3 — Standard indexes:")
        setup_standard_indexes_v3(db)
        print("\nV3 — Atlas Search indexes:")
        search_indexes += setup_search_indexes_v3(db)

    print("\nDone.")
    if search_indexes:
        print(
            "NOTE: Atlas Search indexes build asynchronously. "
            "Check Atlas UI or run db.chunks_v3.listSearchIndexes() to verify status."
        )


if __name__ == "__main__":
    main()
