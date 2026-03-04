"""Create MongoDB Atlas indexes for the three collections.

Usage:
    python -m ingestion_pipeline.setup_atlas

Requires MONGODB_URI in .env (or environment).
"""

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


def main():
    db = get_db()
    print(f"Setting up indexes on database '{DB_NAME}'...\n")

    print("Standard indexes:")
    setup_standard_indexes(db)

    print("\nAtlas Search indexes:")
    search_indexes = setup_search_indexes(db)

    print("\nDone.")
    if search_indexes:
        print(
            "NOTE: Atlas Search indexes build asynchronously. "
            "Check Atlas UI or run db.chunks.listSearchIndexes() to verify status."
        )


if __name__ == "__main__":
    main()
