"""Quick 1-course smoke test: ingest CSCE 670 into storage_test/, then query."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import pymongo
from lightrag import QueryParam
from wrappers import amake_lightrag

TEST_DIR = Path(__file__).parent / "storage_test"


async def main() -> None:
    client = pymongo.MongoClient(config.MONGODB_URI)
    chunks = list(
        client[config.MONGODB_DB]["chunks_v3"]
        .find(
            {"course_id": "CSCE 670"},
            {"_id": 0, "anchor": 1, "content": 1, "chunk_index": 1},
        )
        .sort("chunk_index", 1)
    )
    client.close()
    print(f"Fetched {len(chunks)} chunks for CSCE 670")

    rag = await amake_lightrag(TEST_DIR)
    for i, chunk in enumerate(chunks, 1):
        text = f"{chunk.get('anchor', '')}\n\n{chunk.get('content', '')}"
        print(f"  Inserting chunk {i}/{len(chunks)}...")
        await rag.ainsert(text)

    print("\nIngestion done. Running query...")
    result = await rag.aquery(
        "what topics does CSCE 670 cover?", param=QueryParam(mode="hybrid")
    )
    print("\n=== ANSWER ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
