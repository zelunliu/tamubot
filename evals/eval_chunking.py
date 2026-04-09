"""Chunking quality evaluation for TamuBot.

Measures embedding-space cohesion and separation for chunks stored in MongoDB,
using Voyage-3 cosine similarity as a proxy for semantic quality.

Provides:
  load_chunks_for_crns()     — load chunks from MongoDB for a list of CRNs
  retrieve_for_query()       — retrieve top-k chunks for a query via hybrid search
  intra_chunk_cohesion()     — avg cosine similarity of chunk content to its category centroid
  inter_chunk_separation()   — avg cosine distance between chunk centroids across categories
  chunk_size_stats()         — token-count distribution (min/max/mean/std)
  run_eval()                 — full eval loop over a list of CRNs with Langfuse logging
  print_summary()            — print a formatted summary table
  main()                     — CLI entry point

Usage:
    python evals/eval_chunking.py --crns-file tamu_data/evals/eval_corpus.json
    python evals/eval_chunking.py --crn 123456
    python evals/eval_chunking.py --crns-file tamu_data/evals/eval_corpus.json --output results.json
    python evals/eval_chunking.py --query "What is the grading policy?" --crn 123456
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _centroid(vectors: list[list[float]]) -> list[float]:
    """Compute element-wise mean of a list of vectors."""
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    result = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            result[i] += x
    return [x / n for x in result]


def _tokens_approx(text: str) -> int:
    """Approximate token count: len(text) / 4, rounded."""
    return max(0, round(len(text) / 4))


# ---------------------------------------------------------------------------
# Task 2: MongoDB loader
# ---------------------------------------------------------------------------

def load_chunks_for_crns(crns: list[str]) -> list[dict]:
    """Load all chunks from MongoDB for the given CRNs.

    Returns a flat list of chunk dicts, each containing at minimum:
      crn, chunk_index, category, content, course_id, embedding (may be None).

    Args:
        crns: List of CRN strings (e.g. ["123456", "234567"]).

    Returns:
        List of chunk dicts from the chunks_v3 collection.
    """
    from pymongo import MongoClient

    client = MongoClient(config.MONGODB_URI)
    db = client[config.MONGODB_DB]
    collection = db["chunks_v3"]

    chunks = list(collection.find(
        {"crn": {"$in": crns}},
        {
            "crn": 1,
            "chunk_index": 1,
            "category": 1,
            "content": 1,
            "course_id": 1,
            "embedding": 1,
            "has_table": 1,
            "anchor": 1,
            "_id": 0,
        },
    ))
    client.close()
    return chunks


# ---------------------------------------------------------------------------
# Task 3: retrieve_for_query
# ---------------------------------------------------------------------------

def retrieve_for_query(
    query: str,
    crn: Optional[str] = None,
    course_id: Optional[str] = None,
    top_k: int = 10,
) -> list[dict]:
    """Retrieve top-k chunks for a query using hybrid search.

    Args:
        query:     Search query string.
        crn:       Optional CRN filter (resolved to course_id if provided).
        course_id: Optional course_id filter (e.g. "CSCE 638").
        top_k:     Number of chunks to return after reranking.

    Returns:
        List of chunk dicts sorted by relevance (descending).
    """
    from rag.tools.mongo import hybrid_search
    from rag.tools.voyage import rerank

    filters: dict = {}
    if crn:
        # Look up course_id from crn if not provided
        if not course_id:
            from pymongo import MongoClient
            client = MongoClient(config.MONGODB_URI)
            db = client[config.MONGODB_DB]
            doc = db["chunks_v3"].find_one({"crn": crn}, {"course_id": 1})
            client.close()
            if doc:
                course_id = doc.get("course_id")
    if course_id:
        filters["course_id"] = course_id

    # Retrieve candidates
    candidates = hybrid_search(query, filters=filters or None, k=top_k * 2)
    if not candidates:
        return []

    # Rerank and slice
    return rerank(query, candidates, top_k=top_k)


# ---------------------------------------------------------------------------
# Embedding metrics
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed a list of texts using Voyage-3."""
    import voyageai
    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    # Voyage AI has a batch limit; chunk into groups of 128
    batch_size = 128
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = vo.embed(batch, model="voyage-3", input_type=input_type)
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def intra_chunk_cohesion(chunks: list[dict]) -> dict:
    """Measure intra-chunk cohesion: how similar each chunk is to its category centroid.

    For each category, embeds all chunk contents, computes the centroid, then
    averages the cosine similarity of each chunk to that centroid.

    Args:
        chunks: List of chunk dicts (must have 'content' and 'category' keys).

    Returns:
        Dict with:
          overall_cohesion  — mean cosine similarity across all chunks
          per_category      — {category: mean_cosine} mapping
          n_chunks          — total number of chunks evaluated
    """
    if not chunks:
        return {"overall_cohesion": 0.0, "per_category": {}, "n_chunks": 0}

    # Group by category
    by_category: dict[str, list[dict]] = {}
    for c in chunks:
        cat = c.get("category", "UNKNOWN")
        by_category.setdefault(cat, []).append(c)

    # Embed all contents at once for efficiency
    contents = [c.get("content", "") for c in chunks]
    chunk_idx_map = {id(c): i for i, c in enumerate(chunks)}

    try:
        all_embeddings = _embed_texts(contents, input_type="document")
    except Exception as e:
        return {"error": str(e), "overall_cohesion": 0.0, "per_category": {}, "n_chunks": 0}

    # Map chunk → embedding
    embeddings_by_chunk: dict[int, list[float]] = {
        id(c): all_embeddings[i] for i, c in enumerate(chunks)
    }

    per_category: dict[str, float] = {}
    all_sims: list[float] = []

    for cat, cat_chunks in by_category.items():
        cat_embeds = [embeddings_by_chunk[id(c)] for c in cat_chunks]
        centroid = _centroid(cat_embeds)
        sims = [_cosine(e, centroid) for e in cat_embeds]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        per_category[cat] = round(avg_sim, 4)
        all_sims.extend(sims)

    overall = sum(all_sims) / len(all_sims) if all_sims else 0.0
    return {
        "overall_cohesion": round(overall, 4),
        "per_category": per_category,
        "n_chunks": len(chunks),
    }


def inter_chunk_separation(chunks: list[dict]) -> dict:
    """Measure inter-chunk separation: cosine distance between category centroids.

    A higher distance means categories are more semantically distinct in
    embedding space — a sign of clean chunking boundaries.

    Args:
        chunks: List of chunk dicts (must have 'content' and 'category' keys).

    Returns:
        Dict with:
          avg_separation     — mean pairwise centroid cosine distance
          pairwise           — {(cat_a, cat_b): distance} for all category pairs
          n_categories       — number of distinct categories
    """
    if not chunks:
        return {"avg_separation": 0.0, "pairwise": {}, "n_categories": 0}

    by_category: dict[str, list[str]] = {}
    for c in chunks:
        cat = c.get("category", "UNKNOWN")
        by_category.setdefault(cat, []).append(c.get("content", ""))

    categories = list(by_category.keys())
    if len(categories) < 2:
        return {
            "avg_separation": 0.0,
            "pairwise": {},
            "n_categories": len(categories),
        }

    # Embed each category's contents and compute centroids
    centroids: dict[str, list[float]] = {}
    for cat, texts in by_category.items():
        try:
            embeds = _embed_texts(texts, input_type="document")
            centroids[cat] = _centroid(embeds)
        except Exception:
            continue

    # Pairwise cosine distance between centroids
    pairwise: dict[str, float] = {}
    distances: list[float] = []
    cats = list(centroids.keys())
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a, b = cats[i], cats[j]
            sim = _cosine(centroids[a], centroids[b])
            dist = 1.0 - sim
            pairwise[f"{a} vs {b}"] = round(dist, 4)
            distances.append(dist)

    avg = sum(distances) / len(distances) if distances else 0.0
    return {
        "avg_separation": round(avg, 4),
        "pairwise": pairwise,
        "n_categories": len(cats),
    }


def chunk_size_stats(chunks: list[dict]) -> dict:
    """Compute token-count distribution statistics for a list of chunks.

    Args:
        chunks: List of chunk dicts with 'content' key.

    Returns:
        Dict with min, max, mean, std, median token counts and n_chunks.
    """
    if not chunks:
        return {"min": 0, "max": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "n_chunks": 0}

    counts = [_tokens_approx(c.get("content", "")) for c in chunks]
    n = len(counts)
    mean = sum(counts) / n
    variance = sum((x - mean) ** 2 for x in counts) / n
    std = math.sqrt(variance)
    sorted_counts = sorted(counts)
    mid = n // 2
    median = (sorted_counts[mid] if n % 2 == 1
              else (sorted_counts[mid - 1] + sorted_counts[mid]) / 2.0)

    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(mean, 1),
        "std": round(std, 1),
        "median": round(median, 1),
        "n_chunks": n,
    }


# ---------------------------------------------------------------------------
# Task 4: run_eval loop + Langfuse logging
# ---------------------------------------------------------------------------

def run_eval(
    crns: list[str],
    trace_id: Optional[str] = None,
    log_to_langfuse: bool = True,
) -> dict:
    """Run full chunking evaluation for a list of CRNs.

    Loads chunks from MongoDB, computes cohesion, separation, and size stats,
    then optionally logs aggregate scores to Langfuse.

    Args:
        crns:             List of CRN strings to evaluate.
        trace_id:         Langfuse trace ID to attach scores to. Optional.
        log_to_langfuse:  Whether to upload scores to Langfuse (default True).

    Returns:
        Dict with keys: crns, n_chunks, size_stats, cohesion, separation,
        and per_crn results.
    """
    print(f"Loading chunks for {len(crns)} CRN(s)...")
    chunks = load_chunks_for_crns(crns)
    if not chunks:
        return {"error": "No chunks found for given CRNs", "crns": crns}

    print(f"  Loaded {len(chunks)} chunks across {len(set(c.get('crn') for c in chunks))} CRNs.")

    # Size stats (no API calls needed)
    size = chunk_size_stats(chunks)

    # Cohesion (requires Voyage embeds)
    print("Computing intra-chunk cohesion...")
    cohesion = intra_chunk_cohesion(chunks)

    # Separation (requires Voyage embeds — use cached centroids when possible)
    print("Computing inter-chunk separation...")
    separation = inter_chunk_separation(chunks)

    # Per-CRN breakdown (size only — avoids redundant embed calls)
    per_crn: dict[str, dict] = {}
    for crn in crns:
        crn_chunks = [c for c in chunks if c.get("crn") == crn]
        per_crn[crn] = chunk_size_stats(crn_chunks)

    result = {
        "crns": crns,
        "n_chunks": len(chunks),
        "size_stats": size,
        "cohesion": cohesion,
        "separation": separation,
        "per_crn": per_crn,
    }

    # Langfuse logging
    if log_to_langfuse:
        try:
            from rag.tools.langfuse import get_langfuse
            lf = get_langfuse()
            if lf:
                scores_to_log = {
                    "chunking_cohesion": cohesion.get("overall_cohesion", 0.0),
                    "chunking_separation": separation.get("avg_separation", 0.0),
                    "chunk_mean_tokens": size.get("mean", 0.0),
                    "chunk_std_tokens": size.get("std", 0.0),
                }
                for name, value in scores_to_log.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        lf.create_score(
                            trace_id=trace_id or "eval_chunking",
                            name=name,
                            value=float(value),
                            comment=f"eval_chunking — {len(crns)} CRNs, {len(chunks)} chunks",
                        )
                print(f"  Langfuse scores logged for trace '{trace_id or 'eval_chunking'}'.")
        except Exception as e:
            print(f"  Langfuse logging skipped: {e}")

    return result


# ---------------------------------------------------------------------------
# Task 5: print_summary + JSON output
# ---------------------------------------------------------------------------

def print_summary(result: dict) -> None:
    """Print a formatted summary of chunking evaluation results.

    Args:
        result: Dict returned by run_eval().
    """
    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    crns = result.get("crns", [])
    n = result.get("n_chunks", 0)
    size = result.get("size_stats", {})
    cohesion = result.get("cohesion", {})
    separation = result.get("separation", {})

    sep_line = "=" * 56
    print(f"\n{sep_line}")
    print(f"  CHUNKING EVAL SUMMARY — {len(crns)} CRN(s), {n} chunks")
    print(sep_line)

    print(f"\n  TOKEN SIZE DISTRIBUTION")
    print(f"    min:    {size.get('min', 'N/A')}")
    print(f"    max:    {size.get('max', 'N/A')}")
    print(f"    mean:   {size.get('mean', 'N/A')}")
    print(f"    std:    {size.get('std', 'N/A')}")
    print(f"    median: {size.get('median', 'N/A')}")

    print(f"\n  EMBEDDING COHESION (intra-category)")
    print(f"    overall: {cohesion.get('overall_cohesion', 'N/A')}")
    per_cat = cohesion.get("per_category", {})
    if per_cat:
        for cat, score in sorted(per_cat.items()):
            print(f"      {cat:<30s}  {score:.4f}")

    print(f"\n  EMBEDDING SEPARATION (inter-category)")
    print(f"    avg:     {separation.get('avg_separation', 'N/A')}")
    print(f"    n_cats:  {separation.get('n_categories', 'N/A')}")

    per_crn = result.get("per_crn", {})
    if per_crn and len(per_crn) > 1:
        print(f"\n  PER-CRN TOKEN STATS")
        print(f"    {'CRN':<12}  {'n':>5}  {'mean':>7}  {'std':>7}")
        for crn, s in sorted(per_crn.items()):
            print(f"    {crn:<12}  {s.get('n_chunks', 0):>5}  "
                  f"{s.get('mean', 0):>7.1f}  {s.get('std', 0):>7.1f}")

    print(f"\n{sep_line}\n")


# ---------------------------------------------------------------------------
# Task 6: CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TamuBot chunking quality evaluation"
    )
    parser.add_argument(
        "--crn",
        help="Single CRN to evaluate",
    )
    parser.add_argument(
        "--crns-file",
        type=Path,
        help="Path to JSON file containing a list of CRNs (e.g. eval_corpus.json)",
    )
    parser.add_argument(
        "--query",
        help="Run retrieve_for_query() for this query and show top-k results",
    )
    parser.add_argument(
        "--course-id",
        help="Course ID filter for --query mode (e.g. 'CSCE 638')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results for --query mode (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write eval results JSON to this path",
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Skip Langfuse score logging",
    )
    parser.add_argument(
        "--trace-id",
        help="Langfuse trace ID for score upload (optional)",
    )
    args = parser.parse_args()

    # Validate API key requirements
    if not config.VOYAGE_API_KEY and not args.query:
        print("ERROR: VOYAGE_API_KEY is not set. Embedding metrics require Voyage AI.")
        sys.exit(1)

    # Query mode: retrieve and print top chunks
    if args.query:
        print(f"Retrieving top-{args.top_k} chunks for: '{args.query}'")
        chunks = retrieve_for_query(
            query=args.query,
            crn=args.crn,
            course_id=args.course_id,
            top_k=args.top_k,
        )
        if not chunks:
            print("  No chunks retrieved.")
        else:
            for i, c in enumerate(chunks, 1):
                score = c.get("score", "N/A")
                course = c.get("course_id", "?")
                cat = c.get("category", "?")
                preview = c.get("content", "")[:100].replace("\n", " ")
                print(f"  [{i:2d}] {course} | {cat} | score={score} | {preview}...")
        return

    # Resolve CRNs
    crns: list[str] = []
    if args.crn:
        crns = [args.crn]
    elif args.crns_file:
        if not args.crns_file.exists():
            print(f"ERROR: File not found: {args.crns_file}")
            sys.exit(1)
        with args.crns_file.open(encoding="utf-8") as f:
            data = json.load(f)
        # Support both plain list and dict with "crns" key
        if isinstance(data, list):
            crns = [str(item) for item in data]
        elif isinstance(data, dict):
            crns = [str(item) for item in data.get("crns", data.get("crnList", list(data.keys())))]
        else:
            print("ERROR: Unrecognised CRN file format (expected list or dict).")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    if not crns:
        print("ERROR: No CRNs resolved from input.")
        sys.exit(1)

    print(f"Evaluating {len(crns)} CRN(s)...")
    result = run_eval(
        crns=crns,
        trace_id=args.trace_id,
        log_to_langfuse=not args.no_langfuse,
    )

    print_summary(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serialisable items if any
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
