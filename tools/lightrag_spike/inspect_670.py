"""Inspect LightRAG graph quality for a CSCE 670 ingestion.

Usage:
    python tools/lightrag_spike/inspect_670.py --storage-dir storage_iter1 [--no-queries]

Reports:
  1. Entity stats: count, type distribution, anchoring score
  2. Novel entity types (not in SYLLABUS_ENTITY_TYPES)
  3. Relationship count and avg weight
  4. Source-doc fact coverage (regex-based, no LLM)
  5. Query precision: 5 test questions, context token count (requires LightRAG)
"""

import argparse
import asyncio
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wrappers import SYLLABUS_ENTITY_TYPES, amake_lightrag_improved  # noqa: E402

SOURCE_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "tamu_data/processed/v3_step2_boilerplate/202611_CSCE_670_600_46627_v010.md"
)
SPIKE_DIR = Path(__file__).resolve().parent
COURSE_ID = "CSCE 670"

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"

TEST_QUESTIONS = [
    "What is the grading policy for CSCE 670?",
    "Who teaches CSCE 670 and what are their office hours?",
    "What topics does CSCE 670 cover each week?",
    "What are the prerequisites for CSCE 670?",
    "What are the homework and quiz deadlines for CSCE 670?",
]

# Key facts that should be present in the graph (pattern → description)
EXPECTED_FACTS = {
    r"Yu Zhang": "instructor name",
    r"yuzhang@tamu\.edu": "instructor email",
    r"PETR 222": "office location",
    r"30%": "homework weight",
    r"20%": "quiz weight (or project weight)",
    r"five (late days|programming assignments)": "late days / 5 assignments",
    r"CSCE 310|CSCE 603": "prerequisites",
    r"Jan(uary)? 26|Jan 26": "HW0 due date",
    r"Feb(ruary)? 9|Feb 9": "HW1 due date",
    r"May 4": "final exam date",
    r"HRBB 113": "classroom",
    r"MWF|Monday.*Wednesday.*Friday": "meeting days",
    r"1:50\s*[Pp][Mm]": "start time",
    r"Mining of Massive Datasets|Introduction to Information Retrieval": "textbooks",
    r"Boolean Retrieval": "week 1 topic",
    r"Large Language Models": "LLM topic weeks",
    r"word2vec|Word2Vec": "week 8 topic",
    r"3:30\s*[–-]\s*5:30\s*[Pp][Mm]": "final exam time",
}


def parse_graphml(storage_dir: Path) -> tuple[list[dict], list[dict]]:
    """Parse GraphML and return (nodes, edges)."""
    graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
    if not graph_file.exists():
        return [], []

    tree = ET.parse(graph_file)
    root = tree.getroot()

    # Build key_id → attr_name mapping
    key_map = {}
    for key in root.findall(f"{{{GRAPHML_NS}}}key"):
        key_map[key.attrib["id"]] = key.attrib.get("attr.name", key.attrib["id"])

    nodes = []
    edges = []
    graph = root.find(f"{{{GRAPHML_NS}}}graph")
    if graph is None:
        return [], []

    for elem in graph:
        data = {}
        for d in elem.findall(f"{{{GRAPHML_NS}}}data"):
            attr = key_map.get(d.attrib["key"], d.attrib["key"])
            data[attr] = d.text or ""
        if elem.tag == f"{{{GRAPHML_NS}}}node":
            data["_id"] = elem.attrib.get("id", "")
            nodes.append(data)
        elif elem.tag == f"{{{GRAPHML_NS}}}edge":
            data["_source"] = elem.attrib.get("source", "")
            data["_target"] = elem.attrib.get("target", "")
            edges.append(data)

    return nodes, edges


def analyze_graph(nodes: list[dict], edges: list[dict]) -> dict:
    """Compute graph quality metrics."""
    if not nodes:
        return {"error": "No nodes found"}

    type_counts = Counter(n.get("entity_type", "unknown") for n in nodes)

    # Anchoring score: entity name or description contains course id
    anchored = sum(
        1 for n in nodes
        if COURSE_ID.lower() in (n.get("_id", "") + " " + n.get("description", "")).lower()
    )

    # Strictly anchored: name itself contains course id
    name_anchored = sum(
        1 for n in nodes
        if COURSE_ID.lower() in n.get("_id", "").lower()
    )

    # Novel types
    known = {t.lower() for t in SYLLABUS_ENTITY_TYPES}
    novel_types = {
        n.get("entity_type", "")
        for n in nodes
        if n.get("entity_type", "").lower() not in known
        and n.get("entity_type", "") not in ("", "other", "Other")
    }
    other_types = sum(1 for n in nodes if n.get("entity_type", "").lower() == "other")

    # Description quality
    desc_lens = [len(n.get("description", "")) for n in nodes]
    avg_desc = sum(desc_lens) / len(desc_lens) if desc_lens else 0

    # Noise entities (very short name + short desc)
    noise = [
        n["_id"] for n in nodes
        if len(n.get("_id", "")) <= 5 and len(n.get("description", "")) < 60
    ]

    return {
        "total_entities": len(nodes),
        "total_relationships": len(edges),
        "type_distribution": dict(type_counts.most_common()),
        "anchoring_pct": round(100 * anchored / len(nodes), 1),
        "name_anchored_pct": round(100 * name_anchored / len(nodes), 1),
        "novel_types": sorted(novel_types),
        "other_type_count": other_types,
        "avg_description_len": round(avg_desc),
        "noise_entities": noise,
    }


def check_fact_coverage(storage_dir: Path) -> dict:
    """Check whether expected facts from source doc appear in the graph."""
    graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
    if not graph_file.exists():
        return {}

    # Build full graph text for searching
    graph_text = graph_file.read_text(encoding="utf-8").lower()

    # Also check kv_store for chunk text
    chunk_store = storage_dir / "kv_store_text_chunks.json"
    chunk_text = ""
    if chunk_store.exists():
        chunk_text = chunk_store.read_text(encoding="utf-8").lower()

    combined = graph_text + " " + chunk_text

    results = {}
    for pattern, label in EXPECTED_FACTS.items():
        found = bool(re.search(pattern.lower(), combined))
        results[label] = found

    found_count = sum(results.values())
    return {
        "coverage_pct": round(100 * found_count / len(results)),
        "found": found_count,
        "total": len(results),
        "details": results,
    }


async def test_query_precision(storage_dir: Path, top_k: int = 40, related_chunk_number: int = 5) -> list[dict]:
    """Run test queries and measure context token count."""
    from lightrag import QueryParam

    rag = await amake_lightrag_improved(working_dir=storage_dir, top_k=top_k, related_chunk_number=related_chunk_number)

    results = []
    for q in TEST_QUESTIONS:
        try:
            context = await rag.aquery(q, param=QueryParam(mode="hybrid", only_need_context=True))
            context_str = str(context) if context else ""
            # Rough token estimate: chars / 4
            tokens = len(context_str) // 4
            results.append({
                "question": q,
                "context_tokens": tokens,
                "context_preview": context_str[:300].replace("\n", " "),
                "has_content": len(context_str) > 50,
            })
        except Exception as e:
            results.append({"question": q, "error": str(e)[:120]})

    return results


def print_report(label: str, metrics: dict, coverage: dict, queries: list[dict] | None) -> None:
    print(f"\n{'=' * 60}")
    print(f"INSPECTION REPORT: {label}")
    print("=" * 60)

    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return

    print(f"\n[ENTITIES]")
    print(f"  Total entities:        {metrics['total_entities']}")
    print(f"  Total relationships:   {metrics['total_relationships']}")
    print(f"  Avg description len:   {metrics['avg_description_len']} chars")
    print(f"  Anchored (name+desc):  {metrics['anchoring_pct']}%")
    print(f"  Anchored (name only):  {metrics['name_anchored_pct']}%")
    print(f"  Classified as Other:   {metrics['other_type_count']}")

    print(f"\n[TYPE DISTRIBUTION]")
    for t, c in metrics["type_distribution"].items():
        marker = "  ✓" if t.lower() in {x.lower() for x in SYLLABUS_ENTITY_TYPES} else "  ?"
        print(f"{marker} {t}: {c}")

    if metrics["novel_types"]:
        print(f"\n[NOVEL TYPES INTRODUCED]")
        for t in metrics["novel_types"]:
            print(f"  + {t}")

    if metrics["noise_entities"]:
        print(f"\n[NOISE ENTITIES (short name + short desc)]")
        for e in metrics["noise_entities"]:
            print(f"  - '{e}'")

    if coverage:
        print(f"\n[FACT COVERAGE] {coverage['found']}/{coverage['total']} = {coverage['coverage_pct']}%")
        for label_k, found in coverage["details"].items():
            icon = "✓" if found else "✗"
            print(f"  {icon} {label_k}")

    if queries:
        print(f"\n[QUERY PRECISION]")
        for r in queries:
            if "error" in r:
                print(f"  ✗ {r['question'][:50]}... → ERROR: {r['error']}")
            else:
                status = "✓" if r["has_content"] else "✗ (empty)"
                print(f"  {status} ~{r['context_tokens']}t | {r['question'][:55]}")
                if r["context_preview"]:
                    print(f"       Preview: {r['context_preview'][:120]}")

    print()


async def main_async(storage_dir: Path, run_queries: bool, top_k: int = 40, related_chunk_number: int = 5) -> None:
    nodes, edges = parse_graphml(storage_dir)
    metrics = analyze_graph(nodes, edges)
    coverage = check_fact_coverage(storage_dir)
    queries = await test_query_precision(storage_dir, top_k=top_k, related_chunk_number=related_chunk_number) if run_queries else None
    label = storage_dir.name
    print_report(label, metrics, coverage, queries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CSCE 670 LightRAG graph quality")
    parser.add_argument("--storage-dir", required=True, help="Storage dir name under tools/lightrag_spike/")
    parser.add_argument("--no-queries", action="store_true", help="Skip query precision test (saves API calls)")
    parser.add_argument("--top-k", type=int, default=40, help="top_k for query retrieval (default: 40)")
    parser.add_argument("--related-chunks", type=int, default=5, help="Raw text chunks to include in context (default: 5)")
    args = parser.parse_args()

    storage_dir = SPIKE_DIR / args.storage_dir
    if not storage_dir.exists():
        print(f"Storage dir not found: {storage_dir}")
        sys.exit(1)

    asyncio.run(main_async(storage_dir, run_queries=not args.no_queries, top_k=args.top_k, related_chunk_number=args.related_chunks))


if __name__ == "__main__":
    main()
