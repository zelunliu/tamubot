"""Shared LightRAG setup: TAMU gateway LLM wrapper + Voyage-3 embedder wrapper."""

import asyncio
import sys
from pathlib import Path

import numpy as np

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

WORKING_DIR = Path(__file__).parent / "storage"
EMBED_DIM = 1024  # Voyage-3

# Domain-specific entity types for academic course syllabi.
# Treated as preferred/suggested types, not a hard whitelist.
SYLLABUS_ENTITY_TYPES = [
    "Course",
    "Instructor",
    "CourseTopic",
    "Assignment",
    "GradeComponent",
    "AcademicPolicy",
    "Prerequisite",
    "Textbook",
    "Exam",
    "LearningOutcome",
    "ScheduleEvent",
]


def patch_prompts_for_syllabi() -> None:
    """Patch LightRAG extraction prompts for syllabus-specific extraction.

    Changes from defaults:
    1. Allow novel entity types instead of forcing 'Other'
    2. Extended anchoring rule — CourseTopic entities now included
    3. Suppress generic 'The Course' entity
    4. Enforce Title Case for entity names to prevent case-drift duplicates
    """
    from lightrag.prompt import PROMPTS

    system = PROMPTS["entity_extraction_system_prompt"]

    # Allow novel types (replace hard 'Other' fallback)
    system = system.replace(
        "If none of the provided entity types apply, do not add new entity type and classify it as `Other`.",
        "If none of the provided entity types apply, introduce a new descriptive entity type that best captures the nature of the entity.",
    )

    # Three rules inserted after the entity_description bullet
    rules = (
        "\n    *   **Course Anchoring:** When an entity is specific to a particular course — "
        "including CourseTopic entities (weekly topics, lecture themes, subject areas covered), "
        "policies, assignments, exams, and schedule items — include the course identifier "
        "(e.g., 'CSCE 670') in the entity name to prevent cross-course confusion. "
        "Use 'CSCE 670 Boolean Retrieval' not 'Boolean Retrieval', "
        "'CSCE 670 Late Work Policy' not 'Late Work Policy', "
        "'CSCE 670 Homework 1' not 'Homework 1'."
        "\n    *   **No Generic Course Entity:** Never extract 'The Course' or 'This Course' as an "
        "entity name. If the entity refers to the course itself, use the specific course identifier "
        "(e.g., 'CSCE 670') instead."
        "\n    *   **Consistent Title Case:** Always use Title Case for entity names "
        "(capitalize the first letter of each major word, e.g., 'Information Retrieval', "
        "not 'information retrieval' or 'INFORMATION RETRIEVAL') to prevent duplicate nodes "
        "from capitalization variations."
    )
    marker = "*   `entity_description`: Provide a concise yet comprehensive description"
    system = system.replace(marker, rules + "\n    " + marker)

    PROMPTS["entity_extraction_system_prompt"] = system


def normalize_entity_names(storage_dir: Path) -> dict:
    """Post-process graphml to merge case-insensitive duplicate entity names.

    Finds pairs of nodes whose names differ only by case. Keeps the node with
    more edges, redirects all edges from discarded nodes to the kept node,
    removes discarded nodes. Writes cleaned graphml back (backs up original).

    Note: Only modifies graph_chunk_entity_relation.graphml. Vector indices in
    the .json embedding files are not updated — this fixes graph traversal quality
    but does not affect embedding-based lookups.

    Returns: {"merged": N, "kept": [...], "removed": [...]}
    """
    import shutil
    import xml.etree.ElementTree as ET

    graph_file = storage_dir / "graph_chunk_entity_relation.graphml"
    if not graph_file.exists():
        return {"merged": 0, "kept": [], "removed": []}

    NS = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", NS)
    tree = ET.parse(graph_file)
    root = tree.getroot()
    graph = root.find(f"{{{NS}}}graph")
    if graph is None:
        return {"merged": 0, "kept": [], "removed": []}

    # Count edges per node
    edge_count: dict[str, int] = {}
    for elem in graph:
        if elem.tag == f"{{{NS}}}edge":
            for attr in ("source", "target"):
                nid = elem.attrib.get(attr, "")
                edge_count[nid] = edge_count.get(nid, 0) + 1

    # Group node IDs by lowercase
    lower_to_ids: dict[str, list[str]] = {}
    for elem in graph:
        if elem.tag == f"{{{NS}}}node":
            nid = elem.attrib.get("id", "")
            lower_to_ids.setdefault(nid.lower(), []).append(nid)

    # Build merge map: discard → keep (keep = highest degree)
    merge_map: dict[str, str] = {}
    kept: list[str] = []
    removed: list[str] = []
    for ids in lower_to_ids.values():
        if len(ids) > 1:
            ids.sort(key=lambda i: edge_count.get(i, 0), reverse=True)
            kept.append(ids[0])
            for discard in ids[1:]:
                merge_map[discard] = ids[0]
                removed.append(discard)

    if not merge_map:
        return {"merged": 0, "kept": [], "removed": []}

    discard_set = set(merge_map)
    to_remove: list = []
    for elem in graph:
        if elem.tag == f"{{{NS}}}edge":
            src = elem.attrib.get("source", "")
            tgt = elem.attrib.get("target", "")
            new_src = merge_map.get(src, src)
            new_tgt = merge_map.get(tgt, tgt)
            if new_src == new_tgt:
                to_remove.append(elem)  # self-loop after merge
            else:
                elem.set("source", new_src)
                elem.set("target", new_tgt)
        elif elem.tag == f"{{{NS}}}node":
            if elem.attrib.get("id", "") in discard_set:
                to_remove.append(elem)

    for elem in to_remove:
        graph.remove(elem)

    shutil.copy(graph_file, graph_file.with_suffix(".graphml.bak"))
    tree.write(str(graph_file), encoding="unicode", xml_declaration=True)
    return {"merged": len(merge_map), "kept": kept, "removed": removed}


def make_tamu_llm_func():
    """Return an async LLM callable using the TAMU OpenAI-compatible gateway.

    TAMU gateway always returns SSE regardless of stream param, so we must
    use stream=True and join chunks. max_tokens must be >= 4096.
    """
    from openai import AsyncOpenAI

    async_client = AsyncOpenAI(
        api_key=config.TAMU_API_KEY,
        base_url=config.TAMU_BASE_URL,
    )

    async def tamu_llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in (history_messages or []):
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        stream = await async_client.chat.completions.create(
            model=config.TAMU_MODEL,
            messages=messages,
            max_tokens=8192,
            stream=True,
        )
        chunks = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
        return "".join(chunks)

    return tamu_llm_func


def make_voyage_embed_func() -> EmbeddingFunc:
    """Return an EmbeddingFunc using Voyage-3 (same model as the v4 pipeline)."""
    import voyageai

    client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    async def voyage_embed(texts: list[str]) -> np.ndarray:
        result = await asyncio.to_thread(
            client.embed, texts, model="voyage-3", input_type="document"
        )
        return np.array(result.embeddings)

    return EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=8192,
        func=voyage_embed,
    )


def make_lightrag(working_dir: Path | None = None) -> LightRAG:
    """Return a fully configured LightRAG instance (original defaults).

    Args:
        working_dir: Directory for graph/vector/KV storage.
                     Defaults to tools/lightrag_spike/storage/.
    """
    wd = working_dir or WORKING_DIR
    wd.mkdir(parents=True, exist_ok=True)

    return LightRAG(
        working_dir=str(wd),
        llm_model_func=make_tamu_llm_func(),
        embedding_func=make_voyage_embed_func(),
    )


def make_lightrag_improved(
    working_dir: Path | None = None,
    entity_types: list[str] | None = None,
    chunk_token_size: int = 500,
    chunk_overlap_token_size: int = 100,
    entity_extract_max_gleaning: int = 2,
    patch_prompts: bool = True,
    top_k: int = 15,
    related_chunk_number: int = 2,
) -> LightRAG:
    """Return a configured LightRAG instance with syllabus-optimised settings.

    Defaults derived from 5-iteration empirical tuning on CSCE 670:
      - chunk_token_size=500: splits ~1600-token syllabi into 5 chunks,
        each extractable within 8192-token output budget (smaller = redundant entities)
      - entity_extract_max_gleaning=2: second pass catches relationships in
        dense sections (e.g. course schedule)
      - top_k=15: reduces graph fan-out; in multi-course graphs this ensures
        precision — only anchored course-specific nodes are retrieved
      - related_chunk_number=2: limits raw text included in context

    Args:
        working_dir: Storage directory.
        entity_types: Preferred entity types. Defaults to SYLLABUS_ENTITY_TYPES.
        chunk_token_size: Tokens per internal chunk.
        chunk_overlap_token_size: Overlap between chunks.
        entity_extract_max_gleaning: Extra extraction passes per chunk.
        patch_prompts: If True, patches PROMPTS for novel types + anchoring.
        top_k: Seed nodes for graph traversal per query.
        related_chunk_number: Raw text chunks to include in query context.
    """
    if patch_prompts:
        patch_prompts_for_syllabi()

    wd = working_dir or WORKING_DIR
    wd.mkdir(parents=True, exist_ok=True)

    types = entity_types or SYLLABUS_ENTITY_TYPES

    return LightRAG(
        working_dir=str(wd),
        llm_model_func=make_tamu_llm_func(),
        embedding_func=make_voyage_embed_func(),
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
        entity_extract_max_gleaning=entity_extract_max_gleaning,
        top_k=top_k,
        related_chunk_number=related_chunk_number,
        addon_params={"entity_types": types, "language": "English"},
    )


async def amake_lightrag(working_dir: Path | None = None) -> LightRAG:
    """Return an initialized LightRAG instance (calls initialize_storages)."""
    rag = make_lightrag(working_dir)
    await rag.initialize_storages()
    return rag


async def amake_lightrag_improved(
    working_dir: Path | None = None,
    **kwargs,
) -> LightRAG:
    """Return an initialized improved LightRAG instance."""
    rag = make_lightrag_improved(working_dir, **kwargs)
    await rag.initialize_storages()
    return rag
