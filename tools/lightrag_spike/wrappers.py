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
    2. Add anchoring rule so entity names include course ID when course-specific
    """
    from lightrag.prompt import PROMPTS

    system = PROMPTS["entity_extraction_system_prompt"]

    # Allow novel types (replace hard 'Other' fallback)
    system = system.replace(
        "If none of the provided entity types apply, do not add new entity type and classify it as `Other`.",
        "If none of the provided entity types apply, introduce a new descriptive entity type that best captures the nature of the entity.",
    )

    # Add anchoring rule after the entity_type instruction line
    anchoring_rule = (
        "\n    *   **Course Anchoring:** When an entity is specific to a particular course "
        "(e.g., a policy, assignment, or schedule item), include the course identifier "
        "(e.g., 'CSCE 670') in the entity name to prevent cross-course confusion. "
        "For example, use 'CSCE 670 Late Work Policy' rather than 'Late Work Policy', "
        "and 'CSCE 670 Homework 1' rather than 'Homework 1'."
    )
    # Insert after the entity_description bullet
    marker = "*   `entity_description`: Provide a concise yet comprehensive description"
    system = system.replace(marker, anchoring_rule + "\n    " + marker)

    PROMPTS["entity_extraction_system_prompt"] = system


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
