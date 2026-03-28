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
        history_messages: list = [],
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        stream = await async_client.chat.completions.create(
            model=config.TAMU_MODEL,
            messages=messages,
            max_tokens=4096,
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

    async def voyage_embed(texts: list[str]) -> np.ndarray:
        client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
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
    """Return a fully configured LightRAG instance.

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
