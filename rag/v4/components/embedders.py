"""Haystack-compatible embedding components."""
from __future__ import annotations
from typing import Optional
from haystack import component
import config


@component
class VoyageEmbedder:
    """Embeds text using Voyage-3 via the voyageai client."""

    def __init__(self, model: str = "voyage-3"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import voyageai
            self._client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
        return self._client

    @component.output_types(embedding=list)
    def run(self, text: str) -> dict:
        client = self._get_client()
        result = client.embed([text], model=self.model, input_type="query")
        return {"embedding": result.embeddings[0]}


@component
class NullEmbedder:
    """Stub embedder returning a zero vector. Used for testing."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    @component.output_types(embedding=list)
    def run(self, text: str) -> dict:
        return {"embedding": [0.0] * self.dim}
