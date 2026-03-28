"""Custom mem0 embedder backed by Voyage AI (voyage-3, 1024 dims)."""
from __future__ import annotations

from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class VoyageMem0Embedder(EmbeddingBase):
    """mem0 embedder backed by Voyage AI voyage-3 model (1024-dim embeddings)."""

    DIMS = 1024

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        if config is None:
            config = BaseEmbedderConfig(embedding_dims=self.DIMS)
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import voyageai

            import config as app_config
            self._client = voyageai.Client(api_key=app_config.VOYAGE_API_KEY)
        return self._client

    def embed(
        self,
        text: str,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> list:
        """Embed text using Voyage AI voyage-3 model."""
        client = self._get_client()
        result = client.embed([text], model="voyage-3", input_type="query")
        return result.embeddings[0]
