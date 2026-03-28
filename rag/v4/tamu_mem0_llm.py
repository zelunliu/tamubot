"""Custom mem0 LLM adapter for the TAMU AI gateway.

The TAMU gateway always returns SSE. This adapter wraps call_llm()
which handles the streaming internally, presenting a blocking interface to mem0.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase


class TamuMem0LLM(LLMBase):
    """mem0 LLM adapter backed by the TAMU AI gateway via call_llm()."""

    def __init__(self, config: Optional[BaseLlmConfig] = None):
        if config is None:
            config = BaseLlmConfig()
        super().__init__(config)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> str:
        """Generate a response using the TAMU AI gateway."""
        from rag.llm_client import call_llm
        result = call_llm(messages, temperature=0.1, max_tokens=4096)
        return result.text
