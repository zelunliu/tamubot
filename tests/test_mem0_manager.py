"""Tests for Mem0Manager backed by cloud API."""
from unittest.mock import MagicMock, patch


def test_mem0_manager_add_turn_calls_cloud_api():
    """add_turn should call MemoryClient.add with correct user_id."""
    with patch("rag.tools.mem0.MemoryClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from rag.tools.mem0 import Mem0Manager
        manager = Mem0Manager("session-123")
        manager.add_turn("hello", "hi there")

        mock_instance.add.assert_called_once()
        call_kwargs = mock_instance.add.call_args[1]
        assert call_kwargs["user_id"] == "session-123"


def test_mem0_manager_search_context_returns_facts():
    """search_context should format MemoryClient.search results as bullet list."""
    with patch("rag.tools.mem0.MemoryClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.search.return_value = [
            {"memory": "user is a CS senior"},
            {"memory": "prefers online courses"},
        ]
        MockClient.return_value = mock_instance

        from rag.tools.mem0 import Mem0Manager
        manager = Mem0Manager("session-123")
        result = manager.search_context("what courses should I take", top_k=3)

        assert "CS senior" in result
        assert "online" in result
        # Should be bullet-formatted
        assert result.startswith("-")
