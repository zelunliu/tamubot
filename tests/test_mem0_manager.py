"""Tests for Mem0Manager backed by cloud API."""
import time
from unittest.mock import MagicMock, patch

from rag.tools.mem0 import Mem0Manager


def test_mem0_manager_add_turn_calls_cloud_api():
    """add_turn should call MemoryClient.add with correct user_id."""
    with patch("rag.tools.mem0.MemoryClient") as MockClient, \
         patch("config.MEM0_API_KEY", "test-api-key"):
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        manager = Mem0Manager("session-123")
        manager.add_turn("hello", "hi there")

        mock_instance.add.assert_called_once()
        call_kwargs = mock_instance.add.call_args[1]
        assert call_kwargs["user_id"] == "session-123"

        call_messages = mock_instance.add.call_args[0][0]
        assert call_messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]


def test_mem0_manager_search_context_returns_facts():
    """search_context should format MemoryClient.search results as bullet list."""
    with patch("rag.tools.mem0.MemoryClient") as MockClient, \
         patch("config.MEM0_API_KEY", "test-api-key"):
        mock_instance = MagicMock()
        mock_instance.search.return_value = [
            {"memory": "user is a CS senior"},
            {"memory": "prefers online courses"},
        ]
        MockClient.return_value = mock_instance

        manager = Mem0Manager("session-123")
        result = manager.search_context("what courses should I take", top_k=3)

        assert "CS senior" in result
        assert "online" in result
        # Should be bullet-formatted
        assert result.startswith("-")


def test_mem0_manager_search_context_empty_results():
    """search_context should return '' when MemoryClient.search returns []."""
    with patch("rag.tools.mem0.MemoryClient") as MockClient, \
         patch("config.MEM0_API_KEY", "test-api-key"):
        mock_instance = MagicMock()
        mock_instance.search.return_value = []
        MockClient.return_value = mock_instance

        manager = Mem0Manager("session-empty")
        result = manager.search_context("anything", top_k=3)

        assert result == ""


def test_mem0_manager_search_context_exception_swallowed():
    """search_context should return '' (not raise) when MemoryClient.search raises."""
    with patch("rag.tools.mem0.MemoryClient") as MockClient, \
         patch("config.MEM0_API_KEY", "test-api-key"):
        mock_instance = MagicMock()
        mock_instance.search.side_effect = Exception("timeout")
        MockClient.return_value = mock_instance

        manager = Mem0Manager("session-err")
        result = manager.search_context("anything", top_k=3)

        assert result == ""


def test_mem0_manager_add_turn_async_fires_thread():
    with patch("rag.tools.mem0.MemoryClient") as MockClient, \
         patch("config.MEM0_API_KEY", "test-api-key"):
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        manager = Mem0Manager("session-async")
        # Call async and give the daemon thread time to complete
        manager.add_turn_async("user msg", "assistant msg")
        time.sleep(0.1)

        mock_instance.add.assert_called_once()
