"""Unit tests for the stop_dance tool."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.stop_dance import StopDance
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestStopDanceToolAttributes:
    """Tests for StopDance tool attributes."""

    def test_stop_dance_has_correct_name(self) -> None:
        """Test StopDance tool has correct name."""
        tool = StopDance()
        assert tool.name == "stop_dance"

    def test_stop_dance_has_description(self) -> None:
        """Test StopDance tool has description."""
        tool = StopDance()
        assert "stop" in tool.description.lower()
        assert "dance" in tool.description.lower()

    def test_stop_dance_has_parameters_schema(self) -> None:
        """Test StopDance tool has correct parameters schema."""
        tool = StopDance()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "dummy" in schema["properties"]
        assert schema["properties"]["dummy"]["type"] == "boolean"
        assert "dummy" in schema["required"]

    def test_stop_dance_spec(self) -> None:
        """Test StopDance tool spec generation."""
        tool = StopDance()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "stop_dance"


class TestStopDanceToolExecution:
    """Tests for StopDance tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_stop_dance_clears_queue(self, mock_deps: ToolDependencies) -> None:
        """Test stop_dance clears the move queue."""
        tool = StopDance()

        result = await tool(mock_deps, dummy=True)

        assert result["status"] == "stopped dance and cleared queue"
        mock_deps.movement_manager.clear_move_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_dance_works_without_dummy(self, mock_deps: ToolDependencies) -> None:
        """Test stop_dance works even without dummy parameter."""
        tool = StopDance()

        # Should still work even if dummy is not provided
        result = await tool(mock_deps)

        assert "status" in result
        mock_deps.movement_manager.clear_move_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_dance_returns_status(self, mock_deps: ToolDependencies) -> None:
        """Test stop_dance returns proper status."""
        tool = StopDance()

        result = await tool(mock_deps, dummy=True)

        assert "status" in result
        assert "stopped" in result["status"]
        assert "cleared" in result["status"]
