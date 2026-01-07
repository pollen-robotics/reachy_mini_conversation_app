"""Unit tests for the stop_emotion tool."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.stop_emotion import StopEmotion
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestStopEmotionToolAttributes:
    """Tests for StopEmotion tool attributes."""

    def test_stop_emotion_has_correct_name(self) -> None:
        """Test StopEmotion tool has correct name."""
        tool = StopEmotion()
        assert tool.name == "stop_emotion"

    def test_stop_emotion_has_description(self) -> None:
        """Test StopEmotion tool has description."""
        tool = StopEmotion()
        assert "emotion" in tool.description.lower()
        assert "stop" in tool.description.lower()

    def test_stop_emotion_has_parameters_schema(self) -> None:
        """Test StopEmotion tool has correct parameters schema."""
        tool = StopEmotion()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "dummy" in schema["properties"]
        assert schema["properties"]["dummy"]["type"] == "boolean"
        assert "dummy" in schema["required"]

    def test_stop_emotion_spec(self) -> None:
        """Test StopEmotion tool spec generation."""
        tool = StopEmotion()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "stop_emotion"


class TestStopEmotionToolExecution:
    """Tests for StopEmotion tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_stop_emotion_clears_queue(self, mock_deps: ToolDependencies) -> None:
        """Test stop_emotion clears the move queue."""
        tool = StopEmotion()

        result = await tool(mock_deps, dummy=True)

        assert result["status"] == "stopped emotion and cleared queue"
        mock_deps.movement_manager.clear_move_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_emotion_works_without_dummy(self, mock_deps: ToolDependencies) -> None:
        """Test stop_emotion works even without dummy parameter."""
        tool = StopEmotion()

        result = await tool(mock_deps)

        assert result["status"] == "stopped emotion and cleared queue"
        mock_deps.movement_manager.clear_move_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_emotion_returns_status(self, mock_deps: ToolDependencies) -> None:
        """Test stop_emotion returns proper status."""
        tool = StopEmotion()

        result = await tool(mock_deps, dummy=True)

        assert "status" in result
        assert "stopped" in result["status"]
        assert "cleared" in result["status"]
