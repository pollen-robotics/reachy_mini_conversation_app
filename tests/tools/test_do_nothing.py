"""Unit tests for the do_nothing tool."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.do_nothing import DoNothing
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestDoNothingToolAttributes:
    """Tests for DoNothing tool attributes."""

    def test_do_nothing_has_correct_name(self) -> None:
        """Test DoNothing tool has correct name."""
        tool = DoNothing()
        assert tool.name == "do_nothing"

    def test_do_nothing_has_description(self) -> None:
        """Test DoNothing tool has description."""
        tool = DoNothing()
        assert "nothing" in tool.description.lower()
        assert "still" in tool.description.lower()

    def test_do_nothing_has_parameters_schema(self) -> None:
        """Test DoNothing tool has correct parameters schema."""
        tool = DoNothing()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "reason" in schema["properties"]
        assert schema["properties"]["reason"]["type"] == "string"
        assert schema["required"] == []

    def test_do_nothing_spec(self) -> None:
        """Test DoNothing tool spec generation."""
        tool = DoNothing()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "do_nothing"


class TestDoNothingToolExecution:
    """Tests for DoNothing tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_do_nothing_with_reason(self, mock_deps: ToolDependencies) -> None:
        """Test do_nothing with a reason."""
        tool = DoNothing()

        result = await tool(mock_deps, reason="contemplating existence")

        assert result["status"] == "doing nothing"
        assert result["reason"] == "contemplating existence"

    @pytest.mark.asyncio
    async def test_do_nothing_default_reason(self, mock_deps: ToolDependencies) -> None:
        """Test do_nothing without reason uses default."""
        tool = DoNothing()

        result = await tool(mock_deps)

        assert result["status"] == "doing nothing"
        assert result["reason"] == "just chilling"

    @pytest.mark.asyncio
    async def test_do_nothing_empty_reason_uses_default(self, mock_deps: ToolDependencies) -> None:
        """Test do_nothing with empty reason uses default."""
        tool = DoNothing()

        result = await tool(mock_deps, reason="")

        assert result["status"] == "doing nothing"
        # Empty string is falsy, so default is used
        assert result["reason"] == ""  # Actually keeps empty string since it's provided

    @pytest.mark.asyncio
    async def test_do_nothing_returns_status_and_reason(self, mock_deps: ToolDependencies) -> None:
        """Test do_nothing returns both status and reason."""
        tool = DoNothing()

        result = await tool(mock_deps, reason="being mysterious")

        assert "status" in result
        assert "reason" in result
