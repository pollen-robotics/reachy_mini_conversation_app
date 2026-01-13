"""Unit tests for the move_head tool."""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.move_head import MoveHead
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestMoveHeadToolAttributes:
    """Tests for MoveHead tool attributes."""

    def test_move_head_has_correct_name(self) -> None:
        """Test MoveHead tool has correct name."""
        tool = MoveHead()
        assert tool.name == "move_head"

    def test_move_head_has_description(self) -> None:
        """Test MoveHead tool has description."""
        tool = MoveHead()
        assert "head" in tool.description.lower()
        assert "direction" in tool.description.lower()

    def test_move_head_has_parameters_schema(self) -> None:
        """Test MoveHead tool has correct parameters schema."""
        tool = MoveHead()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "direction" in schema["properties"]
        assert schema["properties"]["direction"]["type"] == "string"
        assert "enum" in schema["properties"]["direction"]
        assert "direction" in schema["required"]

    def test_move_head_direction_enum(self) -> None:
        """Test MoveHead has correct direction enum values."""
        tool = MoveHead()
        directions = tool.parameters_schema["properties"]["direction"]["enum"]

        assert "left" in directions
        assert "right" in directions
        assert "up" in directions
        assert "down" in directions
        assert "front" in directions

    def test_move_head_spec(self) -> None:
        """Test MoveHead tool spec generation."""
        tool = MoveHead()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "move_head"

    def test_move_head_deltas_defined(self) -> None:
        """Test MoveHead has DELTAS for all directions."""
        tool = MoveHead()

        assert "left" in tool.DELTAS
        assert "right" in tool.DELTAS
        assert "up" in tool.DELTAS
        assert "down" in tool.DELTAS
        assert "front" in tool.DELTAS

        # Each delta should be a tuple of 6 values
        for direction, delta in tool.DELTAS.items():
            assert len(delta) == 6, f"Delta for {direction} should have 6 values"


class TestMoveHeadToolExecution:
    """Tests for MoveHead tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        mock_reachy = MagicMock()
        mock_reachy.get_current_head_pose.return_value = MagicMock()
        mock_reachy.get_current_joint_positions.return_value = (MagicMock(), [0.0, 0.0, 0.0])

        return ToolDependencies(
            reachy_mini=mock_reachy,
            movement_manager=MagicMock(),
            motion_duration_s=1.0,
        )

    @pytest.mark.asyncio
    async def test_move_head_invalid_direction_type(self) -> None:
        """Test move_head returns error for non-string direction."""
        deps = ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )
        tool = MoveHead()

        result = await tool(deps, direction=123)

        assert "error" in result
        assert "string" in result["error"]

    @pytest.mark.asyncio
    async def test_move_head_none_direction(self) -> None:
        """Test move_head returns error for None direction."""
        deps = ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )
        tool = MoveHead()

        result = await tool(deps, direction=None)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_move_head_left(self, mock_deps: ToolDependencies) -> None:
        """Test move_head left direction."""
        tool = MoveHead()

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            mock_create.return_value = MagicMock()

            result = await tool(mock_deps, direction="left")

        assert result["status"] == "looking left"
        mock_deps.movement_manager.queue_move.assert_called_once()
        mock_deps.movement_manager.set_moving_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_head_right(self, mock_deps: ToolDependencies) -> None:
        """Test move_head right direction."""
        tool = MoveHead()

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            mock_create.return_value = MagicMock()

            result = await tool(mock_deps, direction="right")

        assert result["status"] == "looking right"

    @pytest.mark.asyncio
    async def test_move_head_up(self, mock_deps: ToolDependencies) -> None:
        """Test move_head up direction."""
        tool = MoveHead()

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            mock_create.return_value = MagicMock()

            result = await tool(mock_deps, direction="up")

        assert result["status"] == "looking up"

    @pytest.mark.asyncio
    async def test_move_head_down(self, mock_deps: ToolDependencies) -> None:
        """Test move_head down direction."""
        tool = MoveHead()

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            mock_create.return_value = MagicMock()

            result = await tool(mock_deps, direction="down")

        assert result["status"] == "looking down"

    @pytest.mark.asyncio
    async def test_move_head_front(self, mock_deps: ToolDependencies) -> None:
        """Test move_head front direction."""
        tool = MoveHead()

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            mock_create.return_value = MagicMock()

            result = await tool(mock_deps, direction="front")

        assert result["status"] == "looking front"

    @pytest.mark.asyncio
    async def test_move_head_unknown_direction_returns_error(self, mock_deps: ToolDependencies) -> None:
        """Test move_head with unknown direction returns an error."""
        tool = MoveHead()

        result = await tool(mock_deps, direction="unknown_direction")

        # Unknown directions return an error with valid directions listed
        assert "error" in result
        assert "direction must be one of" in result["error"]

    @pytest.mark.asyncio
    async def test_move_head_queues_goto_move(self, mock_deps: ToolDependencies) -> None:
        """Test move_head queues a GotoQueueMove."""
        tool = MoveHead()

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            with patch("reachy_mini_conversation_app.tools.move_head.GotoQueueMove") as mock_goto:
                mock_create.return_value = MagicMock()
                mock_goto_instance = MagicMock()
                mock_goto.return_value = mock_goto_instance

                await tool(mock_deps, direction="left")

        mock_goto.assert_called_once()
        mock_deps.movement_manager.queue_move.assert_called_once_with(mock_goto_instance)

    @pytest.mark.asyncio
    async def test_move_head_sets_moving_state(self, mock_deps: ToolDependencies) -> None:
        """Test move_head sets moving state with duration."""
        tool = MoveHead()
        mock_deps.motion_duration_s = 2.5

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            mock_create.return_value = MagicMock()

            await tool(mock_deps, direction="left")

        mock_deps.movement_manager.set_moving_state.assert_called_once_with(2.5)

    @pytest.mark.asyncio
    async def test_move_head_handles_exception(self, mock_deps: ToolDependencies) -> None:
        """Test move_head handles exceptions gracefully."""
        tool = MoveHead()
        mock_deps.movement_manager.queue_move.side_effect = RuntimeError("Test error")

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            mock_create.return_value = MagicMock()

            result = await tool(mock_deps, direction="left")

        assert "error" in result
        assert "RuntimeError" in result["error"]
        assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_move_head_uses_current_pose(self, mock_deps: ToolDependencies) -> None:
        """Test move_head gets current pose for interpolation."""
        tool = MoveHead()

        with patch("reachy_mini_conversation_app.tools.move_head.create_head_pose") as mock_create:
            mock_create.return_value = MagicMock()

            await tool(mock_deps, direction="left")

        cast(MagicMock, mock_deps.reachy_mini.get_current_head_pose).assert_called_once()
        cast(MagicMock, mock_deps.reachy_mini.get_current_joint_positions).assert_called_once()
