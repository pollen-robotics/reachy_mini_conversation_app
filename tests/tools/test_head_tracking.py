"""Unit tests for the head_tracking tool."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.head_tracking import HeadTracking


class TestHeadTrackingToolAttributes:
    """Tests for HeadTracking tool attributes."""

    def test_head_tracking_has_correct_name(self) -> None:
        """Test HeadTracking tool has correct name."""
        tool = HeadTracking()
        assert tool.name == "head_tracking"

    def test_head_tracking_has_description(self) -> None:
        """Test HeadTracking tool has description."""
        tool = HeadTracking()
        assert "head" in tool.description.lower()
        assert "tracking" in tool.description.lower()

    def test_head_tracking_has_parameters_schema(self) -> None:
        """Test HeadTracking tool has correct parameters schema."""
        tool = HeadTracking()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "start" in schema["properties"]
        assert schema["properties"]["start"]["type"] == "boolean"
        assert "start" in schema["required"]

    def test_head_tracking_spec(self) -> None:
        """Test HeadTracking tool spec generation."""
        tool = HeadTracking()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "head_tracking"


class TestHeadTrackingToolExecution:
    """Tests for HeadTracking tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
            camera_worker=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_head_tracking_start(self, mock_deps: ToolDependencies) -> None:
        """Test head_tracking starts tracking."""
        tool = HeadTracking()

        result = await tool(mock_deps, start=True)

        assert result["status"] == "head tracking started"
        mock_deps.camera_worker.set_head_tracking_enabled.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_head_tracking_stop(self, mock_deps: ToolDependencies) -> None:
        """Test head_tracking stops tracking."""
        tool = HeadTracking()

        result = await tool(mock_deps, start=False)

        assert result["status"] == "head tracking stopped"
        mock_deps.camera_worker.set_head_tracking_enabled.assert_called_once_with(False)

    @pytest.mark.asyncio
    async def test_head_tracking_no_camera_worker(self) -> None:
        """Test head_tracking works without camera worker."""
        deps = ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
            camera_worker=None,
        )
        tool = HeadTracking()

        result = await tool(deps, start=True)

        assert result["status"] == "head tracking started"

    @pytest.mark.asyncio
    async def test_head_tracking_converts_to_bool(self, mock_deps: ToolDependencies) -> None:
        """Test head_tracking converts start parameter to bool."""
        tool = HeadTracking()

        # Test with truthy value
        result = await tool(mock_deps, start=1)
        assert result["status"] == "head tracking started"

        # Reset mock
        mock_deps.camera_worker.set_head_tracking_enabled.reset_mock()

        # Test with falsy value
        result = await tool(mock_deps, start=0)
        assert result["status"] == "head tracking stopped"

    @pytest.mark.asyncio
    async def test_head_tracking_missing_start_defaults_to_false(self, mock_deps: ToolDependencies) -> None:
        """Test head_tracking with missing start parameter defaults to False."""
        tool = HeadTracking()

        result = await tool(mock_deps)

        assert result["status"] == "head tracking stopped"
        mock_deps.camera_worker.set_head_tracking_enabled.assert_called_once_with(False)
