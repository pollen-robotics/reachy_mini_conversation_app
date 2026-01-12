"""Unit tests for the background_demo tool."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.example.background_demo import BackgroundDemoTool


class TestBackgroundDemoToolAttributes:
    """Tests for BackgroundDemoTool tool attributes."""

    def test_background_demo_has_correct_name(self) -> None:
        """Test BackgroundDemoTool tool has correct name."""
        tool = BackgroundDemoTool()
        assert tool.name == "background_demo"

    def test_background_demo_has_description(self) -> None:
        """Test BackgroundDemoTool tool has description."""
        tool = BackgroundDemoTool()
        assert "demo" in tool.description.lower()
        assert "background" in tool.description.lower()

    def test_background_demo_has_parameters_schema(self) -> None:
        """Test BackgroundDemoTool tool has correct parameters schema."""
        tool = BackgroundDemoTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "duration" in schema["properties"]
        assert "background" in schema["properties"]
        assert "with_progress" in schema["properties"]
        assert schema["properties"]["duration"]["type"] == "integer"
        assert schema["properties"]["background"]["type"] == "boolean"
        assert schema["properties"]["with_progress"]["type"] == "boolean"

    def test_background_demo_spec(self) -> None:
        """Test BackgroundDemoTool tool spec generation."""
        tool = BackgroundDemoTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "background_demo"


class TestBackgroundDemoToolExecution:
    """Tests for BackgroundDemoTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_background_demo_synchronous(self, mock_deps: ToolDependencies) -> None:
        """Test background_demo runs synchronously when background=False."""
        tool = BackgroundDemoTool()

        mock_run = AsyncMock(return_value={"status": "success", "duration": 1, "message": "Done"})
        with patch.object(tool, "_run_demo", mock_run):
            result = await tool(mock_deps, duration=1, background=False)

        assert result["status"] == "success"
        mock_run.assert_called_once_with(1, with_progress=False)

    @pytest.mark.asyncio
    async def test_background_demo_async(self, mock_deps: ToolDependencies) -> None:
        """Test background_demo runs in background when background=True."""
        tool = BackgroundDemoTool()

        mock_task = MagicMock()
        mock_task.id = "test-task-123"

        mock_manager = MagicMock()

        # Capture and close the coroutine passed to start_task to avoid warning
        async def capture_start_task(**kwargs: Any) -> MagicMock:
            coro = kwargs.get("coroutine")
            if coro is not None:
                coro.close()  # Close the coroutine to prevent warning
            return mock_task

        mock_manager.start_task = AsyncMock(side_effect=capture_start_task)

        with patch(
            "reachy_mini_conversation_app.profiles.example.background_demo.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, duration=5, background=True)

        assert result["status"] == "started"
        assert result["task_id"] == "test-task-123"
        assert result["duration"] == 5
        mock_manager.start_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_background_demo_default_duration(self, mock_deps: ToolDependencies) -> None:
        """Test background_demo uses default duration of 5."""
        tool = BackgroundDemoTool()

        mock_run = AsyncMock(return_value={"status": "success"})
        with patch.object(tool, "_run_demo", mock_run):
            await tool(mock_deps, background=False)

        mock_run.assert_called_once_with(5, with_progress=False)

    @pytest.mark.asyncio
    async def test_background_demo_invalid_duration_uses_default(self, mock_deps: ToolDependencies) -> None:
        """Test background_demo uses default for invalid duration."""
        tool = BackgroundDemoTool()

        mock_run = AsyncMock(return_value={"status": "success"})
        with patch.object(tool, "_run_demo", mock_run):
            await tool(mock_deps, duration=-1, background=False)

        mock_run.assert_called_once_with(5, with_progress=False)

    @pytest.mark.asyncio
    async def test_background_demo_caps_duration_at_60(self, mock_deps: ToolDependencies) -> None:
        """Test background_demo caps duration at 60 seconds."""
        tool = BackgroundDemoTool()

        mock_run = AsyncMock(return_value={"status": "success"})
        with patch.object(tool, "_run_demo", mock_run):
            await tool(mock_deps, duration=120, background=False)

        mock_run.assert_called_once_with(60, with_progress=False)

    @pytest.mark.asyncio
    async def test_background_demo_with_progress(self, mock_deps: ToolDependencies) -> None:
        """Test background_demo with progress tracking."""
        tool = BackgroundDemoTool()

        mock_task = MagicMock()
        mock_task.id = "test-task-456"

        mock_manager = MagicMock()

        # Capture and close the coroutine passed to start_task to avoid warning
        captured_kwargs: dict[str, Any] = {}

        async def capture_start_task(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            coro = kwargs.get("coroutine")
            if coro is not None:
                coro.close()
            return mock_task

        mock_manager.start_task = AsyncMock(side_effect=capture_start_task)

        with patch(
            "reachy_mini_conversation_app.profiles.example.background_demo.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, duration=3, background=True, with_progress=True)

        assert result["with_progress"] is True
        mock_manager.start_task.assert_called_once()
        assert captured_kwargs["with_progress"] is True


class TestBackgroundDemoRunDemo:
    """Tests for _run_demo method."""

    @pytest.mark.asyncio
    async def test_run_demo_without_progress(self) -> None:
        """Test _run_demo without progress tracking."""
        tool = BackgroundDemoTool()

        mock_sleep = AsyncMock()
        with patch("reachy_mini_conversation_app.profiles.example.background_demo.asyncio.sleep", mock_sleep):
            result = await tool._run_demo(2, with_progress=False)

        assert result["status"] == "success"
        assert result["duration"] == 2
        mock_sleep.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_run_demo_with_progress(self) -> None:
        """Test _run_demo with progress tracking."""
        tool = BackgroundDemoTool()

        mock_task = MagicMock()
        mock_task.id = "task-progress-test"
        mock_task.name = "background_demo"

        mock_manager = MagicMock()
        mock_manager.get_running_tasks.return_value = [mock_task]
        mock_manager.update_progress = AsyncMock()

        mock_sleep = AsyncMock()
        with patch(
            "reachy_mini_conversation_app.profiles.example.background_demo.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            with patch("reachy_mini_conversation_app.profiles.example.background_demo.asyncio.sleep", mock_sleep):
                result = await tool._run_demo(3, with_progress=True)

        assert result["status"] == "success"
        assert result["duration"] == 3
        # Should have 3 progress updates (one per second)
        assert mock_manager.update_progress.call_count == 3

    @pytest.mark.asyncio
    async def test_run_demo_returns_message(self) -> None:
        """Test _run_demo returns completion message."""
        tool = BackgroundDemoTool()

        mock_sleep = AsyncMock()
        with patch("reachy_mini_conversation_app.profiles.example.background_demo.asyncio.sleep", mock_sleep):
            result = await tool._run_demo(5, with_progress=False)

        assert "message" in result
        assert "5 seconds" in result["message"]

    @pytest.mark.asyncio
    async def test_run_demo_progress_no_running_tasks(self) -> None:
        """Test _run_demo with progress when no running tasks found."""
        tool = BackgroundDemoTool()

        mock_manager = MagicMock()
        mock_manager.get_running_tasks.return_value = []  # No tasks running
        mock_manager.update_progress = AsyncMock()

        mock_sleep = AsyncMock()
        with patch(
            "reachy_mini_conversation_app.profiles.example.background_demo.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            with patch("reachy_mini_conversation_app.profiles.example.background_demo.asyncio.sleep", mock_sleep):
                result = await tool._run_demo(2, with_progress=True)

        assert result["status"] == "success"
        # update_progress should NOT be called since current_task_id is None
        mock_manager.update_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_demo_progress_with_other_tasks(self) -> None:
        """Test _run_demo with progress when other tasks are running but not background_demo."""
        tool = BackgroundDemoTool()

        # A task that is NOT background_demo
        mock_other_task = MagicMock()
        mock_other_task.id = "other-task-id"
        mock_other_task.name = "some_other_task"

        mock_manager = MagicMock()
        mock_manager.get_running_tasks.return_value = [mock_other_task]  # Different task
        mock_manager.update_progress = AsyncMock()

        mock_sleep = AsyncMock()
        with patch(
            "reachy_mini_conversation_app.profiles.example.background_demo.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            with patch("reachy_mini_conversation_app.profiles.example.background_demo.asyncio.sleep", mock_sleep):
                result = await tool._run_demo(2, with_progress=True)

        assert result["status"] == "success"
        # update_progress should NOT be called since we didn't find a matching task
        mock_manager.update_progress.assert_not_called()
