"""Unit tests for the task_cancel tool."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from reachy_mini_conversation_app.tools.task_cancel import TaskCancelTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.background_tasks import TaskStatus


class TestTaskCancelToolAttributes:
    """Tests for TaskCancelTool tool attributes."""

    def test_task_cancel_has_correct_name(self) -> None:
        """Test TaskCancelTool tool has correct name."""
        tool = TaskCancelTool()
        assert tool.name == "task_cancel"

    def test_task_cancel_has_description(self) -> None:
        """Test TaskCancelTool tool has description."""
        tool = TaskCancelTool()
        assert "cancel" in tool.description.lower()
        assert "task" in tool.description.lower()

    def test_task_cancel_has_parameters_schema(self) -> None:
        """Test TaskCancelTool tool has correct parameters schema."""
        tool = TaskCancelTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "task_id" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert schema["properties"]["task_id"]["type"] == "string"
        assert schema["properties"]["confirmed"]["type"] == "boolean"
        assert "task_id" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_task_cancel_spec(self) -> None:
        """Test TaskCancelTool tool spec generation."""
        tool = TaskCancelTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "task_cancel"


class TestTaskCancelToolExecution:
    """Tests for TaskCancelTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_task_cancel_missing_task_id(self, mock_deps: ToolDependencies) -> None:
        """Test task_cancel with missing task_id."""
        tool = TaskCancelTool()

        result = await tool(mock_deps, confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_task_cancel_empty_task_id(self, mock_deps: ToolDependencies) -> None:
        """Test task_cancel with empty task_id."""
        tool = TaskCancelTool()

        result = await tool(mock_deps, task_id="", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_task_cancel_task_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test task_cancel when task not found."""
        tool = TaskCancelTool()

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = None

        with patch(
            "reachy_mini_conversation_app.tools.task_cancel.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="nonexistent", confirmed=True)

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_task_cancel_task_not_running(self, mock_deps: ToolDependencies) -> None:
        """Test task_cancel when task is not running."""
        tool = TaskCancelTool()

        mock_task = MagicMock()
        mock_task.id = "task-123"
        mock_task.name = "completed_task"
        mock_task.status = TaskStatus.COMPLETED

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task

        with patch(
            "reachy_mini_conversation_app.tools.task_cancel.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="task-123", confirmed=True)

        assert result["status"] == "not_running"
        assert "not running" in result["message"]

    @pytest.mark.asyncio
    async def test_task_cancel_requires_confirmation(self, mock_deps: ToolDependencies) -> None:
        """Test task_cancel requires confirmation."""
        tool = TaskCancelTool()

        mock_task = MagicMock()
        mock_task.id = "task-456"
        mock_task.name = "running_task"
        mock_task.description = "A running task"
        mock_task.status = TaskStatus.RUNNING

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task

        with patch(
            "reachy_mini_conversation_app.tools.task_cancel.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="task-456", confirmed=False)

        assert result["status"] == "confirmation_required"
        assert "Are you sure" in result["message"]
        assert result["task_id"] == "task-456"

    @pytest.mark.asyncio
    async def test_task_cancel_success(self, mock_deps: ToolDependencies) -> None:
        """Test task_cancel successfully cancels task."""
        tool = TaskCancelTool()

        mock_task = MagicMock()
        mock_task.id = "task-789"
        mock_task.name = "cancelable_task"
        mock_task.status = TaskStatus.RUNNING

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task
        mock_manager.cancel_task = AsyncMock(return_value=True)

        with patch(
            "reachy_mini_conversation_app.tools.task_cancel.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="task-789", confirmed=True)

        assert result["status"] == "cancelled"
        assert "cancelled" in result["message"].lower()
        mock_manager.cancel_task.assert_called_once_with("task-789")

    @pytest.mark.asyncio
    async def test_task_cancel_failure(self, mock_deps: ToolDependencies) -> None:
        """Test task_cancel when cancellation fails."""
        tool = TaskCancelTool()

        mock_task = MagicMock()
        mock_task.id = "task-fail"
        mock_task.name = "uncancelable_task"
        mock_task.status = TaskStatus.RUNNING

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task
        mock_manager.cancel_task = AsyncMock(return_value=False)

        with patch(
            "reachy_mini_conversation_app.tools.task_cancel.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="task-fail", confirmed=True)

        assert "error" in result
        assert "Could not cancel" in result["error"]

    @pytest.mark.asyncio
    async def test_task_cancel_default_confirmed_false(self, mock_deps: ToolDependencies) -> None:
        """Test task_cancel defaults confirmed to False."""
        tool = TaskCancelTool()

        mock_task = MagicMock()
        mock_task.id = "task-default"
        mock_task.name = "default_task"
        mock_task.description = "Default task"
        mock_task.status = TaskStatus.RUNNING

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task

        with patch(
            "reachy_mini_conversation_app.tools.task_cancel.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            # Not passing confirmed parameter
            result = await tool(mock_deps, task_id="task-default")

        # Should require confirmation
        assert result["status"] == "confirmation_required"
