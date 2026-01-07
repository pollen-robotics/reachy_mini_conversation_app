"""Unit tests for the task_status tool."""

import time
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.task_status import TaskStatusTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.background_tasks import TaskStatus


class TestTaskStatusToolAttributes:
    """Tests for TaskStatusTool tool attributes."""

    def test_task_status_has_correct_name(self) -> None:
        """Test TaskStatusTool tool has correct name."""
        tool = TaskStatusTool()
        assert tool.name == "task_status"

    def test_task_status_has_description(self) -> None:
        """Test TaskStatusTool tool has description."""
        tool = TaskStatusTool()
        assert "status" in tool.description.lower()
        assert "task" in tool.description.lower()

    def test_task_status_has_parameters_schema(self) -> None:
        """Test TaskStatusTool tool has correct parameters schema."""
        tool = TaskStatusTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "task_id" in schema["properties"]
        assert schema["properties"]["task_id"]["type"] == "string"
        assert schema["required"] == []

    def test_task_status_spec(self) -> None:
        """Test TaskStatusTool tool spec generation."""
        tool = TaskStatusTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "task_status"


class TestTaskStatusToolExecution:
    """Tests for TaskStatusTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_task_status_no_tasks_running(self, mock_deps: ToolDependencies) -> None:
        """Test task_status when no tasks are running."""
        tool = TaskStatusTool()

        mock_manager = MagicMock()
        mock_manager.get_running_tasks.return_value = []

        with patch(
            "reachy_mini_conversation_app.tools.task_status.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps)

        assert result["status"] == "idle"
        assert "No tasks running" in result["message"]

    @pytest.mark.asyncio
    async def test_task_status_with_running_tasks(self, mock_deps: ToolDependencies) -> None:
        """Test task_status with running tasks."""
        tool = TaskStatusTool()

        mock_task = MagicMock()
        mock_task.id = "task-123"
        mock_task.name = "test_task"
        mock_task.description = "A test task"
        mock_task.started_at = time.monotonic() - 5  # 5 seconds ago
        mock_task.progress = None
        mock_task.progress_message = None

        mock_manager = MagicMock()
        mock_manager.get_running_tasks.return_value = [mock_task]

        with patch(
            "reachy_mini_conversation_app.tools.task_status.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps)

        assert result["status"] == "running"
        assert result["count"] == 1
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task_id"] == "task-123"

    @pytest.mark.asyncio
    async def test_task_status_specific_task_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test task_status for specific task not found."""
        tool = TaskStatusTool()

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = None

        with patch(
            "reachy_mini_conversation_app.tools.task_status.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_task_status_specific_task_found(self, mock_deps: ToolDependencies) -> None:
        """Test task_status for specific task found."""
        tool = TaskStatusTool()

        mock_task = MagicMock()
        mock_task.id = "task-456"
        mock_task.name = "specific_task"
        mock_task.description = "A specific task"
        mock_task.started_at = time.monotonic() - 10
        mock_task.status = TaskStatus.RUNNING
        mock_task.progress = None
        mock_task.progress_message = None
        mock_task.result = None
        mock_task.error = None

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task

        with patch(
            "reachy_mini_conversation_app.tools.task_status.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="task-456")

        assert result["task_id"] == "task-456"
        assert result["name"] == "specific_task"
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_task_status_with_progress(self, mock_deps: ToolDependencies) -> None:
        """Test task_status with progress tracking."""
        tool = TaskStatusTool()

        mock_task = MagicMock()
        mock_task.id = "task-789"
        mock_task.name = "progress_task"
        mock_task.description = "Task with progress"
        mock_task.started_at = time.monotonic() - 3
        mock_task.status = TaskStatus.RUNNING
        mock_task.progress = 0.5
        mock_task.progress_message = "Halfway there"
        mock_task.result = None
        mock_task.error = None

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task

        with patch(
            "reachy_mini_conversation_app.tools.task_status.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="task-789")

        assert result["progress"] == 50.0
        assert result["progress_percent"] == "50%"
        assert result["progress_message"] == "Halfway there"

    @pytest.mark.asyncio
    async def test_task_status_completed_task(self, mock_deps: ToolDependencies) -> None:
        """Test task_status for completed task."""
        tool = TaskStatusTool()

        mock_task = MagicMock()
        mock_task.id = "task-completed"
        mock_task.name = "completed_task"
        mock_task.description = "Completed task"
        mock_task.started_at = time.monotonic() - 20
        mock_task.status = TaskStatus.COMPLETED
        mock_task.progress = None
        mock_task.progress_message = None
        mock_task.result = {"data": "success"}
        mock_task.error = None

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task

        with patch(
            "reachy_mini_conversation_app.tools.task_status.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="task-completed")

        assert result["status"] == "completed"
        assert result["result"] == {"data": "success"}

    @pytest.mark.asyncio
    async def test_task_status_failed_task(self, mock_deps: ToolDependencies) -> None:
        """Test task_status for failed task."""
        tool = TaskStatusTool()

        mock_task = MagicMock()
        mock_task.id = "task-failed"
        mock_task.name = "failed_task"
        mock_task.description = "Failed task"
        mock_task.started_at = time.monotonic() - 15
        mock_task.status = TaskStatus.FAILED
        mock_task.progress = None
        mock_task.progress_message = None
        mock_task.result = None
        mock_task.error = "Something went wrong"

        mock_manager = MagicMock()
        mock_manager.get_task.return_value = mock_task

        with patch(
            "reachy_mini_conversation_app.tools.task_status.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps, task_id="task-failed")

        assert result["status"] == "failed"
        assert result["error"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_task_status_multiple_running_tasks(self, mock_deps: ToolDependencies) -> None:
        """Test task_status with multiple running tasks."""
        tool = TaskStatusTool()

        mock_task1 = MagicMock()
        mock_task1.id = "task-1"
        mock_task1.name = "task_one"
        mock_task1.description = "First task"
        mock_task1.started_at = time.monotonic() - 10
        mock_task1.progress = None
        mock_task1.progress_message = None

        mock_task2 = MagicMock()
        mock_task2.id = "task-2"
        mock_task2.name = "task_two"
        mock_task2.description = "Second task"
        mock_task2.started_at = time.monotonic() - 5
        mock_task2.progress = 0.75
        mock_task2.progress_message = "Almost done"

        mock_manager = MagicMock()
        mock_manager.get_running_tasks.return_value = [mock_task1, mock_task2]

        with patch(
            "reachy_mini_conversation_app.tools.task_status.BackgroundTaskManager.get_instance",
            return_value=mock_manager,
        ):
            result = await tool(mock_deps)

        assert result["status"] == "running"
        assert result["count"] == 2
        assert len(result["tasks"]) == 2
