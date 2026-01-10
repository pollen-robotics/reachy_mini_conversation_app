"""Unit tests for the background_tasks module."""

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from reachy_mini_conversation_app.background_tasks import (
    TaskStatus,
    TaskNotification,
    BackgroundTask,
    BackgroundTaskManager,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_values(self) -> None:
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_task_status_has_all_expected_states(self) -> None:
        """Test that all expected states are present."""
        states = [s.value for s in TaskStatus]
        assert "pending" in states
        assert "running" in states
        assert "completed" in states
        assert "failed" in states
        assert "cancelled" in states


class TestTaskNotification:
    """Tests for TaskNotification dataclass."""

    def test_task_notification_required_fields(self) -> None:
        """Test TaskNotification with required fields only."""
        notification = TaskNotification(
            task_id="abc123",
            task_name="test_task",
            status=TaskStatus.COMPLETED,
            message="Task completed successfully",
        )

        assert notification.task_id == "abc123"
        assert notification.task_name == "test_task"
        assert notification.status == TaskStatus.COMPLETED
        assert notification.message == "Task completed successfully"
        assert notification.result is None
        assert notification.error is None

    def test_task_notification_all_fields(self) -> None:
        """Test TaskNotification with all fields."""
        result = {"output": "test output"}
        notification = TaskNotification(
            task_id="abc123",
            task_name="test_task",
            status=TaskStatus.FAILED,
            message="Task failed",
            result=result,
            error="Something went wrong",
        )

        assert notification.result == result
        assert notification.error == "Something went wrong"


class TestBackgroundTask:
    """Tests for BackgroundTask dataclass."""

    def test_background_task_defaults(self) -> None:
        """Test BackgroundTask with default values."""
        task = BackgroundTask(
            id="task123",
            name="test_task",
            description="A test task",
        )

        assert task.id == "task123"
        assert task.name == "test_task"
        assert task.description == "A test task"
        assert task.status == TaskStatus.PENDING
        assert task.progress is None
        assert task.progress_message is None
        assert task.result is None
        assert task.error is None
        assert task.started_at > 0
        assert task.completed_at is None
        assert task._task is None

    def test_background_task_with_progress(self) -> None:
        """Test BackgroundTask with progress tracking."""
        task = BackgroundTask(
            id="task123",
            name="test_task",
            description="A test task",
            progress=0.5,
            progress_message="50% complete",
        )

        assert task.progress == 0.5
        assert task.progress_message == "50% complete"


class TestBackgroundTaskManager:
    """Tests for BackgroundTaskManager class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        BackgroundTaskManager.reset_instance()
        yield
        BackgroundTaskManager.reset_instance()

    def test_get_instance_creates_singleton(self) -> None:
        """Test that get_instance creates a singleton."""
        manager1 = BackgroundTaskManager.get_instance()
        manager2 = BackgroundTaskManager.get_instance()

        assert manager1 is manager2

    def test_reset_instance_clears_singleton(self) -> None:
        """Test that reset_instance clears the singleton."""
        manager1 = BackgroundTaskManager.get_instance()
        BackgroundTaskManager.reset_instance()
        manager2 = BackgroundTaskManager.get_instance()

        assert manager1 is not manager2

    @pytest.mark.asyncio
    async def test_set_connection(self) -> None:
        """Test setting connection references."""
        manager = BackgroundTaskManager.get_instance()
        mock_connection = MagicMock()
        mock_queue: asyncio.Queue[MagicMock] = asyncio.Queue()

        manager.set_connection(mock_connection, mock_queue)

        assert manager.connection is mock_connection
        assert manager.output_queue is mock_queue

    @pytest.mark.asyncio
    async def test_clear_connection(self) -> None:
        """Test clearing connection references."""
        manager = BackgroundTaskManager.get_instance()
        mock_connection = MagicMock()
        mock_queue: asyncio.Queue[MagicMock] = asyncio.Queue()

        manager.set_connection(mock_connection, mock_queue)
        manager.clear_connection()

        assert manager.connection is None
        assert manager.output_queue is None

    def test_connection_property_returns_none_when_not_set(self) -> None:
        """Test connection property returns None when not set."""
        manager = BackgroundTaskManager.get_instance()
        assert manager.connection is None

    def test_output_queue_property_returns_none_when_not_set(self) -> None:
        """Test output_queue property returns None when not set."""
        manager = BackgroundTaskManager.get_instance()
        assert manager.output_queue is None

    @pytest.mark.asyncio
    async def test_start_task_creates_running_task(self) -> None:
        """Test that start_task creates a running task."""
        manager = BackgroundTaskManager.get_instance()

        started = asyncio.Event()

        async def dummy_coroutine() -> dict:
            started.set()
            await asyncio.sleep(10)  # Long sleep, we'll cancel
            return {"status": "success"}

        task = await manager.start_task(
            name="test_task",
            description="Test description",
            coroutine=dummy_coroutine(),
        )

        # Wait for coroutine to actually start
        await asyncio.wait_for(started.wait(), timeout=1.0)

        assert task.id is not None
        assert len(task.id) == 8  # UUID truncated to 8 chars
        assert task.name == "test_task"
        assert task.description == "Test description"
        assert task.status == TaskStatus.RUNNING
        assert task.progress is None
        assert task._task is not None

        # Cancel and await the task properly
        task._task.cancel()
        try:
            await task._task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_start_task_with_progress(self) -> None:
        """Test that start_task with progress tracking sets progress to 0."""
        manager = BackgroundTaskManager.get_instance()

        started = asyncio.Event()

        async def dummy_coroutine() -> dict:
            started.set()
            await asyncio.sleep(10)
            return {}

        task = await manager.start_task(
            name="test_task",
            description="Test",
            coroutine=dummy_coroutine(),
            with_progress=True,
        )

        # Wait for coroutine to start
        await asyncio.wait_for(started.wait(), timeout=1.0)

        assert task.progress == 0.0

        # Cancel and await the task properly
        task._task.cancel()
        try:
            await task._task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_task_completes_successfully(self) -> None:
        """Test that a task completes and queues notification."""
        manager = BackgroundTaskManager.get_instance()

        async def success_coroutine() -> dict:
            return {"message": "Done!"}

        task = await manager.start_task(
            name="success_task",
            description="Test success",
            coroutine=success_coroutine(),
        )

        # Wait for completion
        await asyncio.sleep(0.1)

        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"message": "Done!"}
        assert task.completed_at is not None

        # Check notification was queued
        notification = manager.get_pending_notification()
        assert notification is not None
        assert notification.task_name == "success_task"
        assert notification.status == TaskStatus.COMPLETED
        assert notification.message == "Done!"

    @pytest.mark.asyncio
    async def test_task_fails_with_exception(self) -> None:
        """Test that a failing task sets error status."""
        manager = BackgroundTaskManager.get_instance()

        async def failing_coroutine() -> dict:
            raise ValueError("Test error")

        task = await manager.start_task(
            name="failing_task",
            description="Test failure",
            coroutine=failing_coroutine(),
        )

        # Wait for completion
        await asyncio.sleep(0.1)

        assert task.status == TaskStatus.FAILED
        assert task.error == "Test error"
        assert task.completed_at is not None

        # Check notification
        notification = manager.get_pending_notification()
        assert notification is not None
        assert notification.status == TaskStatus.FAILED
        assert "failed" in notification.message

    @pytest.mark.asyncio
    async def test_cancel_task(self) -> None:
        """Test cancelling a running task."""
        manager = BackgroundTaskManager.get_instance()

        cancel_reached = False

        async def long_coroutine() -> dict:
            nonlocal cancel_reached
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancel_reached = True
                raise
            return {}

        task = await manager.start_task(
            name="long_task",
            description="Long running task",
            coroutine=long_coroutine(),
        )

        # Give the task a moment to start
        await asyncio.sleep(0.01)

        # Cancel the task
        result = await manager.cancel_task(task.id)
        assert result is True

        # Wait for the task wrapper to finish processing
        await asyncio.sleep(0.1)

        # The status should be CANCELLED (set by _run_task before re-raising)
        assert task.status == TaskStatus.CANCELLED
        assert cancel_reached is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self) -> None:
        """Test cancelling a task that doesn't exist."""
        manager = BackgroundTaskManager.get_instance()

        result = await manager.cancel_task("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self) -> None:
        """Test that completed tasks cannot be cancelled."""
        manager = BackgroundTaskManager.get_instance()

        async def quick_coroutine() -> dict:
            return {}

        task = await manager.start_task(
            name="quick_task",
            description="Quick task",
            coroutine=quick_coroutine(),
        )

        await asyncio.sleep(0.1)  # Wait for completion

        result = await manager.cancel_task(task.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_progress(self) -> None:
        """Test updating task progress."""
        manager = BackgroundTaskManager.get_instance()

        started = asyncio.Event()

        async def progress_coroutine() -> dict:
            started.set()
            await asyncio.sleep(10)
            return {}

        task = await manager.start_task(
            name="progress_task",
            description="Task with progress",
            coroutine=progress_coroutine(),
            with_progress=True,
        )

        # Wait for coroutine to start
        await asyncio.wait_for(started.wait(), timeout=1.0)

        # Update progress
        result = await manager.update_progress(task.id, 0.5, "Halfway there")

        assert result is True
        assert task.progress == 0.5
        assert task.progress_message == "Halfway there"

        # Cancel and await the task properly
        task._task.cancel()
        try:
            await task._task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_update_progress_clamps_values(self) -> None:
        """Test that progress values are clamped to 0.0-1.0."""
        manager = BackgroundTaskManager.get_instance()

        started = asyncio.Event()

        async def dummy() -> dict:
            started.set()
            await asyncio.sleep(10)
            return {}

        task = await manager.start_task(
            name="test",
            description="Test",
            coroutine=dummy(),
            with_progress=True,
        )

        # Wait for coroutine to start
        await asyncio.wait_for(started.wait(), timeout=1.0)

        # Test clamping high value
        await manager.update_progress(task.id, 1.5)
        assert task.progress == 1.0

        # Test clamping low value
        await manager.update_progress(task.id, -0.5)
        assert task.progress == 0.0

        # Cancel and await the task properly
        task._task.cancel()
        try:
            await task._task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_update_progress_fails_for_task_without_progress(self) -> None:
        """Test that update_progress fails for tasks not tracking progress."""
        manager = BackgroundTaskManager.get_instance()

        started = asyncio.Event()

        async def dummy() -> dict:
            started.set()
            await asyncio.sleep(10)
            return {}

        task = await manager.start_task(
            name="test",
            description="Test",
            coroutine=dummy(),
            with_progress=False,
        )

        # Wait for coroutine to start
        await asyncio.wait_for(started.wait(), timeout=1.0)

        result = await manager.update_progress(task.id, 0.5)
        assert result is False

        # Cancel and await the task properly
        task._task.cancel()
        try:
            await task._task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_update_progress_fails_for_nonexistent_task(self) -> None:
        """Test that update_progress fails for nonexistent task."""
        manager = BackgroundTaskManager.get_instance()

        result = await manager.update_progress("nonexistent", 0.5)
        assert result is False

    def test_get_task(self) -> None:
        """Test getting a task by ID."""
        manager = BackgroundTaskManager.get_instance()

        # Manually add a task
        task = BackgroundTask(id="test123", name="test", description="Test")
        manager._tasks["test123"] = task

        result = manager.get_task("test123")
        assert result is task

        result = manager.get_task("nonexistent")
        assert result is None

    def test_get_running_tasks(self) -> None:
        """Test getting running tasks."""
        manager = BackgroundTaskManager.get_instance()

        # Add tasks with different statuses
        manager._tasks["running1"] = BackgroundTask(
            id="running1", name="task1", description="Running 1", status=TaskStatus.RUNNING
        )
        manager._tasks["running2"] = BackgroundTask(
            id="running2", name="task2", description="Running 2", status=TaskStatus.RUNNING
        )
        manager._tasks["completed"] = BackgroundTask(
            id="completed", name="task3", description="Completed", status=TaskStatus.COMPLETED
        )

        running = manager.get_running_tasks()

        assert len(running) == 2
        assert all(t.status == TaskStatus.RUNNING for t in running)

    def test_get_all_tasks_sorted_by_start_time(self) -> None:
        """Test getting all tasks sorted by start time."""
        manager = BackgroundTaskManager.get_instance()

        # Add tasks with different start times
        now = time.monotonic()
        manager._tasks["old"] = BackgroundTask(
            id="old", name="old_task", description="Old", started_at=now - 100
        )
        manager._tasks["new"] = BackgroundTask(
            id="new", name="new_task", description="New", started_at=now
        )
        manager._tasks["mid"] = BackgroundTask(
            id="mid", name="mid_task", description="Mid", started_at=now - 50
        )

        tasks = manager.get_all_tasks()

        assert len(tasks) == 3
        assert tasks[0].id == "new"
        assert tasks[1].id == "mid"
        assert tasks[2].id == "old"

    def test_get_all_tasks_respects_limit(self) -> None:
        """Test that get_all_tasks respects the limit parameter."""
        manager = BackgroundTaskManager.get_instance()

        for i in range(20):
            manager._tasks[f"task{i}"] = BackgroundTask(
                id=f"task{i}", name=f"task{i}", description=f"Task {i}"
            )

        tasks = manager.get_all_tasks(limit=5)
        assert len(tasks) == 5

    def test_get_pending_notification_empty_queue(self) -> None:
        """Test get_pending_notification with empty queue."""
        manager = BackgroundTaskManager.get_instance()

        result = manager.get_pending_notification()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_notification_with_timeout(self) -> None:
        """Test get_notification with timeout."""
        manager = BackgroundTaskManager.get_instance()

        # Should timeout and return None
        result = await manager.get_notification(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_notification_success(self) -> None:
        """Test get_notification returns notification when available."""
        manager = BackgroundTaskManager.get_instance()

        # Add notification to queue
        notification = TaskNotification(
            task_id="test",
            task_name="test_task",
            status=TaskStatus.COMPLETED,
            message="Done",
        )
        await manager._notification_queue.put(notification)

        result = await manager.get_notification(timeout=1.0)
        assert result is notification

    def test_has_pending_notifications(self) -> None:
        """Test has_pending_notifications."""
        manager = BackgroundTaskManager.get_instance()

        assert manager.has_pending_notifications() is False

        manager._notification_queue.put_nowait(
            TaskNotification(
                task_id="test",
                task_name="test",
                status=TaskStatus.COMPLETED,
                message="Done",
            )
        )

        assert manager.has_pending_notifications() is True

    def test_cleanup_old_tasks(self) -> None:
        """Test cleanup_old_tasks removes old completed tasks."""
        manager = BackgroundTaskManager.get_instance()

        now = time.monotonic()

        # Add old completed task
        manager._tasks["old_completed"] = BackgroundTask(
            id="old_completed",
            name="old",
            description="Old completed",
            status=TaskStatus.COMPLETED,
            completed_at=now - 7200,  # 2 hours ago
        )

        # Add recent completed task
        manager._tasks["recent_completed"] = BackgroundTask(
            id="recent_completed",
            name="recent",
            description="Recent completed",
            status=TaskStatus.COMPLETED,
            completed_at=now - 100,  # 100 seconds ago
        )

        # Add running task (should not be removed)
        manager._tasks["running"] = BackgroundTask(
            id="running",
            name="running",
            description="Running",
            status=TaskStatus.RUNNING,
        )

        removed = manager.cleanup_old_tasks(max_age_seconds=3600)

        assert removed == 1
        assert "old_completed" not in manager._tasks
        assert "recent_completed" in manager._tasks
        assert "running" in manager._tasks

    def test_cleanup_old_tasks_handles_failed_and_cancelled(self) -> None:
        """Test that cleanup handles failed and cancelled tasks."""
        manager = BackgroundTaskManager.get_instance()

        now = time.monotonic()

        manager._tasks["old_failed"] = BackgroundTask(
            id="old_failed",
            name="failed",
            description="Old failed",
            status=TaskStatus.FAILED,
            completed_at=now - 7200,
        )

        manager._tasks["old_cancelled"] = BackgroundTask(
            id="old_cancelled",
            name="cancelled",
            description="Old cancelled",
            status=TaskStatus.CANCELLED,
            completed_at=now - 7200,
        )

        removed = manager.cleanup_old_tasks(max_age_seconds=3600)

        assert removed == 2
        assert "old_failed" not in manager._tasks
        assert "old_cancelled" not in manager._tasks

    def test_get_status_summary(self) -> None:
        """Test get_status_summary returns correct summary."""
        manager = BackgroundTaskManager.get_instance()

        now = time.monotonic()

        # Add tasks with different statuses
        manager._tasks["pending"] = BackgroundTask(
            id="pending", name="pending", description="Pending", status=TaskStatus.PENDING
        )
        manager._tasks["running"] = BackgroundTask(
            id="running",
            name="running_task",
            description="Running task desc",
            status=TaskStatus.RUNNING,
            progress=0.5,
            progress_message="50% done",
            started_at=now - 10,
        )
        manager._tasks["completed"] = BackgroundTask(
            id="completed", name="completed", description="Completed", status=TaskStatus.COMPLETED
        )
        manager._tasks["failed"] = BackgroundTask(
            id="failed", name="failed", description="Failed", status=TaskStatus.FAILED
        )

        summary = manager.get_status_summary()

        assert summary["total"] == 4
        assert summary["counts"]["pending"] == 1
        assert summary["counts"]["running"] == 1
        assert summary["counts"]["completed"] == 1
        assert summary["counts"]["failed"] == 1
        assert summary["counts"]["cancelled"] == 0

        assert len(summary["running"]) == 1
        running_task = summary["running"][0]
        assert running_task["id"] == "running"
        assert running_task["name"] == "running_task"
        assert running_task["description"] == "Running task desc"
        assert running_task["progress"] == 0.5
        assert running_task["progress_message"] == "50% done"
        assert running_task["elapsed_seconds"] >= 10


class TestSummarizeResult:
    """Tests for _summarize_result method."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        BackgroundTaskManager.reset_instance()
        yield
        BackgroundTaskManager.reset_instance()

    def test_summarize_result_none(self) -> None:
        """Test summarizing None result."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result(None)
        assert result == "Task completed successfully."

    def test_summarize_result_empty_dict(self) -> None:
        """Test summarizing empty dict."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({})
        assert result == "Task completed successfully."

    def test_summarize_result_with_message(self) -> None:
        """Test summarizing result with message key."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({"message": "Custom message"})
        assert result == "Custom message"

    def test_summarize_result_with_success_status(self) -> None:
        """Test summarizing result with success status."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({"status": "success"})
        assert result == "Operation completed successfully."

    def test_summarize_result_with_success_and_path(self) -> None:
        """Test summarizing result with success status and path."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({"status": "success", "path": "/tmp/output.txt"})
        assert result == "Saved to /tmp/output.txt"

    def test_summarize_result_with_success_and_output(self) -> None:
        """Test summarizing result with success status and output."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({"status": "success", "output": "Hello world"})
        assert result == "Output: Hello world"

    def test_summarize_result_truncates_long_output(self) -> None:
        """Test that long outputs are truncated."""
        manager = BackgroundTaskManager.get_instance()
        long_output = "x" * 200
        result = manager._summarize_result({"status": "success", "output": long_output})
        assert len(result) < 200
        assert "..." in result

    def test_summarize_result_with_error_status(self) -> None:
        """Test summarizing result with error status."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({"status": "error", "error": "Something failed"})
        assert result == "Error: Something failed"

    def test_summarize_result_with_error_status_no_error_key(self) -> None:
        """Test summarizing result with error status but no error key."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({"status": "error"})
        assert result == "Error: Unknown error"

    def test_summarize_result_with_error_key_only(self) -> None:
        """Test summarizing result with just error key."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({"error": "Direct error"})
        assert result == "Error: Direct error"

    def test_summarize_result_fallback(self) -> None:
        """Test summarizing result with no recognized keys."""
        manager = BackgroundTaskManager.get_instance()
        result = manager._summarize_result({"foo": "bar", "baz": 123})
        assert result == "Task completed."

    def test_summarize_result_with_unknown_status(self) -> None:
        """Test summarizing result with unknown status value (branch 237->239)."""
        manager = BackgroundTaskManager.get_instance()
        # Status is present but not "success" or "error"
        result = manager._summarize_result({"status": "pending"})
        assert result == "Task completed."


class TestBackgroundTaskManagerEdgeCases:
    """Tests for edge cases and branch coverage."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        BackgroundTaskManager.reset_instance()
        yield
        BackgroundTaskManager.reset_instance()

    def test_set_connection_with_explicit_loop(self) -> None:
        """Test set_connection when loop is explicitly provided (line 111)."""
        manager = BackgroundTaskManager.get_instance()
        mock_connection = MagicMock()
        mock_queue: asyncio.Queue[MagicMock] = asyncio.Queue()
        explicit_loop = asyncio.new_event_loop()

        try:
            manager.set_connection(mock_connection, mock_queue, loop=explicit_loop)
            assert manager._loop is explicit_loop
        finally:
            explicit_loop.close()

    def test_set_connection_without_running_loop(self) -> None:
        """Test set_connection when no running event loop (lines 115-117)."""
        manager = BackgroundTaskManager.get_instance()
        mock_connection = MagicMock()
        mock_queue: asyncio.Queue[MagicMock] = asyncio.Queue()

        # Call set_connection outside of async context (no running loop)
        # This should trigger the RuntimeError fallback path
        manager.set_connection(mock_connection, mock_queue)

        # Should have created or obtained a loop
        assert manager._loop is not None

    @pytest.mark.asyncio
    async def test_cancel_task_with_no_asyncio_task(self) -> None:
        """Test cancel_task when task._task is None (line 296)."""
        manager = BackgroundTaskManager.get_instance()

        # Manually add a task without an asyncio task attached
        task = BackgroundTask(
            id="notask",
            name="no_task",
            description="Task without asyncio task",
            status=TaskStatus.RUNNING,
        )
        task._task = None  # Explicitly None
        manager._tasks["notask"] = task

        result = await manager.cancel_task("notask")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_notification_without_timeout(self) -> None:
        """Test get_notification without timeout (line 349)."""
        manager = BackgroundTaskManager.get_instance()

        # Add notification to queue first
        notification = TaskNotification(
            task_id="test",
            task_name="test_task",
            status=TaskStatus.COMPLETED,
            message="Done",
        )
        await manager._notification_queue.put(notification)

        # Call without timeout - should return immediately since item is available
        result = await manager.get_notification(timeout=None)
        assert result is notification

    def test_cleanup_old_tasks_logs_when_removing(self) -> None:
        """Test cleanup_old_tasks logging when tasks are removed (lines 377-380)."""
        manager = BackgroundTaskManager.get_instance()

        now = time.monotonic()

        # Add multiple old tasks to ensure removal and logging
        for i in range(3):
            manager._tasks[f"old_task_{i}"] = BackgroundTask(
                id=f"old_task_{i}",
                name=f"old_{i}",
                description=f"Old task {i}",
                status=TaskStatus.COMPLETED,
                completed_at=now - 7200,  # 2 hours ago
            )

        # Should remove all 3 and trigger the logging branch
        removed = manager.cleanup_old_tasks(max_age_seconds=3600)

        assert removed == 3
        assert len(manager._tasks) == 0

    def test_cleanup_old_tasks_nothing_to_remove(self) -> None:
        """Test cleanup_old_tasks when no tasks need removal (branch 377->380 False)."""
        manager = BackgroundTaskManager.get_instance()

        now = time.monotonic()

        # Add only recent tasks that shouldn't be removed
        manager._tasks["recent"] = BackgroundTask(
            id="recent",
            name="recent",
            description="Recent completed",
            status=TaskStatus.COMPLETED,
            completed_at=now - 100,  # Only 100 seconds ago
        )
        manager._tasks["running"] = BackgroundTask(
            id="running",
            name="running",
            description="Running",
            status=TaskStatus.RUNNING,
        )

        # With max_age of 3600, nothing should be removed
        removed = manager.cleanup_old_tasks(max_age_seconds=3600)

        assert removed == 0
        assert len(manager._tasks) == 2
