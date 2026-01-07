"""Background task orchestrator for non-blocking tool execution.

Allows tools to run long operations asynchronously while the robot
continues conversing. Tasks can be tracked, cancelled, and their
completion is announced vocally via a silent notification queue.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskNotification:
    """Notification payload for completed tasks."""

    task_id: str
    task_name: str
    status: TaskStatus
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class BackgroundTask:
    """Represents a background task."""

    id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    progress: Optional[float] = None  # 0.0 - 1.0, None if not tracking progress
    progress_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    completed_at: Optional[float] = None
    _task: Optional[asyncio.Task[None]] = field(default=None, repr=False)


class BackgroundTaskManager:
    """Manages background tasks for non-blocking tool execution.

    Singleton pattern - accessed via get_instance().

    Features:
    - Start async tasks without blocking the conversation
    - Track task status and progress
    - Cancel running tasks
    - Silent notification queue for completion announcements
    """

    _instance: Optional["BackgroundTaskManager"] = None

    def __init__(self) -> None:
        self._tasks: Dict[str, BackgroundTask] = {}
        self._notification_queue: asyncio.Queue[TaskNotification] = asyncio.Queue()
        self._connection_ref: Optional[weakref.ref[Any]] = None
        self._output_queue_ref: Optional[weakref.ref[asyncio.Queue[Any]]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_instance(cls) -> "BackgroundTaskManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def set_connection(
        self,
        connection: Any,
        output_queue: asyncio.Queue[Any],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Set the OpenAI connection and output queue references.

        Args:
            connection: The OpenAI realtime connection object
            output_queue: The async queue for UI updates
            loop: The event loop (defaults to current running loop)
        """
        self._connection_ref = weakref.ref(connection)
        self._output_queue_ref = weakref.ref(output_queue)
        self._loop = loop or asyncio.get_event_loop()
        logger.debug("BackgroundTaskManager: connection set")

    def clear_connection(self) -> None:
        """Clear connection references (called on disconnect)."""
        self._connection_ref = None
        self._output_queue_ref = None
        logger.debug("BackgroundTaskManager: connection cleared")

    @property
    def connection(self) -> Optional[Any]:
        """Get the connection if still available."""
        return self._connection_ref() if self._connection_ref else None

    @property
    def output_queue(self) -> Optional[asyncio.Queue[Any]]:
        """Get the output queue if still available."""
        return self._output_queue_ref() if self._output_queue_ref else None

    async def start_task(
        self,
        name: str,
        description: str,
        coroutine: Coroutine[Any, Any, Dict[str, Any]],
        with_progress: bool = False,
    ) -> BackgroundTask:
        """Start a new background task.

        Args:
            name: Tool/task name (e.g., "github_clone")
            description: Human-readable description (e.g., "Cloning repository X")
            coroutine: The async coroutine to execute
            with_progress: Whether to track progress (0.0-1.0)

        Returns:
            BackgroundTask object with task ID
        """
        task_id = str(uuid.uuid4())[:8]
        bg_task = BackgroundTask(
            id=task_id,
            name=name,
            description=description,
            progress=0.0 if with_progress else None,
        )
        self._tasks[task_id] = bg_task

        # Create the async task
        async_task = asyncio.create_task(
            self._run_task(bg_task, coroutine),
            name=f"bg-{name}-{task_id}",
        )
        bg_task._task = async_task
        bg_task.status = TaskStatus.RUNNING

        logger.info(f"Started background task: {name} (id={task_id})")

        return bg_task

    async def _run_task(
        self,
        bg_task: BackgroundTask,
        coroutine: Coroutine[Any, Any, Dict[str, Any]],
    ) -> None:
        """Execute the task and handle completion."""
        try:
            result = await coroutine
            bg_task.result = result
            bg_task.status = TaskStatus.COMPLETED
            bg_task.completed_at = asyncio.get_event_loop().time()

            # Build completion message
            message = self._summarize_result(result)
            logger.info(f"Background task completed: {bg_task.name} (id={bg_task.id})")

        except asyncio.CancelledError:
            bg_task.status = TaskStatus.CANCELLED
            bg_task.completed_at = asyncio.get_event_loop().time()
            message = f"Task '{bg_task.name}' was cancelled."
            logger.info(f"Background task cancelled: {bg_task.name} (id={bg_task.id})")
            raise

        except Exception as e:
            bg_task.error = str(e)
            bg_task.status = TaskStatus.FAILED
            bg_task.completed_at = asyncio.get_event_loop().time()
            message = f"Task '{bg_task.name}' failed: {e}"
            logger.error(f"Background task failed: {bg_task.name} (id={bg_task.id}): {e}")

        # Queue notification for silent delivery
        notification = TaskNotification(
            task_id=bg_task.id,
            task_name=bg_task.name,
            status=bg_task.status,
            message=message,
            result=bg_task.result,
            error=bg_task.error,
        )
        await self._notification_queue.put(notification)
        logger.debug(f"Queued notification for task: {bg_task.name} (id={bg_task.id})")

    def _summarize_result(self, result: Optional[Dict[str, Any]]) -> str:
        """Create a brief summary of the task result for vocalization."""
        if not result:
            return "Task completed successfully."

        # Look for common result keys
        if "message" in result:
            return str(result["message"])
        if "status" in result:
            status = result["status"]
            if status == "success":
                if "path" in result:
                    return f"Saved to {result['path']}"
                if "output" in result:
                    output = str(result["output"])
                    # Truncate long outputs
                    if len(output) > 100:
                        output = output[:100] + "..."
                    return f"Output: {output}"
                return "Operation completed successfully."
            elif status == "error":
                return f"Error: {result.get('error', 'Unknown error')}"
        if "error" in result:
            return f"Error: {result['error']}"

        return "Task completed."

    async def update_progress(
        self,
        task_id: str,
        progress: float,
        message: Optional[str] = None,
    ) -> bool:
        """Update progress for a task (for tasks with with_progress=True).

        Args:
            task_id: The task ID
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message (e.g., "50% downloaded")

        Returns:
            True if updated successfully, False if task not found or not tracking progress
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.progress is None:
            # Task not tracking progress
            return False

        task.progress = max(0.0, min(1.0, progress))
        task.progress_message = message
        logger.debug(f"Task {task_id} progress: {progress:.1%} - {message or ''}")
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task by ID.

        Args:
            task_id: The task ID to cancel

        Returns:
            True if cancelled, False if task not found or not running
        """
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning(f"Cannot cancel task {task_id}: not found")
            return False

        if task.status != TaskStatus.RUNNING:
            logger.warning(f"Cannot cancel task {task_id}: status is {task.status.value}")
            return False

        if task._task:
            task._task.cancel()
            logger.info(f"Cancelled task: {task.name} (id={task_id})")
            return True

        return False

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_running_tasks(self) -> list[BackgroundTask]:
        """Get all currently running tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]

    def get_all_tasks(self, limit: int = 10) -> list[BackgroundTask]:
        """Get recent tasks (most recent first).

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of tasks sorted by start time (most recent first)
        """
        sorted_tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.started_at,
            reverse=True,
        )
        return sorted_tasks[:limit]

    def get_pending_notification(self) -> Optional[TaskNotification]:
        """Get the next pending notification (non-blocking).

        Returns:
            TaskNotification if available, None otherwise
        """
        try:
            return self._notification_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_notification(self, timeout: Optional[float] = None) -> Optional[TaskNotification]:
        """Get the next notification (blocking with optional timeout).

        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            TaskNotification if available within timeout, None otherwise
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    self._notification_queue.get(),
                    timeout=timeout,
                )
            else:
                return await self._notification_queue.get()
        except asyncio.TimeoutError:
            return None

    def has_pending_notifications(self) -> bool:
        """Check if there are pending notifications."""
        return not self._notification_queue.empty()

    def cleanup_old_tasks(self, max_age_seconds: float = 3600) -> int:
        """Remove completed/failed/cancelled tasks older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds for completed tasks

        Returns:
            Number of tasks removed
        """
        now = asyncio.get_event_loop().time()
        to_remove = []

        for task_id, task in self._tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task.completed_at and (now - task.completed_at) > max_age_seconds:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tasks")

        return len(to_remove)

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all tasks status.

        Returns:
            Dict with counts by status and list of running tasks
        """
        running = []
        counts = {status.value: 0 for status in TaskStatus}

        for task in self._tasks.values():
            counts[task.status.value] += 1
            if task.status == TaskStatus.RUNNING:
                elapsed = asyncio.get_event_loop().time() - task.started_at
                running.append({
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "elapsed_seconds": round(elapsed, 1),
                    "progress": task.progress,
                    "progress_message": task.progress_message,
                })

        return {
            "total": len(self._tasks),
            "counts": counts,
            "running": running,
        }
