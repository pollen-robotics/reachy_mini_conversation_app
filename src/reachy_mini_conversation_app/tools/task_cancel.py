"""Task cancel tool - cancel running background tasks."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.background_tasks import (
    BackgroundTaskManager,
    TaskStatus,
)


logger = logging.getLogger(__name__)


class TaskCancelTool(Tool):
    """Cancel a running background task."""

    name = "task_cancel"
    description = (
        "Cancel a running background task. "
        "Use this when the user wants to stop a task that's running in the background. "
        "Requires confirmation before cancelling."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to cancel",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm cancellation. Always ask the user for confirmation first.",
            },
        },
        "required": ["task_id", "confirmed"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a background task."""
        task_id = kwargs.get("task_id", "")
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: task_cancel task_id={task_id} confirmed={confirmed}")

        if not task_id:
            return {"error": "Task ID is required."}

        manager = BackgroundTaskManager.get_instance()
        task = manager.get_task(task_id)

        if not task:
            return {"error": f"Task {task_id} not found."}

        # Check if task is still running
        if task.status != TaskStatus.RUNNING:
            return {
                "status": "not_running",
                "message": f"Task '{task.name}' is not running (status: {task.status.value}).",
                "task_id": task_id,
            }

        # Require confirmation
        if not confirmed:
            return {
                "status": "confirmation_required",
                "message": f"Are you sure you want to cancel the task '{task.name}'?",
                "task_id": task_id,
                "task_name": task.name,
                "task_description": task.description,
                "hint": "Set confirmed=true after user approval to proceed with cancellation.",
            }

        # Cancel the task
        if await manager.cancel_task(task_id):
            return {
                "status": "cancelled",
                "message": f"Task '{task.name}' has been cancelled.",
                "task_id": task_id,
                "task_name": task.name,
            }
        else:
            return {
                "error": f"Could not cancel task {task_id}. It may have already completed.",
            }
