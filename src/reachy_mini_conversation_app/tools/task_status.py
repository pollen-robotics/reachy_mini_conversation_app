"""Task status tool - check status of background tasks."""

import time
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.background_tasks import (
    TaskStatus,
    BackgroundTaskManager,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class TaskStatusTool(Tool):
    """Check status of background tasks."""

    name = "task_status"
    description = (
        "Check the status of background tasks. "
        "Use this when the user asks about running tasks or wants to know what's happening in the background."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Specific task ID to check (optional, shows all running tasks if omitted)",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Get status of background tasks."""
        task_id = kwargs.get("task_id")
        manager = BackgroundTaskManager.get_instance()

        logger.info(f"Tool call: task_status task_id={task_id}")

        if task_id:
            # Get specific task
            task = manager.get_task(task_id)
            if not task:
                return {"error": f"Task {task_id} not found."}

            elapsed = time.monotonic() - task.started_at
            result: Dict[str, Any] = {
                "task_id": task.id,
                "name": task.name,
                "display_name": task.display_name,
                "description": task.description,
                "status": task.status.value,
                "elapsed_seconds": round(elapsed, 1),
            }

            # Add progress if tracking
            if task.progress is not None:
                result["progress"] = round(task.progress * 100, 1)
                result["progress_percent"] = f"{task.progress:.0%}"
                if task.progress_message:
                    result["progress_message"] = task.progress_message

            # Add result/error for completed tasks
            if task.status == TaskStatus.COMPLETED and task.result:
                result["result"] = task.result
            if task.status == TaskStatus.FAILED and task.error:
                result["error"] = task.error

            return result

        # Get all running tasks
        running = manager.get_running_tasks()
        if not running:
            return {
                "status": "idle",
                "message": "No tasks running in the background.",
            }

        tasks_info = []
        for task in running:
            elapsed = time.monotonic() - task.started_at
            task_info: Dict[str, Any] = {
                "task_id": task.id,
                "name": task.name,
                "display_name": task.display_name,
                "description": task.description,
                "elapsed_seconds": round(elapsed, 1),
            }

            # Add progress if tracking
            if task.progress is not None:
                task_info["progress"] = round(task.progress * 100, 1)
                task_info["progress_percent"] = f"{task.progress:.0%}"
                if task.progress_message:
                    task_info["progress_message"] = task.progress_message

            tasks_info.append(task_info)

        return {
            "status": "running",
            "count": len(tasks_info),
            "message": f"{len(tasks_info)} task(s) running in the background.",
            "tasks": tasks_info,
        }
