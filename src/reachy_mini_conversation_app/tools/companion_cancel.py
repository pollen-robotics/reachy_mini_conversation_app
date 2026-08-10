import logging
from typing import Any

from reachy_mini_conversation_app.companion.client import (
    CompanionClientError,
    companion_task_to_tool_result,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CompanionCancel(Tool):
    """Cancel a durable companion task."""

    name = "companion_cancel"
    description = (
        "Cancel a queued, running, or waiting background assistant task only when the user clearly requests it. "
        "Use the exact task_id and report the returned state briefly."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "minLength": 1,
                "description": "The durable task identifier to cancel.",
            },
        },
        "required": ["task_id"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Cancel a companion task and return its durable state."""
        task_id = kwargs.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            logger.warning("companion_cancel: empty task ID")
            return {"error": "task_id must be a non-empty string"}
        if deps.companion_tasks is None:
            logger.warning("companion_cancel: companion assistant is not configured")
            return {"error": "The background assistant is not configured."}

        try:
            task = await deps.companion_tasks.cancel(task_id)
        except CompanionClientError as exc:
            logger.warning("companion_cancel: request rejected: %s", exc)
            return {"error": str(exc)}
        logger.info("Cancelled companion task %s", task.task_id)
        return companion_task_to_tool_result(task)
