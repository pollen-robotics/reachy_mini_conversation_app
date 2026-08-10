import logging
from typing import Any

from reachy_mini_conversation_app.companion.client import (
    CompanionClientError,
    companion_task_to_tool_result,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CompanionStart(Tool):
    """Start a durable delegated task."""

    name = "companion_start"
    description = (
        "Delegate a complete task that requires sustained, multi-step, or external work to the background assistant. "
        "Collect obvious missing constraints before calling. A queued response means work started, not finished. "
        "The user can keep talking, and Reachy will announce a later question or completed brief."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "minLength": 1,
                "description": "The complete delegated task, including its goal, constraints, and desired brief.",
            },
        },
        "required": ["request"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Start a companion task and return its durable state."""
        request = kwargs.get("request")
        if not isinstance(request, str) or not request.strip():
            logger.warning("companion_start: empty request")
            return {"error": "request must be a non-empty string"}
        if deps.companion_tasks is None:
            logger.warning("companion_start: companion assistant is not configured")
            return {"error": "The background assistant is not configured."}

        try:
            task = await deps.companion_tasks.start(request)
        except CompanionClientError as exc:
            logger.warning("companion_start: request rejected: %s", exc)
            return {"error": str(exc)}

        logger.info("Started companion task %s with status %s", task.task_id, task.status.value)
        return companion_task_to_tool_result(task)
