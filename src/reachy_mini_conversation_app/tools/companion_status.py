import logging
from typing import Any

from reachy_mini_conversation_app.companion.client import (
    CompanionClientError,
    companion_task_to_tool_result,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CompanionStatus(Tool):
    """Read the latest durable companion task state."""

    name = "companion_status"
    description = (
        "Check a background assistant task when the user asks for its progress or result status. "
        "Pass its task_id when known; omit it to check the most recent task. "
        "Summarize the returned state briefly and never claim queued or running work is finished. "
        "If the status is completed, say the background task finished; if failed, say the background task failed. "
        "If the status is input_required, ask the returned question; after the user answers, call "
        "companion_answer with the returned task_id and question_id."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The durable task identifier. Omit for the most recent task.",
            },
        },
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return the latest state of a companion task."""
        task_id = kwargs.get("task_id")
        if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
            logger.warning("companion_status: invalid task ID")
            return {"error": "task_id must be a non-empty string when provided"}
        if deps.companion_tasks is None:
            logger.warning("companion_status: companion assistant is not configured")
            return {"error": "The background assistant is not configured."}

        try:
            task = await deps.companion_tasks.get(task_id)
        except (CompanionClientError, ValueError) as exc:
            logger.warning("companion_status: request rejected: %s", exc)
            return {"error": str(exc)}
        return companion_task_to_tool_result(task)
