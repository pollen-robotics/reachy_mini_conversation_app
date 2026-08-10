import logging
from typing import Any

from reachy_mini_conversation_app.companion.client import (
    CompanionClientError,
    companion_task_to_tool_result,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CompanionAnswer(Tool):
    """Answer a pending question for a durable companion task."""

    name = "companion_answer"
    description = (
        "Submit the user's non-sensitive answer to a pending background assistant question. "
        "Use the exact task_id and question_id supplied by the background task notice. "
        "The answer queues the next worker phase; it does not authorize an external write action. "
        "Do not call this until the user has answered."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "minLength": 1,
                "description": "The durable task identifier returned by companion_start.",
            },
            "question_id": {
                "type": "string",
                "minLength": 1,
                "description": "The question identifier supplied by the latest background task notice.",
            },
            "answer": {
                "type": "string",
                "minLength": 1,
                "description": "The user's answer to the pending question.",
            },
        },
        "required": ["task_id", "question_id", "answer"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Answer a companion question and return the resulting durable state."""
        task_id = kwargs.get("task_id")
        question_id = kwargs.get("question_id")
        answer = kwargs.get("answer")
        if not isinstance(task_id, str) or not task_id.strip():
            logger.warning("companion_answer: empty task ID")
            return {"error": "task_id must be a non-empty string"}
        if not isinstance(question_id, str) or not question_id.strip():
            logger.warning("companion_answer: empty question ID")
            return {"error": "question_id must be a non-empty string"}
        if not isinstance(answer, str) or not answer.strip():
            logger.warning("companion_answer: empty answer")
            return {"error": "answer must be a non-empty string"}
        if deps.companion_tasks is None:
            logger.warning("companion_answer: companion assistant is not configured")
            return {"error": "The background assistant is not configured."}

        try:
            task = await deps.companion_tasks.answer(task_id, question_id, answer)
        except CompanionClientError as exc:
            logger.warning("companion_answer: request rejected: %s", exc)
            return {"error": str(exc)}

        logger.info("Queued companion task %s after user input", task.task_id)
        return companion_task_to_tool_result(task)
