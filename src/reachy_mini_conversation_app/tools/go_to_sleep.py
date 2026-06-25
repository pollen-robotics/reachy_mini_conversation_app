import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GoToSleep(Tool):
    """Put Reachy to sleep and stop the current app."""

    name = "go_to_sleep"
    description = (
        "Use only when the user explicitly asks Reachy to go to sleep, stop the current app, shut down this app, "
        "or end the conversation. Do not use for idle turns, sleepy emotions, silence, or ambiguous requests."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "confirmation": {
                "type": "string",
                "enum": ["go_to_sleep"],
                "description": "Required exact confirmation for an explicit user request to stop this app.",
            },
        },
        "required": ["confirmation"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Put Reachy to sleep and request app shutdown."""
        if kwargs.get("confirmation") != "go_to_sleep":
            return {"error": "go_to_sleep requires explicit confirmation"}
        if deps.go_to_sleep is None:
            return {"error": "go_to_sleep is unavailable in this runtime"}

        logger.info("Tool call: go_to_sleep")
        try:
            return await asyncio.to_thread(deps.go_to_sleep)
        except Exception as e:
            logger.error("go_to_sleep failed: %s", e)
            return {"error": f"go_to_sleep failed: {type(e).__name__}: {e}"}
