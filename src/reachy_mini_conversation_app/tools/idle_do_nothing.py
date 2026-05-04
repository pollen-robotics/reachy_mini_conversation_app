import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class IdleDoNothing(Tool):
    """Explicitly choose no action during an idle turn."""

    name = "idle_do_nothing"
    description = (
        "Use only in response to an idle time update when you intentionally want Reachy to stay still and silent "
        "instead of choosing another idle action."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional reason for staying idle during this idle turn.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stay still and silent for the current idle turn."""
        reason = kwargs.get("reason", "idle turn")
        logger.info("Tool call: idle_do_nothing reason=%s", reason)
        return {"status": "idle", "reason": reason}
