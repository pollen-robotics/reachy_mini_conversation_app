import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class IdleDoNothing(Tool):
    """Internal no-op selected by the local idle policy."""

    name = "idle_do_nothing"
    description = (
        "Internal idle action: keep Reachy still and silent for the current idle turn. "
        "This tool is selected by app code rather than exposed for normal model tool calling."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional internal reason for staying idle during this idle turn.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stay still and silent for the current idle turn."""
        reason = kwargs.get("reason", "idle turn")
        logger.info("Tool call: idle_do_nothing reason=%s", reason)
        return {"status": "idle", "reason": reason}
