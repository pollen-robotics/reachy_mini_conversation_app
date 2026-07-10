import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class HeadTracking(Tool):
    """Enable or disable following the user's face with the head."""

    name = "head_tracking"
    description = (
        "Enable or disable following the user's face with the head. "
        "Use when asked to follow, keep looking at, or stop following the user."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "True to start following the user's face, false to stop.",
            },
        },
        "required": ["enabled"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Toggle head tracking."""
        enabled = bool(kwargs.get("enabled", True))
        logger.info("Tool call: head_tracking enabled=%s", enabled)
        deps.movement_manager.set_head_tracking(enabled)
        return {"status": "following" if enabled else "stopped following"}
