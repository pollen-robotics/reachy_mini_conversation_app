import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class StopDance(Tool):
    """Stop the current dance move."""

    name = "stop_dance"
    description = "Stop the current dance move"
    parameters_schema = {
        "type": "object",
        "properties": {
            "dummy": {
                "type": "boolean",
                "description": "dummy boolean, set it to true",
            },
        },
        "required": ["dummy"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stop the current dance move."""
        logger.info("Tool call: stop_dance")
        movement_manager = deps.movement_manager
        movement_manager.clear_move_queue()
        return {"status": "stopped dance and cleared queue"}
