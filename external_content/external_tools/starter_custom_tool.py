"""Example external tool implementation."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class StarterCustomTool(Tool):
    """Placeholder custom tool - demonstrates external tool loading."""

    name = "starter_custom_tool"
    description = "A placeholder custom tool loaded from outside the library"
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Optional message to include in the response",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute the placeholder tool."""
        message = kwargs.get("message", "Hello from custom tool!")
        logger.info(f"Tool call: starter_custom_tool message={message}")

        return {"status": "success", "message": message}
