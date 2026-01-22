"""
Custom tool example - can be loaded from outside the reachy_mini_conversation_app library.

To use this tool, set these environment variables:
    export REACHY_MINI_CUSTOM_PROFILE=custom_profile
    export PROFILES_DIRECTORY=/path/to/custom_profiles_and_tools
    export TOOLS_DIRECTORY=/path/to/custom_profiles_and_tools/custom_tool
    
Or add them to your .env file.
"""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class NewCustomTool(Tool):
    """Placeholder custom tool - demonstrates external tool loading."""

    name = "new_custom_tool"
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
        """Placeholder tool execution."""
        message = kwargs.get("message", "Hello from custom tool!")
        logger.info(f"Tool call: custom_greeting message={message}")

        return {"status": "success", "message": message}
