"""Example of a demo built using a custom tool and custom instructions."""

import logging
from typing import Any, Dict

# Import a tool from the tools library: use an alias.
from tools_library.sweep_look import SweepLook as SweepLook
from reachy_mini_conversation_app.tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


# Create a custom tool
class ThrowStone(Tool):
    """Example of custom tool call."""

    name = "throw_stone"
    description = "Throw a stone."
    parameters_schema = {
        "type": "object",
        "properties": {
            "stone_type": {
                "type": "string",
                "description": "Optional type of stone to be thrown.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute the throw_stone tool."""
        stone_type = kwargs.get("stone_type", "Default stone")
        logger.info(f"🥌 Throwing stone of type {stone_type}")
        return {"status": "A stone has been thrown", "stone_type": stone_type}
