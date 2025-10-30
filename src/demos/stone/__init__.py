"""Example of a demo.

The demo is built using a tool and a piece of prompt from the libraries,
along with a custom tool and custom instructions.

"""

import logging
from typing import Any, Dict

# Build the instructions by importing a prompt piece (witty identity) from the prompt library
from prompts_library import witty_identity

# Import a tool from the tools library: use an alias.
from tools_library.sweep_look import SweepLook as SweepLook
from reachy_mini_conversation_app.tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


# Start from a prompt imported from the library
instructions = witty_identity.prompt

# Then add custom instructions.
instructions += """
You love throwing stones but you try not to be seen by others when doing it.
You use the sweep_look tool when user asks you to check your surroundings.
You use the throw_stone tool when user asks you to throw a stone.
You love telling jokes about lobsters.
"""


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
