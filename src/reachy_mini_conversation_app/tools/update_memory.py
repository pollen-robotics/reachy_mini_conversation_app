"""Update memory tool for modifying memory blocks."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.memory import get_memory_manager
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class UpdateMemory(Tool):
    """Update a memory block (user/facts/robot)."""

    name = "update_memory"
    description = "Update a memory block to save important information. Use 'user' for user preferences, 'facts' for conversation facts, 'robot' for robot-specific context (requires label)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "block_type": {
                "type": "string",
                "enum": ["user", "facts", "robot"],
                "description": "Type of memory block to update",
            },
            "content": {
                "type": "string",
                "description": "The content to store in the memory block",
            },
            "label": {
                "type": "string",
                "description": "Label for robot blocks (e.g., 'greeting_style'). Required for robot blocks, ignored for user/facts.",
            },
        },
        "required": ["block_type", "content"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute memory update."""
        block_type = kwargs.get("block_type")
        content = kwargs.get("content")
        label = kwargs.get("label")

        if not block_type or not content:
            return {"error": "block_type and content are required"}

        # Validate block_type
        if block_type not in ["user", "facts", "robot"]:
            return {"error": "block_type must be 'user', 'facts', or 'robot'"}

        # Validate robot block has label
        if block_type == "robot" and not label:
            return {"error": "Robot blocks require a 'label' parameter"}

        logger.info("Updating memory: block_type=%s, label=%s", block_type, label)

        try:
            manager = get_memory_manager()
            await manager.initialize()

            # Update the block
            block = await manager.update_block(block_type, content, label)

            logger.info("Memory updated successfully: %s", block.label)
            return {
                "status": "success",
                "message": f"Memory block '{block.label}' updated successfully",
                "block_type": block.block_type,
                "label": block.label,
                "content_length": len(block.content),
                "char_limit": block.char_limit,
            }

        except ValueError as e:
            logger.warning("Memory update validation failed: %s", e)
            return {"error": str(e)}
        except Exception as e:
            logger.exception("Memory update failed")
            return {"error": f"Update failed: {str(e)}"}
