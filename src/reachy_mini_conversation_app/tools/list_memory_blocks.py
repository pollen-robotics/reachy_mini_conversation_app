"""List memory blocks tool for viewing all stored memory."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.memory import get_memory_manager
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ListMemoryBlocks(Tool):
    """List all memory blocks with content and metadata."""

    name = "list_memory_blocks"
    description = "List all memory blocks with their content and metadata. Use this to see what information is currently stored in memory."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute memory listing."""
        logger.info("Listing all memory blocks")

        try:
            manager = get_memory_manager()
            await manager.initialize()

            blocks = await manager.get_all_blocks()

            if not blocks:
                logger.info("No memory blocks found")
                return {
                    "status": "success",
                    "message": "No memory blocks found",
                    "blocks": [],
                }

            # Format blocks for LLM
            formatted_blocks = []
            for block in blocks:
                formatted_blocks.append({
                    "block_type": block.block_type,
                    "label": block.label,
                    "content": block.content,
                    "content_length": len(block.content),
                    "char_limit": block.char_limit,
                    "created_at": block.created_at.isoformat(),
                    "updated_at": block.updated_at.isoformat(),
                })

            logger.info("Found %d memory blocks", len(blocks))
            return {
                "status": "success",
                "count": len(blocks),
                "blocks": formatted_blocks,
            }

        except Exception as e:
            logger.exception("Failed to list memory blocks")
            return {"error": f"Failed to list memory: {str(e)}"}
