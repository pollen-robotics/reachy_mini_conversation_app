"""Tool cancel tool - cancel running background tools."""

import logging
from typing import TYPE_CHECKING, Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.tool_constants import ToolState


if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)


class TaskCancel(Tool):
    """Cancel a running background tool task."""

    name = "task_cancel"
    description = (
        "Cancel a running background tool task. "
        "Use this when the user wants to stop a tool that's running in the background. "
        "Requires confirmation before cancelling."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "tool_id": {
                "type": "string",
                "description": "The tool ID to cancel",
            }
        },
        "required": ["tool_id"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a background tool."""
        tool_id = kwargs.get("tool_id", "")
        tool_manager: BackgroundToolManager | None = kwargs.get("tool_manager")

        if tool_manager is None:
            return {"error": "Tool manager is required."}

        logger.info(f"Tool call: tool_cancel tool_id={tool_id}")

        if not tool_id:
            return {"error": "Tool ID is required."}

        tool = tool_manager.get_tool(tool_id)

        if not tool:
            return {"error": f"Tool {tool_id} not found."}

        # Check if tool is still running
        if tool.status != ToolState.RUNNING:
            return {
                "status": f"{tool.status.value}",
                "message": f"Tool '{tool.tool_name}' is not running (status: {tool.status.value}).",
                "tool_id": tool_id,
            }

        # Cancel the tool
        if await tool_manager.cancel_tool(tool_id):
            return {
                "status": "cancelled",
                "message": f"Tool '{tool.tool_name}' has been cancelled.",
                "tool_id": tool_id,
                "tool_name": tool.tool_name,
            }
        else:
            return {
                "error": f"Could not cancel tool {tool_id}. It may have already completed.",
            }
