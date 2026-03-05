"""Tool status tool - check status of background tools."""

import time
import logging
from typing import TYPE_CHECKING, Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.tool_constants import SystemTool


if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)


class TaskStatus(Tool):
    """Check status of background tool tasks."""

    name = "task_status"
    description = (
        "Check the status of background tool tasks. "
        "Use this when the user asks about running tools or wants to know what's happening in the background."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "tool_id": {
                "type": "string",
                "description": "Specific tool ID to check (optional, shows all running tools if omitted)",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Get status of background tools."""
        tool_id: str | None = kwargs.get("tool_id")
        tool_manager: BackgroundToolManager | None = kwargs.get("tool_manager")

        if tool_manager is None:
            return {"error": "Tool manager is required."}

        logger.info(f"Tool call: tool_status tool_id={tool_id}")

        if tool_id:
            tool = tool_manager.get_tool(tool_id)
            if not tool:
                return {"error": f"Tool {tool_id} not found."}

            result: Dict[str, Any] = {
                "tool_id": tool.tool_id,
                "name": tool.tool_name,
                "status": tool.status.value,
                "started_at": tool.started_at,
            }
            if tool.completed_at:
                result["completed_at"] = tool.completed_at

            if tool.progress is not None:
                result["progress_percent"] = f"{tool.progress.progress:.0%}"
                if tool.progress.message:
                    result["progress_message"] = tool.progress.message

            if tool.result:
                result["result"] = tool.result
            if tool.error:
                result["error"] = tool.error

            return result

        # Get all running tools
        running = tool_manager.get_running_tools()
        if not running:
            return {
                "status": "idle",
                "message": "No tools running in the background.",
            }

        tools_info = []
        for tool in [tool for tool in running if tool.tool_name not in [system_tool.value for system_tool in SystemTool]]:
            elapsed = time.monotonic() - tool.started_at
            tool_info: Dict[str, Any] = {
                "tool_id": tool.tool_id,
                "name": tool.tool_name,
                "status": tool.status.value,
                "elapsed_seconds": round(elapsed, 1),
            }

            # Add progress if tracking
            if tool.progress is not None:
                tool_info["progress_percent"] = f"{tool.progress.progress:.0%}"
                if tool.progress.message:
                    tool_info["progress_message"] = tool.progress.message

            tools_info.append(tool_info)

        return {
            "status": "running",
            "count": len(tools_info),
            "message": f"{len(tools_info)} tool(s) running in the background.",
            "tools": tools_info,
        }
