"""Tool status tool - check status of background tools."""

import time
import logging
from typing import TYPE_CHECKING, Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.tool_constants import ToolState, SystemTool


if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundTool, BackgroundToolManager


logger = logging.getLogger(__name__)


def _describe(tool: "BackgroundTool") -> Dict[str, Any]:
    """Summarize one background task: what it is, what it was asked, how long it has run."""
    end = tool.completed_at or time.monotonic()
    info: Dict[str, Any] = {
        "tool_id": tool.tool_id,
        "name": tool.tool_name,
        "status": tool.status.value,
        "elapsed_seconds": round(end - tool.started_at, 1),
    }
    if tool.arguments:
        info["arguments"] = tool.arguments
    if tool.progress is not None:
        info["progress_percent"] = f"{tool.progress.progress:.0%}"
        if tool.progress.message:
            info["progress_message"] = tool.progress.message
    if tool.result:
        info["result"] = tool.result
    if tool.error:
        info["error"] = tool.error
    return info


class TaskStatus(Tool):
    """Check status of background tool tasks."""

    name = "task_status"
    description = (
        "Check the status of background tool tasks: the ones running, and the long ones finished in the last "
        "hour with their result. Use this when the user asks about running tools, what's happening in the "
        "background, or whether something they asked for earlier is ready."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "tool_id": {
                "type": "string",
                "description": "Specific tool ID to check (optional, lists running and recently finished tasks "
                "if omitted)",
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

            return _describe(tool)

        system_tool_names = {system_tool.value for system_tool in SystemTool}
        tasks = [tool for tool in tool_manager.get_all_tools() if tool.tool_name not in system_tool_names]
        running = [_describe(tool) for tool in tasks if tool.status == ToolState.RUNNING]
        # Short tools finish within the turn that called them; only long-running results are worth recalling.
        # The manager forgets finished tasks after an hour.
        finished = [_describe(tool) for tool in tasks if tool.status != ToolState.RUNNING and tool.is_long_running]
        if not running and not finished:
            return {
                "status": "idle",
                "message": "No tools running in the background.",
            }

        return {
            "status": "running" if running else "idle",
            "count": len(running),
            "message": f"{len(running)} tool(s) running in the background, {len(finished)} finished recently.",
            "tools": running,
            "finished": finished,
        }
