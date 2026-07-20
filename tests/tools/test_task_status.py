from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.task_status import TaskStatus
from reachy_mini_conversation_app.tools.tool_constants import ToolState
from reachy_mini_conversation_app.tools.background_tool_manager import ToolProgress, BackgroundTool


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _tool(name: str = "long_job", status: ToolState = ToolState.RUNNING, **kwargs: object) -> BackgroundTool:
    return BackgroundTool(id="1", tool_name=name, is_idle_tool_call=False, status=status, **kwargs)


@pytest.mark.asyncio
async def test_task_status_requires_tool_manager() -> None:
    """Without an injected tool manager the tool reports an error instead of crashing."""
    result = await TaskStatus()(_deps())
    assert result == {"error": "Tool manager is required."}


@pytest.mark.asyncio
async def test_task_status_unknown_tool_id() -> None:
    """Querying a missing tool id returns a not-found error."""
    manager = MagicMock()
    manager.get_tool.return_value = None
    result = await TaskStatus()(_deps(), tool_id="nope", tool_manager=manager)
    assert result == {"error": "Tool nope not found."}


@pytest.mark.asyncio
async def test_task_status_single_tool_formats_progress_and_result() -> None:
    """A specific tool query surfaces progress percent, message and result."""
    tool = _tool(progress=ToolProgress(progress=0.25, message="working"), result={"ok": True})
    manager = MagicMock()
    manager.get_tool.return_value = tool
    result = await TaskStatus()(_deps(), tool_id=tool.tool_id, tool_manager=manager)
    assert result["name"] == "long_job"
    assert result["status"] == "running"
    assert result["progress_percent"] == "25%"
    assert result["progress_message"] == "working"
    assert result["result"] == {"ok": True}


@pytest.mark.asyncio
async def test_task_status_no_running_tools_is_idle() -> None:
    """With nothing running the tool reports idle."""
    manager = MagicMock()
    manager.get_running_tools.return_value = []
    result = await TaskStatus()(_deps(), tool_manager=manager)
    assert result == {"status": "idle", "message": "No tools running in the background."}


@pytest.mark.asyncio
async def test_task_status_lists_running_and_filters_system_tools() -> None:
    """System tools (task_status/task_cancel) are excluded from the running list."""
    manager = MagicMock()
    manager.get_running_tools.return_value = [_tool("long_job"), _tool("task_status")]
    result = await TaskStatus()(_deps(), tool_manager=manager)
    assert result["status"] == "running"
    assert result["count"] == 1
    assert [tool["name"] for tool in result["tools"]] == ["long_job"]
