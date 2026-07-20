from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.task_cancel import TaskCancel
from reachy_mini_conversation_app.tools.tool_constants import ToolState
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundTool


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _tool(status: ToolState = ToolState.RUNNING) -> BackgroundTool:
    return BackgroundTool(id="1", tool_name="long_job", is_idle_tool_call=False, status=status)


@pytest.mark.asyncio
async def test_task_cancel_requires_tool_manager() -> None:
    """Without an injected tool manager the tool reports an error."""
    result = await TaskCancel()(_deps(), tool_id="x")
    assert result == {"error": "Tool manager is required."}


@pytest.mark.asyncio
async def test_task_cancel_requires_tool_id() -> None:
    """An empty tool id is rejected."""
    result = await TaskCancel()(_deps(), tool_id="", tool_manager=MagicMock())
    assert result == {"error": "Tool ID is required."}


@pytest.mark.asyncio
async def test_task_cancel_unknown_tool() -> None:
    """Cancelling a missing tool returns a not-found error."""
    manager = MagicMock()
    manager.get_tool.return_value = None
    result = await TaskCancel()(_deps(), tool_id="nope", tool_manager=manager)
    assert result == {"error": "Tool nope not found."}


@pytest.mark.asyncio
async def test_task_cancel_not_running() -> None:
    """A tool that already finished is reported, not cancelled."""
    manager = MagicMock()
    manager.get_tool.return_value = _tool(ToolState.COMPLETED)
    manager.cancel_tool = AsyncMock()
    result = await TaskCancel()(_deps(), tool_id="1", tool_manager=manager)
    assert result["status"] == "completed"
    manager.cancel_tool.assert_not_called()


@pytest.mark.asyncio
async def test_task_cancel_success() -> None:
    """A running tool that cancels cleanly returns cancelled status."""
    manager = MagicMock()
    manager.get_tool.return_value = _tool(ToolState.RUNNING)
    manager.cancel_tool = AsyncMock(return_value=True)
    result = await TaskCancel()(_deps(), tool_id="1", tool_manager=manager)
    assert result["status"] == "cancelled"
    manager.cancel_tool.assert_awaited_once_with("1")


@pytest.mark.asyncio
async def test_task_cancel_failure() -> None:
    """A running tool that fails to cancel returns an error."""
    manager = MagicMock()
    manager.get_tool.return_value = _tool(ToolState.RUNNING)
    manager.cancel_tool = AsyncMock(return_value=False)
    result = await TaskCancel()(_deps(), tool_id="1", tool_manager=manager)
    assert "error" in result
