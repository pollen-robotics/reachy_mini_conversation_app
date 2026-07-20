from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.move_head import MoveHead
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


def _deps() -> ToolDependencies:
    reachy_mini = MagicMock()
    reachy_mini.get_current_joint_positions.return_value = (0.0, (0.1, 0.2))
    return ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_move_head_rejects_non_string_direction() -> None:
    """A non-string direction is rejected without touching the robot."""
    result = await MoveHead()(_deps(), direction=42)
    assert result == {"error": "direction must be a string"}


@pytest.mark.asyncio
async def test_move_head_queues_goto_move() -> None:
    """A valid direction queues a GotoQueueMove and marks the robot moving."""
    deps = _deps()
    result = await MoveHead()(deps, direction="left")
    assert result == {"status": "looking left"}
    queued_move = deps.movement_manager.queue_move.call_args.args[0]
    assert isinstance(queued_move, GotoQueueMove)
    deps.movement_manager.set_moving_state.assert_called_once_with(deps.motion_duration_s)


@pytest.mark.asyncio
async def test_move_head_reports_robot_failure() -> None:
    """A robot read failure is returned as an error, not raised into the loop."""
    deps = _deps()
    deps.reachy_mini.get_current_head_pose.side_effect = RuntimeError("boom")
    result = await MoveHead()(deps, direction="up")
    assert "error" in result
    assert "RuntimeError" in result["error"]
