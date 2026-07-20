from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.stop_emotion import StopEmotion


@pytest.mark.asyncio
async def test_stop_emotion_clears_queue() -> None:
    """Stopping an emotion clears the movement queue and reports it."""
    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    result = await StopEmotion()(deps)
    assert result == {"status": "stopped emotion and cleared queue"}
    movement_manager.clear_move_queue.assert_called_once_with()
