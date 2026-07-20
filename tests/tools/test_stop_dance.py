from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.stop_dance import StopDance


@pytest.mark.asyncio
async def test_stop_dance_clears_queue() -> None:
    """Stopping a dance clears the movement queue and reports it."""
    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    result = await StopDance()(deps)
    assert result == {"status": "stopped dance and cleared queue"}
    movement_manager.clear_move_queue.assert_called_once_with()
