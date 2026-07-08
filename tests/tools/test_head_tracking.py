from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.head_tracking import HeadTracking


@pytest.mark.asyncio
async def test_head_tracking_enables_and_disables() -> None:
    """The tool forwards the toggle to the movement manager."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = await HeadTracking()(deps, enabled=True)
    deps.movement_manager.set_head_tracking.assert_called_with(True)
    assert result == {"status": "following"}

    result = await HeadTracking()(deps, enabled=False)
    deps.movement_manager.set_head_tracking.assert_called_with(False)
    assert result == {"status": "stopped following"}
