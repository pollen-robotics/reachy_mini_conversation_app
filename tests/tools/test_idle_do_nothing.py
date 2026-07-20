from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.idle_do_nothing import IdleDoNothing


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_idle_do_nothing_default_reason() -> None:
    """Without a reason the tool reports the default idle turn."""
    result = await IdleDoNothing()(_deps())
    assert result == {"status": "idle", "reason": "idle turn"}


@pytest.mark.asyncio
async def test_idle_do_nothing_custom_reason() -> None:
    """A provided reason is echoed back."""
    result = await IdleDoNothing()(_deps(), reason="user is thinking")
    assert result == {"status": "idle", "reason": "user is thinking"}
