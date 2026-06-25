from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.go_to_sleep import GoToSleep


@pytest.mark.asyncio
async def test_go_to_sleep_requires_confirmation() -> None:
    """The tool should not run without the explicit confirmation token."""
    go_to_sleep = MagicMock(return_value={"status": "sleeping"})
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = await GoToSleep()(deps)

    assert result == {"error": "go_to_sleep requires explicit confirmation"}
    go_to_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_go_to_sleep_returns_unavailable_without_runtime_callback() -> None:
    """The tool should fail gracefully if the runtime did not inject a sleep callback."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = await GoToSleep()(deps, confirmation="go_to_sleep")

    assert result == {"error": "go_to_sleep is unavailable in this runtime"}


@pytest.mark.asyncio
async def test_go_to_sleep_calls_runtime_callback() -> None:
    """The tool should delegate the actual movement and app stop to the host runtime."""
    expected = {
        "status": "sleeping",
        "stop_current_app_requested": True,
        "local_stop_requested": True,
    }
    go_to_sleep = MagicMock(return_value=expected)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = await GoToSleep()(deps, confirmation="go_to_sleep")

    assert result == expected
    go_to_sleep.assert_called_once_with()
