"""Tests for the get_time tool."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.get_time import GetTime
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_get_time_local() -> None:
    """Local time returns the expected fields and no error."""
    result = await GetTime()(_deps())

    assert "error" not in result
    assert set(result) >= {"iso", "date", "time", "weekday", "timezone", "summary"}


@pytest.mark.asyncio
async def test_get_time_with_timezone() -> None:
    """A valid IANA timezone is echoed back in the result."""
    result = await GetTime()(_deps(), timezone="Europe/Paris")

    assert result["timezone"] == "Europe/Paris"


@pytest.mark.asyncio
async def test_get_time_rejects_unknown_timezone() -> None:
    """An unknown timezone returns an error."""
    result = await GetTime()(_deps(), timezone="Mars/Olympus_Mons")

    assert "error" in result
