import os
import sys
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
import pytest

from reachy_mini_conversation_app.tools import get_time as get_time_module
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _host_timezone(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    instant = datetime(2026, 7, 1, 12, tzinfo=UTC)
    host_zone = get_time_module.ZoneInfo(name)
    host_clock = MagicMock(wraps=instant)
    host_clock.astimezone.side_effect = lambda zone=None: instant.astimezone(host_zone if zone is None else zone)
    monkeypatch.setattr(get_time_module, "datetime", MagicMock(now=MagicMock(return_value=host_clock)))
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(side_effect=AssertionError("Time queries must stay offline")))


@pytest.mark.asyncio
async def test_get_time_reports_an_explicit_timezone() -> None:
    """A named IANA timezone should be reported with its own offset."""
    result = await get_time_module.GetTime()(_deps(), timezone="Asia/Tokyo")

    assert result["timezone"] == "Asia/Tokyo"
    assert result["utc_offset_minutes"] == 540
    assert result["iso"].startswith(f"{result['date']}T{result['time']}")
    assert result["iso"].endswith("+09:00")


@pytest.mark.asyncio
async def test_get_time_compares_two_timezones() -> None:
    """The difference should be signed from the base timezone to the compared one."""
    result = await get_time_module.GetTime()(_deps(), timezone="Asia/Tokyo", compare_timezone="Asia/Kolkata")

    assert result["compare"]["timezone"] == "Asia/Kolkata"
    assert result["time_difference_minutes"] == -210
    assert result["time_difference_hours"] == -3.5
    assert result["time_difference_summary"] == "Asia/Kolkata is 3 hours 30 minutes behind Asia/Tokyo."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"timezone": "Mars/Olympus_Mons"},
        {"timezone": "Canada"},
        {"timezone": "Asia/Tokyo", "compare_timezone": "../etc/passwd"},
    ],
)
async def test_get_time_rejects_unknown_timezones(arguments: dict[str, str]) -> None:
    """An invalid timezone name should come back as a tool error."""
    result = await get_time_module.GetTime()(_deps(), **arguments)

    assert list(result) == ["error"]


@pytest.mark.asyncio
async def test_get_time_uses_the_effective_host_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The effective host clock should take precedence over the localtime symlink."""
    _host_timezone(monkeypatch, "Asia/Tokyo")
    monkeypatch.setattr(os, "readlink", lambda path: "/usr/share/zoneinfo/Europe/Zurich")

    result = await get_time_module.GetTime()(_deps())

    assert result["timezone"] == "JST"
    assert result["utc_offset_minutes"] == 540
    assert result["time"] == "21:00"
    assert "note" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host_timezone", "expected_name", "expected_offset"),
    [("UTC", "UTC", 0), ("GMT", "GMT", 0), ("Europe/London", "BST", 60)],
)
async def test_get_time_qualifies_possible_defaults_without_a_symlink(
    monkeypatch: pytest.MonkeyPatch,
    host_timezone: str,
    expected_name: str,
    expected_offset: int,
) -> None:
    """A copied localtime file should not hide a possibly unconfigured host timezone."""
    _host_timezone(monkeypatch, host_timezone)
    monkeypatch.setattr(os, "readlink", MagicMock(side_effect=OSError("not a symlink")))

    result = await get_time_module.GetTime()(_deps())

    assert result["timezone"] == expected_name
    assert result["utc_offset_minutes"] == expected_offset
    assert "may not be the user's local time" in result["note"]
    assert "no timezone set" not in result["note"]


@pytest.mark.asyncio
async def test_get_time_preserves_london_without_ip_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A London host should keep its own time even when it matches the wireless image default."""
    _host_timezone(monkeypatch, "Europe/London")
    monkeypatch.setattr(os, "readlink", lambda path: "/usr/share/zoneinfo/Europe/London")

    result = await get_time_module.GetTime()(_deps(), compare_timezone="Asia/Tokyo")

    assert result["timezone"] == "BST"
    assert result["time"] == "13:00"
    assert result["time_difference_minutes"] == 480
    assert "note" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("timezone", ["UTC", "Europe/London"])
async def test_get_time_does_not_qualify_explicit_timezones(timezone: str) -> None:
    """An explicitly requested timezone should not be treated as an uncertain host default."""
    result = await get_time_module.GetTime()(_deps(), timezone=timezone)

    assert result["timezone"] == timezone
    assert "note" not in result


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX TZ overrides are not supported on Windows")
async def test_get_time_honors_a_process_timezone_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process timezone override should apply without changing system configuration."""
    try:
        with monkeypatch.context() as environment:
            environment.setenv("TZ", "Asia/Tokyo")
            time.tzset()
            result = await get_time_module.GetTime()(_deps())
    finally:
        time.tzset()

    assert result["utc_offset_minutes"] == 540
    assert result["timezone"] == "JST"
