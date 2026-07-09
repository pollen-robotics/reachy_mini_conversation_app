"""Tests for the get_time tool."""

import logging
import threading
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.get_time as get_time
from reachy_mini_conversation_app.tools.get_time import GetTime
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class _FakeResponse:
    def __init__(self, text: str = "", json_payload: dict[str, object] | None = None) -> None:
        self.text = text
        self._json_payload = json_payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        if self._json_payload is None:
            raise ValueError("no json")
        return self._json_payload


class _FakeGeoIpLookup:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.urls: list[str] = []

    def __call__(self, url: str, *, timeout: float) -> _FakeResponse:
        self.urls.append(url)
        return self._response


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


async def _warm_tool(monkeypatch: pytest.MonkeyPatch, tool: GetTime, timezone_name: str) -> None:
    monkeypatch.setattr(get_time.httpx, "get", _FakeGeoIpLookup(_FakeResponse(timezone_name)))
    monkeypatch.setattr(get_time, "GEOIP_WARMUP_WAIT_S", 1.0)
    result = await tool(_deps())
    assert result["timezone"] == timezone_name


def test_get_time_schema_requires_timezone_argument() -> None:
    """The model must explicitly choose local time or a requested timezone."""
    spec = GetTime().spec()

    assert spec["parameters"]["required"] == ["timezone"]
    assert "empty string" in spec["parameters"]["properties"]["timezone"]["description"]
    assert "do not ask" in spec["parameters"]["properties"]["timezone"]["description"]
    assert "compare_timezone" in spec["parameters"]["properties"]


def test_warm_local_timezone_cache_starts_single_background_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local timezone warmup should not block startup."""
    tool = GetTime()
    threads: list[dict[str, object]] = []

    class _FakeThread:
        def __init__(self, *, target: object, daemon: bool, name: str) -> None:
            threads.append({"target": target, "daemon": daemon, "name": name, "started": False})

        def start(self) -> None:
            threads[-1]["started"] = True

    monkeypatch.setattr(get_time.threading, "Thread", _FakeThread)

    tool.warm_local_timezone_cache()
    tool.warm_local_timezone_cache()

    assert len(threads) == 1
    assert threads[0]["daemon"] is True
    assert threads[0]["name"] == "get-time-geoip"
    assert threads[0]["started"] is True


@pytest.mark.asyncio
async def test_get_time_local_uses_geoip_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local time is resolved from the robot's public IP."""
    tool = GetTime()
    fake_lookup = _FakeGeoIpLookup(_FakeResponse("Europe/Paris"))
    monkeypatch.setattr(get_time.httpx, "get", fake_lookup)
    monkeypatch.setattr(get_time, "GEOIP_WARMUP_WAIT_S", 1.0)

    result = await tool(_deps())
    cached_result = await tool(_deps())

    assert "error" not in result
    assert set(result) >= {"iso", "date", "time", "weekday", "timezone", "summary"}
    assert result["timezone"] == "Europe/Paris"
    assert cached_result["timezone"] == "Europe/Paris"
    assert fake_lookup.urls == [get_time.GEOIP_TIMEZONE_ENDPOINTS[0][0]]


@pytest.mark.asyncio
async def test_get_time_local_uses_second_geoip_provider_after_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rate-limited primary geo-IP provider should not break local time."""
    tool = GetTime()
    urls: list[str] = []

    class _RateLimitedResponse(_FakeResponse):
        def raise_for_status(self) -> None:
            request = get_time.httpx.Request("GET", "https://ipapi.co/timezone/")
            response = get_time.httpx.Response(429, request=request)
            raise get_time.httpx.HTTPStatusError("rate limited", request=request, response=response)

    def _lookup(url: str, *, timeout: float) -> _FakeResponse:
        urls.append(url)
        if len(urls) == 1:
            return _RateLimitedResponse()
        return _FakeResponse(json_payload={"timezone": "Europe/Paris"})

    monkeypatch.setattr(get_time.httpx, "get", _lookup)
    monkeypatch.setattr(get_time, "GEOIP_WARMUP_WAIT_S", 1.0)

    result = await tool(_deps())

    assert result["timezone"] == "Europe/Paris"
    assert urls == [endpoint[0] for endpoint in get_time.GEOIP_TIMEZONE_ENDPOINTS]


@pytest.mark.asyncio
async def test_get_time_local_returns_error_while_geoip_lookup_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local time does not block on geo-IP when the cache is not ready."""
    tool = GetTime()
    lookup_started = threading.Event()
    release_lookup = threading.Event()
    lookup_finished = threading.Event()

    def _slow_lookup(url: str, *, timeout: float) -> _FakeResponse:
        lookup_started.set()
        release_lookup.wait(timeout=1.0)
        lookup_finished.set()
        return _FakeResponse("Europe/Paris")

    monkeypatch.setattr(get_time.httpx, "get", _slow_lookup)
    monkeypatch.setattr(get_time, "GEOIP_WARMUP_WAIT_S", 0.001)

    result = await tool(_deps())

    assert result == {"error": "local timezone unavailable; geo-IP detection failed"}
    assert lookup_started.wait(timeout=1.0)
    release_lookup.set()
    assert lookup_finished.wait(timeout=1.0)


@pytest.mark.asyncio
async def test_get_time_local_uses_cached_timezone_without_geoip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm local timezone cache keeps the foreground tool path instant."""
    tool = GetTime()
    await _warm_tool(monkeypatch, tool, "Europe/Paris")

    def _unexpected_lookup(url: str, *, timeout: float) -> _FakeResponse:
        raise AssertionError("cached local time should not call geo-IP")

    monkeypatch.setattr(get_time.httpx, "get", _unexpected_lookup)

    result = await tool(_deps())

    assert result["timezone"] == "Europe/Paris"


@pytest.mark.asyncio
async def test_get_time_local_reports_error_when_timezone_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Geo-IP failures fail clearly instead of returning unreliable OS local time."""
    tool = GetTime()

    def _failing_lookup(url: str, *, timeout: float) -> _FakeResponse:
        raise get_time.httpx.ConnectError("offline")

    monkeypatch.setattr(get_time.httpx, "get", _failing_lookup)
    monkeypatch.setattr(get_time, "GEOIP_WARMUP_WAIT_S", 1.0)
    caplog.set_level(logging.WARNING, logger=get_time.__name__)

    result = await tool(_deps())

    assert result == {"error": "local timezone unavailable; geo-IP detection failed"}
    assert any("Failed to resolve timezone from geo-IP" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_get_time_with_timezone() -> None:
    """A valid IANA timezone is echoed back in the result."""
    result = await GetTime()(_deps(), timezone="Asia/Tokyo")

    assert result["timezone"] == "Asia/Tokyo"


@pytest.mark.asyncio
async def test_get_time_with_timezone_does_not_use_geoip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requested IANA timezone is resolved without local geo-IP."""
    tool = GetTime()

    def _unexpected_lookup(url: str, *, timeout: float) -> _FakeResponse:
        raise AssertionError("explicit timezone should not call geo-IP")

    monkeypatch.setattr(get_time.httpx, "get", _unexpected_lookup)

    result = await tool(_deps(), timezone="Asia/Tokyo", compare_timezone="")

    assert result["timezone"] == "Asia/Tokyo"
    assert "compare" not in result


@pytest.mark.asyncio
async def test_get_time_compares_local_time_with_requested_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Time differences are returned by the tool instead of left to model arithmetic."""
    tool = GetTime()
    await _warm_tool(monkeypatch, tool, "Europe/Paris")

    result = await tool(_deps(), timezone="", compare_timezone="Asia/Tokyo")

    assert result["timezone"] == "Europe/Paris"
    assert result["compare"]["timezone"] == "Asia/Tokyo"
    assert result["time_difference_minutes"] == (
        result["compare"]["utc_offset_minutes"] - result["utc_offset_minutes"]
    )
    assert "Asia/Tokyo" in result["time_difference_summary"]


@pytest.mark.asyncio
async def test_get_time_rejects_unknown_timezone() -> None:
    """An unknown timezone returns an error."""
    result = await GetTime()(_deps(), timezone="Mars/Olympus_Mons")

    assert "error" in result
