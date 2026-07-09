"""Tests for the get_weather tool."""

from typing import Any
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.get_weather as get_weather
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.get_weather import GetWeather


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        return self._responses.pop(0)


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_get_weather_returns_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolved location returns a compact weather brief."""
    geocode = _FakeResponse(
        {
            "results": [
                {"name": "Paris", "admin1": "Île-de-France", "country": "France", "latitude": 48.85, "longitude": 2.35}
            ]
        }
    )
    forecast = _FakeResponse(
        {
            "current": {"temperature_2m": 18.0, "weather_code": 2},
            "daily": {
                "temperature_2m_max": [22.0],
                "temperature_2m_min": [12.0],
                "precipitation_probability_max": [10],
            },
        }
    )
    monkeypatch.setattr(get_weather.httpx, "AsyncClient", lambda **kwargs: _FakeClient([geocode, forecast]))

    result = await GetWeather()(_deps(), location="Paris")

    assert result["location"] == "Paris, Île-de-France, France"
    assert result["conditions"] == "partly cloudy"
    assert result["temperature_c"] == 18.0
    assert result["high_c"] == 22.0
    assert result["low_c"] == 12.0
    assert result["rain_chance_pct"] == 10


@pytest.mark.asyncio
async def test_get_weather_rejects_empty_location() -> None:
    """An empty location returns an error."""
    result = await GetWeather()(_deps(), location="")

    assert "error" in result


@pytest.mark.asyncio
async def test_get_weather_reports_unknown_location(monkeypatch: pytest.MonkeyPatch) -> None:
    """A location with no geocoding match returns an error."""
    monkeypatch.setattr(
        get_weather.httpx, "AsyncClient", lambda **kwargs: _FakeClient([_FakeResponse({"results": []})])
    )

    result = await GetWeather()(_deps(), location="Nowherecity")

    assert "error" in result


@pytest.mark.asyncio
async def test_get_weather_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 on the first request is retried and the brief still returns."""
    monkeypatch.setattr(get_weather, "RETRY_DELAY_S", 0)
    rate_limited = _FakeResponse({}, status_code=429)
    geocode = _FakeResponse(
        {"results": [{"name": "Paris", "country": "France", "latitude": 48.85, "longitude": 2.35}]}
    )
    forecast = _FakeResponse(
        {
            "current": {"temperature_2m": 18.0, "weather_code": 0},
            "daily": {
                "temperature_2m_max": [22.0],
                "temperature_2m_min": [12.0],
                "precipitation_probability_max": [10],
            },
        }
    )
    monkeypatch.setattr(
        get_weather.httpx, "AsyncClient", lambda **kwargs: _FakeClient([rate_limited, geocode, forecast])
    )

    result = await GetWeather()(_deps(), location="Paris")

    assert result["location"] == "Paris, France"
