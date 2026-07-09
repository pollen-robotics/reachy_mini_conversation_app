"""Weather tool backed by the Open-Meteo API."""

import asyncio
import logging
from typing import Any

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 8.0
RETRY_DELAY_S = 0.5
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

WMO_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


class GetWeather(Tool):
    """Get today's weather brief for a location via Open-Meteo."""

    name = "get_weather"
    description = "Get today's weather for a place: current conditions, high and low temperature, and rain chance."
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City, region, or postal code."},
        },
        "required": ["location"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Resolve the location and return a compact weather brief for today."""
        location = (kwargs.get("location") or "").strip()
        if not location:
            return {"error": "location must be a non-empty string"}

        logger.info("get_weather location=%s", location)
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            geocode = await _get_json(client, GEOCODING_URL, {"name": location, "count": 1, "language": "en"})
            matches = geocode.get("results") or []
            if not matches:
                return {"error": f"no location match for '{location}'"}
            place = matches[0]
            data = await _get_json(
                client,
                FORECAST_URL,
                {
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "forecast_days": 1,
                    "timezone": "auto",
                },
            )

        current = data.get("current") or {}
        daily = data.get("daily") or {}
        code = current.get("weather_code")
        conditions = (
            WMO_DESCRIPTIONS.get(code, "unknown conditions") if isinstance(code, int) else "unknown conditions"
        )
        label = ", ".join(part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part)
        now_c = current.get("temperature_2m")
        high_c = (daily.get("temperature_2m_max") or [None])[0]
        low_c = (daily.get("temperature_2m_min") or [None])[0]
        rain_pct = (daily.get("precipitation_probability_max") or [None])[0]
        return {
            "location": label,
            "conditions": conditions,
            "temperature_c": now_c,
            "high_c": high_c,
            "low_c": low_c,
            "rain_chance_pct": rain_pct,
            "summary": f"{label}: {conditions}, {now_c}°C now, high {high_c}°C, low {low_c}°C, rain chance {rain_pct}%.",
        }


async def _get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> Any:
    response = await client.get(url, params=params)
    if response.status_code in RETRYABLE_STATUS:
        await asyncio.sleep(RETRY_DELAY_S)
        response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()
