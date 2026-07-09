"""Current date and time tool."""

import logging
import threading
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

GEOIP_TIMEZONE_ENDPOINTS = (
    ("https://ipapi.co/timezone/", "text"),
    ("https://worldtimeapi.org/api/ip", "json"),
)
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
GEOCODING_TIMEOUT_S = 3.0
GEOIP_TIMEOUT_S = 2.0
GEOIP_WARMUP_WAIT_S = 0.05
UTC = ZoneInfo("UTC")


class GetTime(Tool):
    """Report the current date and time in a resolved timezone."""

    name = "get_time"
    description = (
        "Get the current local date and time, the time in a specific IANA timezone or named place, "
        "or the current time difference between two timezones or places."
        " For local time, use the tool's resolver instead of asking the user for their city."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone or named place requested by the user, e.g. 'Europe/Paris' or 'Yekaterinburg, Russia'. Use an empty string for the user's local time; do not ask for their city first.",
            },
            "compare_timezone": {
                "type": "string",
                "description": "Set only for time-difference questions. Use an IANA timezone or named place, e.g. 'Asia/Tokyo' or 'Tokyo'. For differences involving local time, set timezone to an empty string and compare_timezone to the other timezone or place.",
            },
        },
        "required": ["timezone"],
    }

    def __init__(self) -> None:
        """Initialize per-tool geo-IP cache state."""
        self._geoip_lock = threading.Lock()
        self._geoip_lookup_started = False
        self._geoip_lookup_complete = threading.Event()
        self._cached_geoip_timezone_name: str | None = None

    def warm_local_timezone_cache(self) -> None:
        """Resolve local timezone in the background for fast local-time tool calls."""
        with self._geoip_lock:
            if self._cached_geoip_timezone_name or self._geoip_lookup_started:
                return
            self._geoip_lookup_started = True
            self._geoip_lookup_complete.clear()

        threading.Thread(
            target=self._resolve_geoip_timezone_name,
            daemon=True,
            name="get-time-geoip",
        ).start()

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return the current date and time in the requested timezone, or local time."""
        timezone_arg = kwargs.get("timezone")
        if timezone_arg is None:
            tz_name = ""
        elif isinstance(timezone_arg, str):
            tz_name = timezone_arg.strip()
        else:
            return {"error": "timezone must be a string"}

        compare_arg = kwargs.get("compare_timezone")
        if compare_arg is None:
            compare_tz_name = None
        elif isinstance(compare_arg, str):
            compare_tz_name = compare_arg.strip() or None
        else:
            return {"error": "compare_timezone must be a string"}

        resolved_tz_name, timezone, location, error = await self._resolve_timezone(tz_name)
        if error:
            return {"error": error}

        now_utc = datetime.now(UTC)
        now = now_utc.astimezone(timezone)
        result = _format_time(now, resolved_tz_name)
        if location is not None:
            result["location"] = location

        if compare_tz_name is not None:
            compared_tz_name, compared_timezone, compared_location, error = await self._resolve_timezone(
                compare_tz_name
            )
            if error:
                return {"error": error}
            compared_now = now_utc.astimezone(compared_timezone)
            difference_minutes = _offset_minutes(compared_now) - _offset_minutes(now)
            result["compare"] = _format_time(compared_now, compared_tz_name)
            if compared_location is not None:
                result["compare"]["location"] = compared_location
            result["time_difference_minutes"] = difference_minutes
            result["time_difference_hours"] = difference_minutes / 60
            if difference_minutes == 0:
                result["time_difference_summary"] = (
                    f"{compared_tz_name} has the same current UTC offset as {resolved_tz_name}."
                )
            else:
                direction = "ahead of" if difference_minutes > 0 else "behind"
                minutes = abs(difference_minutes)
                hours, remaining_minutes = divmod(minutes, 60)
                duration = f"{hours} hour{'s' if hours != 1 else ''}"
                if remaining_minutes:
                    duration = f"{duration} {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"
                result["time_difference_summary"] = f"{compared_tz_name} is {duration} {direction} {resolved_tz_name}."

        return result

    async def _resolve_timezone(self, tz_name: str) -> tuple[str, ZoneInfo, str | None, str | None]:
        if tz_name:
            timezone = _load_timezone(tz_name)
            if timezone is not None:
                return tz_name, timezone, None, None
            return await _resolve_place_timezone(tz_name)

        local_tz_name = self._resolve_local_timezone_name()
        if local_tz_name is None:
            return "", UTC, None, "local timezone unavailable; geo-IP detection failed"
        timezone = _load_timezone(local_tz_name)
        if timezone is None:
            return local_tz_name, UTC, None, "local timezone unavailable; geo-IP detection failed"
        return local_tz_name, timezone, None, None

    def _resolve_local_timezone_name(self) -> str | None:
        with self._geoip_lock:
            cached_tz_name = self._cached_geoip_timezone_name
            if cached_tz_name:
                return cached_tz_name

        self.warm_local_timezone_cache()
        self._geoip_lookup_complete.wait(GEOIP_WARMUP_WAIT_S)
        with self._geoip_lock:
            return self._cached_geoip_timezone_name

    def _resolve_geoip_timezone_name(self) -> str | None:
        timezone_name: str | None = None
        failure_reason = "no timezone returned"
        for url, response_format in GEOIP_TIMEZONE_ENDPOINTS:
            try:
                response = httpx.get(url, timeout=GEOIP_TIMEOUT_S)
                response.raise_for_status()
            except httpx.HTTPError as e:
                failure_reason = f"{url}: {e}"
                continue

            candidate_timezone_name = ""
            if response_format == "text":
                candidate_timezone_name = response.text.strip()
            else:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    raw_timezone = payload.get("timezone")
                    if isinstance(raw_timezone, str):
                        candidate_timezone_name = raw_timezone.strip()

            if candidate_timezone_name and _load_timezone(candidate_timezone_name) is not None:
                timezone_name = candidate_timezone_name
                break
            else:
                failure_reason = f"{url}: invalid timezone {candidate_timezone_name!r}"

        with self._geoip_lock:
            if timezone_name is None:
                self._geoip_lookup_started = False
            else:
                self._cached_geoip_timezone_name = timezone_name
        if timezone_name is not None:
            logger.info("Resolved timezone from geo-IP: %s", timezone_name)
        else:
            logger.warning("Failed to resolve timezone from geo-IP: %s", failure_reason)
        self._geoip_lookup_complete.set()
        return timezone_name


def _format_time(now: datetime, tz_name: str) -> dict[str, Any]:
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "weekday": now.strftime("%A"),
        "timezone": tz_name,
        "utc_offset_minutes": _offset_minutes(now),
        "summary": now.strftime("%A, %d %B %Y, %H:%M"),
    }


async def _resolve_place_timezone(location: str) -> tuple[str, ZoneInfo, str | None, str | None]:
    error = f"no timezone or place match for '{location}'"
    try:
        async with httpx.AsyncClient(timeout=GEOCODING_TIMEOUT_S) as client:
            response = await client.get(
                GEOCODING_URL,
                params={"name": location, "count": 3, "language": "en"},
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Failed to resolve timezone for place %r: %s", location, e)
        return location, UTC, None, error

    if not isinstance(payload, dict):
        return location, UTC, None, error
    matches = payload.get("results")
    if not isinstance(matches, list):
        return location, UTC, None, error

    for match in matches:
        if not isinstance(match, dict):
            continue
        timezone_name = match.get("timezone")
        if not isinstance(timezone_name, str):
            continue
        timezone_name = timezone_name.strip()
        timezone = _load_timezone(timezone_name)
        if timezone is None:
            continue
        location_label = ", ".join(
            part for part in (match.get("name"), match.get("admin1"), match.get("country")) if isinstance(part, str)
        )
        return timezone_name, timezone, location_label or location, None

    return location, UTC, None, error


def _offset_minutes(now: datetime) -> int:
    offset = now.utcoffset()
    if offset is None:
        return 0
    return int(offset.total_seconds() // 60)


def _load_timezone(tz_name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        return None
