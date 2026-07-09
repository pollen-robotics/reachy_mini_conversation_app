"""Current date and time tool."""

import logging
import threading
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

GEOIP_TIMEZONE_URL = "https://ipapi.co/timezone/"
GEOIP_TIMEOUT_S = 2.0
GEOIP_WARMUP_WAIT_S = 0.05
UTC = ZoneInfo("UTC")


class GetTime(Tool):
    """Report the current date and time in a resolved timezone."""

    name = "get_time"
    description = (
        "Get the current local date and time, the time in a specific IANA timezone, "
        "or the current time difference between two timezones."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone requested by the user, e.g. 'Europe/Paris'. Use an empty string for the user's local time.",
            },
            "compare_timezone": {
                "type": "string",
                "description": "Set for time-difference questions. Use an IANA timezone, e.g. 'Asia/Tokyo', or an empty string for the user's local time.",
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

        threading.Thread(target=self._resolve_geoip_timezone_name, daemon=True, name="get-time-geoip").start()

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
            compare_tz_name = compare_arg.strip()
        else:
            return {"error": "compare_timezone must be a string"}

        resolved_tz_name, timezone, error = self._resolve_timezone(tz_name)
        if error:
            return {"error": error}

        now_utc = datetime.now(UTC)
        now = now_utc.astimezone(timezone)
        result = _format_time(now, resolved_tz_name)

        if compare_tz_name is not None:
            compared_tz_name, compared_timezone, error = self._resolve_timezone(compare_tz_name)
            if error:
                return {"error": error}
            compared_now = now_utc.astimezone(compared_timezone)
            difference_minutes = _offset_minutes(compared_now) - _offset_minutes(now)
            result["compare"] = _format_time(compared_now, compared_tz_name)
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

    def _resolve_timezone(self, tz_name: str) -> tuple[str, ZoneInfo, str | None]:
        if tz_name:
            timezone = _load_timezone(tz_name)
            if timezone is None:
                return tz_name, UTC, f"unknown timezone: {tz_name}"
            return tz_name, timezone, None

        local_tz_name = self._resolve_local_timezone_name()
        if local_tz_name is None:
            return "", UTC, "local timezone unavailable; geo-IP detection failed"
        timezone = _load_timezone(local_tz_name)
        if timezone is None:
            return local_tz_name, UTC, "local timezone unavailable; geo-IP detection failed"
        return local_tz_name, timezone, None

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
        try:
            response = httpx.get(GEOIP_TIMEZONE_URL, timeout=GEOIP_TIMEOUT_S)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Failed to resolve timezone from geo-IP: %s", e)
        else:
            candidate_timezone_name = response.text.strip()
            if candidate_timezone_name and _load_timezone(candidate_timezone_name) is not None:
                timezone_name = candidate_timezone_name
            else:
                logger.warning("Geo-IP timezone lookup returned invalid timezone: %s", candidate_timezone_name)

        with self._geoip_lock:
            if timezone_name is None:
                self._geoip_lookup_started = False
            else:
                self._cached_geoip_timezone_name = timezone_name
        self._geoip_lookup_complete.set()
        if timezone_name is not None:
            logger.info("Resolved timezone from geo-IP: %s", timezone_name)
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
