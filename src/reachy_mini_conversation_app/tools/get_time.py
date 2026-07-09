"""Current date and time tool."""

import logging
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GetTime(Tool):
    """Report the current date and time, optionally for a given timezone."""

    name = "get_time"
    description = "Get the current date and time. Optionally pass an IANA timezone like 'Europe/Paris'."
    parameters_schema = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name (e.g. 'Europe/Paris'). Omit for the robot's local time.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return the current date and time in the requested timezone, or local time."""
        tz_name = (kwargs.get("timezone") or "").strip()
        if tz_name:
            try:
                now = datetime.now(ZoneInfo(tz_name))
            except (ZoneInfoNotFoundError, ValueError):
                return {"error": f"unknown timezone: {tz_name}"}
        else:
            now = datetime.now().astimezone()

        return {
            "iso": now.isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "weekday": now.strftime("%A"),
            "timezone": tz_name or now.tzname() or "",
            "summary": now.strftime("%A, %d %B %Y, %H:%M"),
        }
