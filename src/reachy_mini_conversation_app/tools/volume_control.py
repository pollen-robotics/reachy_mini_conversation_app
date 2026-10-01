import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.daemon_api import DaemonApiError, daemon_request
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_DEVICE_PATHS = {
    "speaker": "/api/volume",
    "microphone": "/api/volume/microphone",
}


class VolumeControl(Tool):
    """Read or change the robot's speaker and microphone volume."""

    name = "volume_control"
    description = (
        "Read or change Reachy's own speaker or microphone volume (0-100). Omit 'level' to read the current "
        "value; call without 'level' first when the user asks for a relative change like 'louder' or 'quieter'. "
        "Setting the speaker volume plays a short confirmation beep on the robot."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "device": {
                "type": "string",
                "enum": list(_DEVICE_PATHS),
                "description": "Which device to act on. Defaults to the speaker.",
            },
            "level": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "New volume level. Omit to just read the current level.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Read the current volume, or set it when a level is given."""
        device = str(kwargs.get("device") or "speaker")
        base_path = _DEVICE_PATHS.get(device)
        if base_path is None:
            return {"error": f"unknown device {device!r}"}

        level = kwargs.get("level")
        if level is not None:
            try:
                level = int(level)
            except (TypeError, ValueError):
                return {"error": f"level must be an integer, got {level!r}"}
            if not 0 <= level <= 100:
                return {"error": "level must be between 0 and 100"}

        logger.info("Tool call: volume_control device=%s level=%s", device, level)
        try:
            if level is None:
                response = await asyncio.to_thread(daemon_request, deps.reachy_mini, f"{base_path}/current")
            else:
                response = await asyncio.to_thread(
                    daemon_request,
                    deps.reachy_mini,
                    f"{base_path}/set",
                    method="POST",
                    payload={"volume": level},
                )
        except DaemonApiError as e:
            logger.error("volume_control failed: %s", e)
            return {"error": str(e)}

        return {"device": device, "volume": response.get("volume") if isinstance(response, dict) else None}
