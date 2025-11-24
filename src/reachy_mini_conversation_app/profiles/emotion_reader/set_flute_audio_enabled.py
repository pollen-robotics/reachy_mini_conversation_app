"""Tool to toggle flute audio playback for emotions."""

from __future__ import annotations

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.profile_settings import update_profile_settings
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class SetFluteAudioEnabled(Tool):
    """Enable or disable flute audio for emotion playback."""

    name = "set_flute_audio_enabled"
    description = "Toggle the flute soundtrack that accompanies emotion playback."
    parameters_schema = {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "Set to true to enable flute audio, false to mute it.",
            },
        },
        "required": ["enabled"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        enabled = kwargs.get("enabled")
        if not isinstance(enabled, bool):
            return {"error": "enabled must be a boolean"}

        settings = update_profile_settings(enable_flute_audio=enabled)
        logger.info("Flute audio %s via tool call", "enabled" if settings.enable_flute_audio else "disabled")
        return {
            "status": "flute audio enabled" if enabled else "flute audio disabled",
            "enable_flute_audio": settings.enable_flute_audio,
        }
