"""Profile-local tool to toggle voice output at runtime."""

from __future__ import annotations

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.profile_settings import update_profile_settings
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class SetVoiceEnabled(Tool):
    """Enable or disable voice responses dynamically."""

    name = "set_voice_enabled"
    description = "Enable or disable voice responses for the current profile."
    parameters_schema = {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "Set to true to enable speech, false to mute.",
            },
        },
        "required": ["enabled"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Flip the voice flag based on the LLM instruction."""
        enabled = kwargs.get("enabled")
        if not isinstance(enabled, bool):
            return {"error": "enabled must be a boolean"}

        settings = update_profile_settings(enable_voice=enabled)
        state = "enabled" if settings.enable_voice else "disabled"
        logger.info(f"Voice responses {state} via tool call.")
        return {
            "status": f"voice {state}",
            "enable_voice": settings.enable_voice,
            "enable_idle_behaviors": settings.enable_idle_behaviors,
        }
