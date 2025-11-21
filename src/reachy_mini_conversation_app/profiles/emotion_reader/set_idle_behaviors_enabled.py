"""Profile-local tool to toggle idle behaviors such as auto-idle tool calls."""

from __future__ import annotations

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.profile_settings import update_profile_settings
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class SetIdleBehaviorsEnabled(Tool):
    """Toggle idle behaviors (automatic dances/idle tool calls)."""

    name = "set_idle_behaviors_enabled"
    description = "Enable or disable idle behaviors (like spontaneous dances)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "Set to true to allow idle behaviors, false to disable them.",
            },
        },
        "required": ["enabled"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Flip the idle-behavior flag at runtime."""
        enabled = kwargs.get("enabled")
        if not isinstance(enabled, bool):
            return {"error": "enabled must be a boolean"}

        settings = update_profile_settings(enable_idle_behaviors=enabled)
        state = "enabled" if settings.enable_idle_behaviors else "disabled"
        logger.info(f"Idle behaviors {state} via tool call.")
        return {
            "status": f"idle behaviors {state}",
            "enable_voice": settings.enable_voice,
            "enable_idle_behaviors": settings.enable_idle_behaviors,
        }
