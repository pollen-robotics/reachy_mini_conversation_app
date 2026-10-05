import re
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.config import get_default_voice, get_available_voices
from reachy_mini_conversation_app.personality import save_user_personality
from reachy_mini_conversation_app.personality_tag import to_tag_token
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Tools of a personality created from an accessory. This tool is left out on purpose:
# core_tools offers it to every profile while a reader is attached.
_DEFAULT_TOOLS = (
    "camera",
    "dance",
    "head_tracking",
    "move_head",
    "play_emotion",
    "stop_dance",
    "stop_emotion",
)


def _sanitize_name(name: str) -> str:
    """Coerce a model-supplied name into a valid profile folder name."""
    sanitized = re.sub(r"\s+", "_", name.strip())
    return re.sub(r"[^a-zA-Z0-9_-]", "", sanitized)


class CreateAccessoryPersonality(Tool):
    """Create a new personality profile and write its NFC code to the blank tag on the reader."""

    name = "create_accessory_personality"
    description = (
        "Create a new personality profile from a name and a system-prompt, "
        "then write its unique code onto the accessory currently placed on the head. "
        "Call this only after the user has confirmed they want to create a new personality "
        "and has described what it should be like."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Short identifier for the personality (e.g. 'pirate', 'chef'). "
                    "Used as the profile folder name — letters, digits, hyphens and underscores only."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Full system-prompt for this personality, crafted from the user's description. "
                    "Should be concise and capture the character, tone and language rules."
                ),
            },
            "voice": {
                "type": "string",
                "description": "TTS voice to use for this personality.",
                # Read from the backend rather than hardcoded: a stale list would
                # be silently rejected and every new personality would sound alike.
                "enum": get_available_voices(),
            },
        },
        "required": ["name", "instructions"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Save the personality, then write its token to the accessory on the reader."""
        name = (kwargs.get("name") or "").strip()
        instructions = (kwargs.get("instructions") or "").strip()
        voice = (kwargs.get("voice") or "").strip() or get_default_voice()

        if not name or not instructions:
            return {"error": "name and instructions are required"}

        name_s = _sanitize_name(name)
        if not name_s:
            return {"error": f"Invalid personality name: {name!r}"}

        logger.info("create_accessory_personality: creating profile %r voice=%r", name_s, voice)

        try:
            personality = save_user_personality(
                name_s,
                instructions,
                voice=voice,
                overwrite=True,
                default_tools=_DEFAULT_TOOLS,
            )
        except Exception as exc:
            logger.error("create_accessory_personality: failed to write profile: %s", exc)
            return {"error": f"Failed to write profile: {exc}"}

        code = to_tag_token(personality)
        if code is None:
            return {"error": f"Cannot write personality {personality!r} to a tag"}

        if deps.blank_tag_present and deps.nfc_client is not None:
            # The write-tag move and the welcome speech follow a successful write (see rfid_routes).
            deps.pending_nfc_write = None
            deps.recently_written_codes.add(code)
            deps.nfc_client.write_tag_in_background(code)
            logger.info("create_accessory_personality: writing %r to the accessory", code)
            return {"status": "writing", "personality": personality, "code": code}

        # No blank accessory on the reader: rfid_routes writes it on the next detection.
        deps.pending_nfc_write = {"code": code, "personality": personality}
        logger.info("create_accessory_personality: no blank tag present — stored pending write for %r", personality)
        return {
            "status": "waiting_for_tag",
            "personality": personality,
            "code": code,
            "message": (
                "No accessory on the head right now. "
                "Ask the user to place the accessory back on the head to program it."
            ),
        }
