import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.tools import speak_in_voice
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools._space_audio import wait_for_quiet


logger = logging.getLogger(__name__)


class RepeatVoiceLine(Tool):
    """Replay a line already said with speak_in_voice, from its saved audio."""

    name = "repeat_voice_line"
    # The replay is the answer; a spoken follow-up would talk over it.
    needs_response = False
    description = (
        "Replay a line already said with speak_in_voice, in the same voice, right away and without generating "
        "anything. Without words it replays the latest line; with a few words from an earlier line it replays "
        "that one (the last five lines are kept)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "words": {
                "type": "string",
                "description": "A few words from the line to replay. Leave empty for the latest line.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Play the chosen line in the next pause."""
        lines = list(speak_in_voice.recent_lines)
        if not lines:
            return {"error": "no voice line has been said yet"}
        words = str(kwargs.get("words") or "").strip().lower()
        matches = [line for line in lines if words in line.text.lower()]
        if not matches:
            return {"error": f"no recent line contains {words!r}", "recent_lines": [line.text for line in lines]}
        line = matches[-1]
        if not line.path.exists():
            return {"error": "the saved audio of that line is gone"}

        logger.info("Tool call: repeat_voice_line %r", line.text)
        await wait_for_quiet(deps)
        await asyncio.to_thread(deps.reachy_mini.media.play_sound, str(line.path))
        return {"status": "playing", "text": line.text, "duration_s": round(line.duration_s, 1)}
