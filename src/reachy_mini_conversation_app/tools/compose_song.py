import re
import json
import asyncio
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools._space_audio import save_audio, call_space_tool


logger = logging.getLogger(__name__)

SONG_SPACE = "victor/MiniMax-Music3-Jam"
_SONG_MCP_URL = "https://victor-minimax-music3-jam.hf.space/gradio_api/mcp/"
_SONG_TOOL = "MiniMax_Music3_Jam_create"
# A 30 s song took ~50 s; leave room for a ZeroGPU queue.
_GENERATION_TIMEOUT_S = 240.0
_DEFAULT_DURATION_S = 30
_MIN_DURATION_S = 10
_MAX_DURATION_S = 60
# The Space returns the WAV inline as base64 in one event: ~14 MB for 60 s of 44.1 kHz stereo.
_MAX_RESULT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class Song:
    """A generated song saved on disk, ready to play."""

    title: str
    path: Path
    duration_s: float
    # The Space's style tags, e.g. "pop, upbeat, female vocals, 120 bpm"; play_song dances to them.
    style: str = ""
    bpm: int | None = None


# shortcut: only the latest song is kept, in memory; keep a list if users want to replay older ones.
latest_song: Song | None = None


def _song_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Find the Space's final JSON result (title, audio, ...) among its text blocks."""
    for block in reversed(result.get("content_blocks") or []):
        try:
            payload = json.loads(block.get("text") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("audio"):
            return payload
    return None


class ComposeSong(Tool):
    """Compose a short sung song with a Hugging Face Space, in the background."""

    name = "compose_song"
    long_running = True
    description = (
        "Write and sing a short original song from a description of its topic, mood and style. "
        "It takes about a minute and runs in the background: tell the user you are on it and keep talking. "
        "When it is ready, tell the user its title and ask whether they want to hear it, then play it with play_song."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "What the song is about, its mood and musical style, in a sentence or two.",
            },
            "duration_s": {
                "type": "integer",
                "description": f"Length in seconds, {_MIN_DURATION_S} to {_MAX_DURATION_S}. Defaults to "
                f"{_DEFAULT_DURATION_S}.",
            },
        },
        "required": ["description"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Generate the song, save it, and report it ready to play."""
        global latest_song
        description = str(kwargs.get("description") or "").strip()
        if not description:
            return {"error": "describe what the song should be about"}
        duration = min(max(int(kwargs.get("duration_s") or _DEFAULT_DURATION_S), _MIN_DURATION_S), _MAX_DURATION_S)

        logger.info("Tool call: compose_song duration=%ds description=%r", duration, description)
        result = await call_space_tool(
            SONG_SPACE,
            _SONG_MCP_URL,
            _SONG_TOOL,
            {"description": description, "audio_duration": duration},
            timeout_s=_GENERATION_TIMEOUT_S,
            max_event_bytes=_MAX_RESULT_BYTES,
        )
        if "error" in result:
            return result

        payload = _song_payload(result)
        if payload is None:
            logger.error("compose_song: no audio in the Space result: %s", str(result.get("text"))[:300])
            return {"error": "the song Space returned no audio"}
        title = str(payload.get("title") or "Untitled")
        try:
            path = await asyncio.to_thread(save_audio, str(payload["audio"]), title)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.error("compose_song could not save the audio: %s", exc)
            return {"error": f"could not save the song: {exc}"}

        style = str(payload.get("tags") or "")
        bpm_match = re.search(r"(\d+)\s*bpm", style, re.IGNORECASE)
        # Only the latest song can be played; a song is up to 14 MB and /tmp may be in RAM.
        if latest_song is not None:
            latest_song.path.unlink(missing_ok=True)
        latest_song = Song(
            title=title,
            path=path,
            duration_s=float(payload.get("duration") or duration),
            style=style,
            bpm=int(bpm_match.group(1)) if bpm_match else None,
        )
        logger.info("compose_song ready: %r (%.0f s, %s) at %s", title, latest_song.duration_s, style, path)
        return {
            "status": "ready",
            "title": title,
            "style": style,
            "duration_s": latest_song.duration_s,
            "next": "Tell the user the song is ready and ask whether to play it now; if yes, call play_song.",
        }
