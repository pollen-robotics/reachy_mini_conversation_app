import random
import asyncio
import logging
from typing import Any
from dataclasses import dataclass

from reachy_mini_conversation_app.tools import compose_song
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove


logger = logging.getLogger(__name__)

# The dances library's moves, by character.
SMOOTH_MOVES = (
    "simple_nod",
    "head_tilt_roll",
    "side_to_side_sway",
    "uh_huh_tilt",
    "groovy_sway_and_roll",
    "pendulum_swing",
    "polyrhythm_combo",
)
RHYTHMIC_MOVES = ("side_peekaboo", "chin_lead", "interwoven_spirals", "jackson_square", "yeah_nod", "dizzy_spin")
INTENSE_MOVES = (
    "headbanger_combo",
    "stumble_and_recover",
    "sharp_side_tilt",
    "chicken_peck",
    "grid_snap",
    "neck_recoil",
    "side_glance_flick",
)


@dataclass(frozen=True)
class DanceStyle:
    """Which moves to dance to a song, and how big."""

    name: str
    moves: tuple[str, ...]
    amplitude_scale: float


CALM = DanceStyle("calm", SMOOTH_MOVES, 0.5)
GROOVE = DanceStyle("groove", SMOOTH_MOVES + RHYTHMIC_MOVES, 0.75)
ENERGETIC = DanceStyle("energetic", SMOOTH_MOVES + RHYTHMIC_MOVES + INTENSE_MOVES, 1.0)

# Matched against the song Space's style tags, e.g. "pop, upbeat, female vocals, 120 bpm".
_ENERGETIC_WORDS = ("rock", "metal", "punk", "edm", "techno", "drum and bass", "dubstep", "hardstyle", "aggressive")
_CALM_WORDS = ("lullaby", "ballad", "ambient", "soft", "acoustic", "chill", "lo-fi", "lofi", "slow", "soothing")
_CALM_BPM = 90
# The library dances at 114 bpm; following the song keeps the moves on its beat, within a sane range.
_MIN_DANCE_BPM, _MAX_DANCE_BPM = 70, 160


def dance_style(song: compose_song.Song) -> DanceStyle:
    """Pick how hard to dance from the song's style tags and tempo."""
    style = song.style.lower()
    bpm = song.bpm or 0
    # Only the genre makes a song energetic: fast pop danced to the full library was too intense.
    if any(word in style for word in _ENERGETIC_WORDS):
        return ENERGETIC
    if any(word in style for word in _CALM_WORDS) or 0 < bpm < _CALM_BPM:
        return CALM
    return GROOVE


def _song_move(name: str, style: DanceStyle, bpm: int | None) -> DanceQueueMove:
    """Build one dance move scaled to the style and timed to the song."""
    move = DanceQueueMove(name)
    params = move.dance_move.move_params
    # Rebind, never update: the library hands every DanceMove the same shared params dict.
    move.dance_move.move_params = {
        key: value * style.amplitude_scale if "amp" in key else value for key, value in params.items()
    }
    if bpm:
        move.dance_move.default_bpm = min(max(bpm, _MIN_DANCE_BPM), _MAX_DANCE_BPM)
    return move


def _queue_dance(deps: ToolDependencies, song: compose_song.Song, style: DanceStyle) -> int:
    """Queue the style's moves until they cover the song; return how many were queued."""
    covered = 0.0
    queued = 0
    while covered < song.duration_s:
        move = _song_move(random.choice(style.moves), style, song.bpm)
        deps.movement_manager.queue_move(move)
        covered += move.duration
        queued += 1
    return queued


class PlaySong(Tool):
    """Play the latest composed song and dance along."""

    name = "play_song"
    # The song speaks for itself; a spoken follow-up would talk over it.
    needs_response = False
    description = (
        "Play the latest song made with compose_song on Reachy's speaker, dancing along until it ends. "
        "Only call it once the user wants to hear the song."
    )
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Start the song and the dance; both run on their own."""
        song = compose_song.latest_song
        if song is None or not song.path.exists():
            return {"error": "no song is ready yet; compose one first"}

        style = dance_style(song)
        logger.info(
            "Tool call: play_song %r (%.0f s, %s, %s dance)", song.title, song.duration_s, song.style, style.name
        )
        try:
            await asyncio.to_thread(deps.reachy_mini.media.play_sound, str(song.path))
        except FileNotFoundError as exc:
            return {"error": f"could not play the song: {exc}"}
        moves = _queue_dance(deps, song, style)
        return {"status": "playing", "title": song.title, "dance": style.name, "dance_moves": moves}
