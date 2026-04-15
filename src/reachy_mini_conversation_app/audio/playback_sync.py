"""Helpers for aligning speech motion with local audio playback."""

import asyncio
from typing import Any


def estimate_pending_playback_seconds(robot: Any) -> float:
    """Best-effort estimate of audio still queued in the local player."""
    media = getattr(robot, "media", None)
    audio = getattr(media, "audio", None)
    if audio is None:
        return 0.0

    next_pts_ns = getattr(audio, "_playback_next_pts_ns", None)
    get_running_time_ns = getattr(audio, "_get_playback_running_time_ns", None)
    if next_pts_ns is None or not callable(get_running_time_ns):
        return 0.0

    try:
        pending_ns = int(next_pts_ns) - int(get_running_time_ns())
    except Exception:
        return 0.0

    return max(0.0, pending_ns / 1e9)


class SpeechMotionReset:
    """Schedule a head-wobbler reset after queued audio finishes playing.

    Shared by both OpenAI and Gemini handlers.
    """

    def __init__(self, head_wobbler: Any, robot: Any) -> None:
        """Initialize the reset helper.

        Args:
            head_wobbler: Object exposing a ``reset()`` method.
            robot: Robot instance used to estimate queued local playback.

        """
        self._head_wobbler = head_wobbler
        self._robot = robot
        self._task: asyncio.Task[None] | None = None

    def cancel(self) -> None:
        """Cancel any pending reset."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def schedule(self) -> None:
        """Reset wobble immediately or after queued playback drains."""
        self.cancel()
        pending_s = estimate_pending_playback_seconds(self._robot)
        if pending_s <= 0:
            self._head_wobbler.reset()
            return

        async def _delayed() -> None:
            try:
                await asyncio.sleep(pending_s)
            except asyncio.CancelledError:
                return
            self._head_wobbler.reset()

        self._task = asyncio.create_task(_delayed(), name="speech-motion-reset")
