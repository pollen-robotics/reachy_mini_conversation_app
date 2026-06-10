"""Exponential backoff for unprompted idle-signal riffs.

The Gemini Live handler nudges the model with an idle signal after the room
has been quiet for a while, which makes the robot riff unprompted. With a
fixed threshold those riffs arrive back-to-back and leave no growing window
for a human to interject. This helper grows the idle threshold by ``factor``
after every signal (capped at ``max_s``) and snaps it back to ``first_s``
whenever the user actually speaks.

Pure synchronous state — no asyncio, no threads. The caller owns the clock
and decides when a signal fires; this class only tracks the moving threshold.
"""

from __future__ import annotations
import logging


logger = logging.getLogger(__name__)


class IdleSignalBackoff:
    """Moving idle threshold: first_s → first_s·factor → … capped at max_s."""

    def __init__(self, *, first_s: float, factor: float, max_s: float) -> None:
        """Store config and start the threshold at ``first_s``."""
        self._first_s = first_s
        self._factor = factor
        self._max_s = max_s
        self._threshold_s = first_s

    @property
    def threshold_s(self) -> float:
        """Seconds of idle time required before the next idle signal."""
        return self._threshold_s

    def on_signal_sent(self) -> None:
        """Grow the threshold after an idle signal fires."""
        self._threshold_s = min(self._threshold_s * self._factor, self._max_s)
        logger.debug("Idle backoff: next idle signal after %.1fs", self._threshold_s)

    def on_user_activity(self) -> None:
        """Reset the threshold when the user speaks."""
        if self._threshold_s != self._first_s:
            logger.debug("Idle backoff: user activity — threshold reset to %.1fs", self._first_s)
        self._threshold_s = self._first_s
