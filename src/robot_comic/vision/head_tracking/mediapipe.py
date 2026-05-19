"""MediaPipe head tracker backed by reachy_mini_toolbox."""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from robot_comic.vision.head_tracking import HeadTracker, HeadTrackerResult


class MediapipeHeadTracker:
    """MediaPipe head tracker provided by reachy_mini_toolbox.

    The reachy_mini_toolbox.vision.HeadTracker() graph build takes ~0.5-1 s
    on the Pi.  Construction is deferred to the first get_head_position()
    call so that cold-start cost is masked by GStreamer's own ~5 s warm-up.
    """

    def __init__(self) -> None:
        """Stash env setup; defer vision.HeadTracker() to first use."""
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/robot-comic-matplotlib")
        self._tracker: Optional[HeadTracker] = None

    def _ensure_tracker(self) -> HeadTracker:
        """Return the underlying tracker, constructing it on first call.

        Idempotent: subsequent calls return the cached instance.  Tests that
        monkeypatch ``instance._tracker`` directly are honoured because we
        only construct when ``_tracker is None``.
        """
        if self._tracker is None:
            from reachy_mini_toolbox import vision

            self._tracker = vision.HeadTracker()
        return self._tracker

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Return the detected head position for a frame."""
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.ascontiguousarray(img[..., ::-1])
        return self._ensure_tracker().get_head_position(img)
