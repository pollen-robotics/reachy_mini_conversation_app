"""MediaPipe head tracker backed by reachy_mini_toolbox."""

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.vision.head_tracking import HeadTracker, HeadTrackerResult


class MediapipeHeadTracker:
    """MediaPipe head tracker provided by reachy_mini_toolbox."""

    def __init__(self) -> None:
        """Initialize the toolbox head tracker lazily."""
        from reachy_mini_toolbox import vision

        self._tracker: HeadTracker = vision.HeadTracker()

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Return the detected head position for a frame."""
        return self._tracker.get_head_position(img)
