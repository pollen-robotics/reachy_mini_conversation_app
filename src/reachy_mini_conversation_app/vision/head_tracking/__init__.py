"""Head-tracking backends and process helpers."""

from typing import Protocol, TypeAlias, SupportsFloat

import numpy as np
from numpy.typing import NDArray


HeadTrackerResult: TypeAlias = tuple[NDArray[np.float32] | None, SupportsFloat | None]


class HeadTracker(Protocol):
    """Shared interface for optional head-tracking backends."""

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Return the detected head position for a frame."""
