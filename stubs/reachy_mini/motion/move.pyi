"""Type stubs for reachy_mini.motion.move module."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

class Move:
    """Base class for robot moves.

    This class defines the interface for all move implementations.
    Subclasses must implement the duration property and evaluate method.
    """

    description: str

    @property
    def duration(self) -> float:
        """Duration of the move in seconds."""
        ...

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate the move at time t.

        Args:
            t: Time in seconds since the start of the move.

        Returns:
            A tuple of (head_pose, antennas, body_yaw) where:
            - head_pose: 4x4 transformation matrix or None
            - antennas: Array of antenna positions or None
            - body_yaw: Body yaw angle or None
        """
        ...
