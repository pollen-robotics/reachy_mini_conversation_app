"""Type stubs for reachy_mini.utils.interpolation module."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

def compose_world_offset(
    primary: NDArray[np.floating[Any]],
    secondary: NDArray[np.floating[Any]],
    reorthonormalize: bool = ...,
) -> NDArray[np.float32]:
    """Compose primary pose with secondary offset in world frame.

    Args:
        primary: Primary 4x4 transformation matrix
        secondary: Secondary 4x4 transformation matrix (offset)
        reorthonormalize: Whether to reorthonormalize the result

    Returns:
        Combined 4x4 transformation matrix.

    """
    ...

def linear_pose_interpolation(
    start: NDArray[np.floating[Any]] | None,
    end: NDArray[np.floating[Any]],
    t: float,
) -> NDArray[np.float64]:
    """Linearly interpolate between two poses.

    Args:
        start: Starting 4x4 transformation matrix (can be None)
        end: Ending 4x4 transformation matrix
        t: Interpolation factor (0 to 1)

    Returns:
        Interpolated 4x4 transformation matrix.

    """
    ...
