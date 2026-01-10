"""Type stubs for reachy_mini.utils package."""

import numpy as np
from numpy.typing import NDArray

def create_head_pose(
    x: float = ...,
    y: float = ...,
    z: float = ...,
    roll: float = ...,
    pitch: float = ...,
    yaw: float = ...,
    degrees: bool = ...,
    mm: bool = ...,
) -> NDArray[np.float32]:
    """Create a 4x4 head pose transformation matrix.

    Args:
        x, y, z: Translation values
        roll, pitch, yaw: Rotation values
        degrees: If True, rotation values are in degrees
        mm: If True, translation values are in millimeters

    Returns:
        A 4x4 transformation matrix.
    """
    ...
