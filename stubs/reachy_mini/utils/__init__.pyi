"""Type stubs for reachy_mini.utils package."""

from typing import Any

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
) -> NDArray[Any]:
    """Create a 4x4 head pose transformation matrix.

    Args:
        x: Translation along X axis
        y: Translation along Y axis
        z: Translation along Z axis
        roll: Rotation around X axis
        pitch: Rotation around Y axis
        yaw: Rotation around Z axis
        degrees: If True, rotation values are in degrees
        mm: If True, translation values are in millimeters

    Returns:
        A 4x4 transformation matrix.

    """
    ...
