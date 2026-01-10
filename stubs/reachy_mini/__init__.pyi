"""Type stubs for reachy_mini package."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move as Move
from reachy_mini.motion.recorded_move import RecordedMoves as RecordedMoves

class ReachyMini:
    """Reachy Mini robot interface."""

    client: Any
    media: Any

    def __init__(self, media_backend: str = ..., localhost_only: bool = ...) -> None: ...
    def get_current_head_pose(self) -> Any: ...
    def get_current_joint_positions(self) -> tuple[Any, tuple[float, float]]: ...
    def set_target(self, head: Any = ..., antennas: tuple[float, float] = ..., body_yaw: float = ...) -> None: ...
    def goto_target(self, head: Any = ..., antennas: list[float] = ..., duration: float = ..., body_yaw: float = ...) -> None: ...
    def look_at_image(
        self,
        x: float,
        y: float,
        duration: float = ...,
        perform_movement: bool = ...,
    ) -> NDArray[np.float64]: ...

class ReachyMiniApp:
    """Base class for Reachy Mini apps."""

    custom_app_url: str
    dont_start_webserver: bool
    settings_app: Any

    def run(self, reachy_mini: ReachyMini, stop_event: Any) -> None: ...
    def wrapped_run(self) -> None: ...
    def stop(self) -> None: ...
    def _get_instance_path(self) -> Any: ...
