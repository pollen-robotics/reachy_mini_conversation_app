"""Queue move implementations for rmscript execution."""

import base64
import logging
import threading
from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move


if TYPE_CHECKING:
    from reachy_mini_conversation_app.camera_worker import CameraWorker


logger = logging.getLogger(__name__)


class GotoQueueMove(Move):  # type: ignore
    """Wrapper for goto moves to work with the movement queue system."""

    def __init__(
        self,
        target_head_pose: NDArray[np.float32],
        start_head_pose: NDArray[np.float32] | None = None,
        target_antennas: Tuple[float, float] = (0, 0),
        start_antennas: Tuple[float, float] | None = None,
        target_body_yaw: float = 0,
        start_body_yaw: float | None = None,
        duration: float = 1.0,
    ):
        """Initialize a GotoQueueMove."""
        self._duration = duration
        self.target_head_pose = target_head_pose
        self.start_head_pose = start_head_pose
        self.target_antennas = target_antennas
        self.start_antennas = start_antennas or (0, 0)
        self.target_body_yaw = target_body_yaw
        self.start_body_yaw = start_body_yaw or 0

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate goto move at time t using linear interpolation."""
        try:
            from reachy_mini.utils import create_head_pose
            from reachy_mini.utils.interpolation import linear_pose_interpolation

            # Clamp t to [0, 1] for interpolation
            t_clamped = max(0, min(1, t / self.duration))

            # Use start pose if available, otherwise neutral
            if self.start_head_pose is not None:
                start_pose = self.start_head_pose
            else:
                start_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

            # Interpolate head pose
            head_pose = linear_pose_interpolation(start_pose, self.target_head_pose, t_clamped)

            # Interpolate antennas - return as numpy array
            antennas = np.array(
                [
                    self.start_antennas[0] + (self.target_antennas[0] - self.start_antennas[0]) * t_clamped,
                    self.start_antennas[1] + (self.target_antennas[1] - self.start_antennas[1]) * t_clamped,
                ],
                dtype=np.float64,
            )

            # Interpolate body yaw
            body_yaw = self.start_body_yaw + (self.target_body_yaw - self.start_body_yaw) * t_clamped

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating goto move at t={t}: {e}")
            # Return target pose on error - convert to float64
            target_head_pose_f64 = self.target_head_pose.astype(np.float64)
            target_antennas_array = np.array([self.target_antennas[0], self.target_antennas[1]], dtype=np.float64)
            return (target_head_pose_f64, target_antennas_array, self.target_body_yaw)


class SoundQueueMove(Move):  # type: ignore
    """Wrapper for sound playback to work with the movement queue system."""

    def __init__(self, sound_file_path: str, duration: float = 0.0, blocking: bool = False, loop: bool = False):
        """Initialize a SoundQueueMove.

        Args:
            sound_file_path: Absolute path to the sound file
            duration: Duration of the sound in seconds (for blocking mode or loop duration)
            blocking: If True, move lasts for sound duration. If False, move is instant.
            loop: If True, loop the sound for the specified duration.

        """
        self.sound_file_path = str(Path(sound_file_path))
        self.loop = loop
        self._duration = duration if (blocking or loop) else 0.01  # Instant for async, full duration for blocking/loop
        self.blocking = blocking
        self._played = False
        self._stop_event = None  # For stopping looped sounds

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate sound move - plays sound on first call, maintains current pose."""
        # Play sound only once at the start
        if not self._played and t <= 0.01:
            self._played = True
            try:
                if self.loop:
                    # Loop mode: play sound repeatedly for specified duration
                    from reachy_mini_conversation_app.rmscript_adapters.sound_player import play_sound_loop
                    self._stop_event = play_sound_loop(Path(self.sound_file_path), self._duration)
                elif self.blocking:
                    # Blocking mode: play in foreground thread (blocks the evaluate call)
                    from reachy_mini_conversation_app.rmscript_adapters.sound_player import play_sound_blocking
                    play_sound_blocking(Path(self.sound_file_path))
                else:
                    # Async mode: play in background thread
                    from reachy_mini_conversation_app.rmscript_adapters.sound_player import play_sound_async
                    play_sound_async(Path(self.sound_file_path))
            except Exception as e:
                logger.error(f"Error playing sound {self.sound_file_path}: {e}")

        # Return None for all joints - maintains current pose
        return (None, None, None)


class PictureQueueMove(Move):  # type: ignore
    """Wrapper for picture capture to work with the movement queue system."""

    def __init__(self, camera_worker: Optional["CameraWorker"] = None, duration: float = 0.01):
        """Initialize a PictureQueueMove.

        Args:
            camera_worker: Camera worker to capture frame from
            duration: Duration of the move (instant by default)

        """
        self.camera_worker = camera_worker
        self._duration = duration
        self._captured = False
        self.picture_base64: Optional[str] = None
        self.saved_path: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate picture move - captures picture on first call, maintains current pose."""
        # Capture picture only once at the start
        if not self._captured and t <= 0.01:
            self._captured = True
            try:
                if self.camera_worker is not None:
                    frame = self.camera_worker.get_latest_frame()  # Returns RGB
                    if frame is not None:
                        # Generate unique filename with timestamp
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"/tmp/reachy_picture_{timestamp}.jpg"

                        # Save picture to /tmp
                        cv2.imwrite(filename, frame)
                        self.saved_path = filename
                        logger.info(f"📸 Picture saved to {filename}")

                        # Also encode as base64 for returning to LLM
                        with open(filename, 'rb') as f:
                            self.picture_base64 = base64.b64encode(f.read()).decode('utf-8')
                    else:
                        logger.warning("No frame available from camera worker")
                        self.picture_base64 = None
                        self.saved_path = None
                else:
                    logger.warning("No camera worker available for picture capture")
                    self.picture_base64 = None
                    self.saved_path = None
            except Exception as e:
                logger.error(f"Error capturing picture: {e}")
                self.picture_base64 = None
                self.saved_path = None

        # Return None for all joints - maintains current pose
        return (None, None, None)
