"""Picture capture utility for ReachyMiniScript."""

import base64
import logging
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move


if TYPE_CHECKING:
    from reachy_mini_conversation_app.camera_worker import CameraWorker


logger = logging.getLogger(__name__)


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
