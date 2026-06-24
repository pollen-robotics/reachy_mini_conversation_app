"""Camera worker thread with frame buffering."""

import time
import logging
import threading

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini


logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker buffering the latest frame for tools and UI."""

    def __init__(self, reachy_mini: ReachyMini) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini

        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Run the camera worker loop, keeping the latest frame buffered."""
        logger.debug("Starting camera working loop")

        while not self._stop_event.is_set():
            try:
                frame = self.reachy_mini.media.get_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                time.sleep(0.04)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")
