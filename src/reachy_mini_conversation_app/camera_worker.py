"""Camera worker thread with frame buffering and improved head tracking.

Improvements over the original implementation:
- EMA (Exponential Moving Average) smoothing for jitter-free head following
- Proportional gain with dead zone to reduce micro-movements
- Faster response when face moves significantly (adaptive smoothing)
- Reduced face-lost delay for more natural behavior
- Configurable parameters via class attributes
"""

import time
import logging
import threading
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation
from reachy_mini_conversation_app.vision.head_tracking import HeadTracker


logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and improved head tracking.

    The head tracking uses a proportional controller with EMA smoothing:
    - Smooth following when face moves slowly (conversation)
    - Faster response when face moves significantly (person moving)
    - Dead zone to prevent jitter when face is centered
    - Graceful return to neutral when face is lost
    """

    # --- Tunable parameters ---
    TRACKING_GAIN: float = 0.75  # Proportional gain (0.6 in original, higher = more responsive)
    EMA_ALPHA: float = 0.3  # Smoothing factor (0 = no change, 1 = no smoothing)
    EMA_ALPHA_FAST: float = 0.6  # Faster smoothing when face moves a lot
    LARGE_MOVEMENT_THRESHOLD: float = 0.15  # Normalized units, triggers fast tracking
    DEAD_ZONE: float = 0.02  # Ignore movements smaller than this (reduces jitter)
    FACE_LOST_DELAY: float = 1.0  # Seconds before starting return to neutral (was 2.0)
    INTERPOLATION_DURATION: float = 0.8  # Seconds to interpolate back to neutral (was 1.0)
    LOOP_PERIOD: float = 0.033  # ~30 FPS camera loop (was 0.04 = 25 FPS)

    def __init__(self, reachy_mini: ReachyMini, head_tracker: HeadTracker | None = None) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.face_tracking_lock = threading.Lock()

        # EMA state for smooth tracking
        self._ema_translation = np.zeros(3, dtype=np.float64)
        self._ema_rotation = np.zeros(3, dtype=np.float64)
        self._prev_eye_center: NDArray[np.float32] | None = None

        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None

        self.previous_head_tracking_state = self.is_head_tracking_enabled

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking."""
        self.is_head_tracking_enabled = enabled
        logger.info(f"Head tracking {'enabled' if enabled else 'disabled'}")

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
        head_tracker_close = getattr(self.head_tracker, "close", None)
        if callable(head_tracker_close):
            head_tracker_close()
        logger.debug("Camera worker stopped")

    def _compute_adaptive_alpha(self, eye_center: NDArray[np.float32]) -> float:
        """Choose EMA alpha based on how much the face moved since last frame.

        Large movements get less smoothing (faster response),
        small movements get more smoothing (less jitter).
        """
        if self._prev_eye_center is None:
            self._prev_eye_center = eye_center.copy()
            return self.EMA_ALPHA_FAST  # First detection: respond quickly

        movement = float(np.linalg.norm(eye_center - self._prev_eye_center))
        self._prev_eye_center = eye_center.copy()

        if movement > self.LARGE_MOVEMENT_THRESHOLD:
            return self.EMA_ALPHA_FAST
        return self.EMA_ALPHA

    def _apply_dead_zone(self, value: float) -> float:
        """Zero out values below the dead zone threshold."""
        if abs(value) < self.DEAD_ZONE:
            return 0.0
        return value

    def working_loop(self) -> None:
        """Run the camera worker loop with improved tracking."""
        logger.debug("Starting camera working loop (improved head tracking)")

        neutral_pose = np.eye(4)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame

                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None

                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    if self.is_head_tracking_enabled and self.head_tracker is not None:
                        eye_center, _ = self.head_tracker.get_head_position(frame)

                        if eye_center is not None:
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = None

                            # Adaptive smoothing based on movement magnitude
                            alpha = self._compute_adaptive_alpha(eye_center)

                            # Convert normalized coords to pixel coords for look_at
                            h, w, _ = frame.shape
                            eye_center_norm = (eye_center + 1) / 2
                            eye_center_pixels = [
                                eye_center_norm[0] * w,
                                eye_center_norm[1] * h,
                            ]

                            target_pose = self.reachy_mini.look_at_image(
                                eye_center_pixels[0],
                                eye_center_pixels[1],
                                duration=0.0,
                                perform_movement=False,
                            )

                            # Extract raw translation and rotation
                            raw_translation = target_pose[:3, 3] * self.TRACKING_GAIN
                            raw_rotation = (
                                R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)
                                * self.TRACKING_GAIN
                            )

                            # Apply dead zone
                            raw_translation = np.array([
                                self._apply_dead_zone(raw_translation[i])
                                for i in range(3)
                            ])
                            raw_rotation = np.array([
                                self._apply_dead_zone(raw_rotation[i])
                                for i in range(3)
                            ])

                            # EMA smoothing
                            self._ema_translation = (
                                alpha * raw_translation
                                + (1 - alpha) * self._ema_translation
                            )
                            self._ema_rotation = (
                                alpha * raw_rotation
                                + (1 - alpha) * self._ema_rotation
                            )

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    self._ema_translation[0],
                                    self._ema_translation[1],
                                    self._ema_translation[2],
                                    self._ema_rotation[0],
                                    self._ema_rotation[1],
                                    self._ema_rotation[2],
                                ]

                        elif self.last_face_detected_time is None or self.last_face_detected_time == current_time:
                            pass

                    # Return to neutral when face is lost
                    if self.last_face_detected_time is not None:
                        time_since_face_lost = current_time - self.last_face_detected_time

                        if time_since_face_lost >= self.FACE_LOST_DELAY:
                            if self.interpolation_start_time is None:
                                self.interpolation_start_time = current_time
                                with self.face_tracking_lock:
                                    current_translation = self.face_tracking_offsets[:3]
                                    current_rotation_euler = self.face_tracking_offsets[3:]
                                    pose_matrix = np.eye(4, dtype=np.float32)
                                    pose_matrix[:3, 3] = current_translation
                                    pose_matrix[:3, :3] = R.from_euler(
                                        "xyz",
                                        current_rotation_euler,
                                    ).as_matrix()
                                    self.interpolation_start_pose = pose_matrix

                            elapsed_interpolation = current_time - self.interpolation_start_time
                            t = min(1.0, elapsed_interpolation / self.INTERPOLATION_DURATION)

                            interpolated_pose = linear_pose_interpolation(
                                self.interpolation_start_pose,
                                neutral_pose,
                                t,
                            )

                            translation = interpolated_pose[:3, 3]
                            rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler(
                                "xyz", degrees=False
                            )

                            # Also decay the EMA state so it doesn't jump on re-detection
                            decay = 1.0 - t
                            self._ema_translation *= decay
                            self._ema_rotation *= decay

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],
                                ]

                            if t >= 1.0:
                                self.last_face_detected_time = None
                                self.interpolation_start_time = None
                                self.interpolation_start_pose = None
                                self._prev_eye_center = None

                time.sleep(self.LOOP_PERIOD)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")
