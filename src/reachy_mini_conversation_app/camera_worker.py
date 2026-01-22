"""Camera worker thread with frame buffering and face tracking.

Ported from main_works.py camera_worker() function to provide:
- 30Hz+ camera polling with thread-safe frame buffering
- Face tracking integration with smooth interpolation
- Latest frame always available for tools
"""

import time
import logging
import threading
from typing import Any, List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation


logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(
        self,
        reachy_mini: ReachyMini,
        head_tracker: Any = None,
        enable_preview: bool = False,
        preview_window_name: str = "ReachyMini POV",
        preview_window_normal: bool = True,
    ) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        # Preview vision params
        self._last_eye_center: NDArray[np.float32] | None = None
        self.enable_preview = enable_preview
        self.preview_window_name = preview_window_name
        self.preview_window_normal = preview_window_normal
        self._preview_initialized = False
        self.show_head_overlay = False

        # Thread-safe frame storage
        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Face tracking state
        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()

        # Face tracking timing variables (same as main_works.py)
        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None
        self.face_lost_delay = 2.0  # seconds to wait before starting interpolation
        self.interpolation_duration = 1.0  # seconds to interpolate back to neutral

        # Track state changes
        self.previous_head_tracking_state = self.is_head_tracking_enabled

    def eye_center_pixels(
        self,
        frame: NDArray[np.uint8],
        eye_center: NDArray[np.float32],
    ) -> list[float]:
        h, w, _ = frame.shape
        eye_center_norm = (eye_center + 1) / 2
        return [
            eye_center_norm[0] * w,
            eye_center_norm[1] * h,
        ]

    def handle_preview_keys(self) -> None:
        key = cv2.waitKey(1) & 0xFF

        # Disable preview by showing black screen
        if key == ord("d"):
            self.enable_preview = False
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            self.init_preview_window()
            cv2.imshow(self.preview_window_name, blank)
        # Enable preview
        elif key == ord("p"):
            self.enable_preview = True
        # Enable head tracking
        elif key == ord("h"):
            self.show_head_overlay = not self.show_head_overlay

    def init_preview_window(self) -> None:
        if self._preview_initialized:
            return
        flags = cv2.WINDOW_NORMAL if self.preview_window_normal else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(self.preview_window_name, flags)
        self._preview_initialized = True

    def destroy_preview_window(self) -> None:
        if not self._preview_initialized:
            return
        try:
            cv2.destroyWindow(self.preview_window_name)
        except Exception:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        self._preview_initialized = False

    def preview_step(self, frame: NDArray[np.uint8]) -> None:
        self.init_preview_window()

        display = frame
        if self.show_head_overlay and self.head_tracker is not None:
            display = frame.copy()

            eye_center = self._last_eye_center
            if eye_center is not None:
                eye_center_pixels = self.eye_center_pixels(frame, eye_center)
                cx, cy = int(eye_center_pixels[0]), int(eye_center_pixels[1])

                box_size = 300
                x1, y1 = cx - box_size // 2, cy - box_size // 2
                x2, y2 = cx + box_size // 2, cy + box_size // 2

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display, "Head overlay ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )
            else:
                cv2.putText(
                    display,
                    "Head overlay ON (no head)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow(self.preview_window_name, display)

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            # Return a copy in original BGR format (OpenCV native)
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
        self.destroy_preview_window()
        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Enable the camera worker loop.

        Ported from main_works.py camera_worker() with same logic.
        """
        logger.debug("Starting camera working loop")

        # Initialize head tracker if available
        neutral_pose = np.eye(4)  # Neutral pose (identity matrix)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Get frame from robot
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    # Thread-safe frame storage
                    with self.frame_lock:
                        self.latest_frame = frame  # .copy()

                    # Preview the frame
                    if self.enable_preview or self._preview_initialized:
                        self.handle_preview_keys()
                        if self.enable_preview:
                            self.preview_step(frame)

                    # Check if face tracking was just disabled
                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        # Face tracking was just disabled - start interpolation to neutral
                        self.last_face_detected_time = current_time  # Trigger the face-lost logic
                        # Will be set by the face-lost interpolation
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None

                    # Update tracking state
                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    # Handle face tracking if enabled and head tracker available
                    if self.is_head_tracking_enabled and self.head_tracker is not None:
                        eye_center, _ = self.head_tracker.get_head_position(frame)
                        self._last_eye_center = eye_center

                        if eye_center is not None:
                            # Face detected - immediately switch to tracking
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = None  # Stop any interpolation

                            # Convert normalized coordinates to pixel coordinates
                            eye_center_pixels = self.eye_center_pixels(frame, eye_center)

                            # Get the head pose needed to look at the target, but don't perform movement
                            target_pose = self.reachy_mini.look_at_image(
                                eye_center_pixels[0],
                                eye_center_pixels[1],
                                duration=0.0,
                                perform_movement=False,
                            )

                            # Extract translation and rotation from the target pose directly
                            translation = target_pose[:3, 3]
                            rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

                            # Scale down translation and rotation because smaller FOV
                            translation *= 0.6
                            rotation *= 0.6

                            # Thread-safe update of face tracking offsets (use pose as-is)
                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],  # x, y, z
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],  # roll, pitch, yaw
                                ]

                        # No face detected while tracking enabled - set face lost timestamp
                        elif self.last_face_detected_time is None or self.last_face_detected_time == current_time:
                            # Only update if we haven't already set a face lost time
                            # (current_time check prevents overriding the disable-triggered timestamp)
                            pass

                    # Handle smooth interpolation (works for both face-lost and tracking-disabled cases)
                    if self.last_face_detected_time is not None:
                        time_since_face_lost = current_time - self.last_face_detected_time

                        if time_since_face_lost >= self.face_lost_delay:
                            # Start interpolation if not already started
                            if self.interpolation_start_time is None:
                                self.interpolation_start_time = current_time
                                # Capture current pose as start of interpolation
                                with self.face_tracking_lock:
                                    current_translation = self.face_tracking_offsets[:3]
                                    current_rotation_euler = self.face_tracking_offsets[3:]
                                    # Convert to 4x4 pose matrix
                                    pose_matrix = np.eye(4, dtype=np.float32)
                                    pose_matrix[:3, 3] = current_translation
                                    pose_matrix[:3, :3] = R.from_euler(
                                        "xyz",
                                        current_rotation_euler,
                                    ).as_matrix()
                                    self.interpolation_start_pose = pose_matrix

                            # Calculate interpolation progress (t from 0 to 1)
                            elapsed_interpolation = current_time - self.interpolation_start_time
                            t = min(1.0, elapsed_interpolation / self.interpolation_duration)

                            # Interpolate between current pose and neutral pose
                            interpolated_pose = linear_pose_interpolation(
                                self.interpolation_start_pose,
                                neutral_pose,
                                t,
                            )

                            # Extract translation and rotation from interpolated pose
                            translation = interpolated_pose[:3, 3]
                            rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

                            # Thread-safe update of face tracking offsets
                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],  # x, y, z
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],  # roll, pitch, yaw
                                ]

                            # If interpolation is complete, reset timing
                            if t >= 1.0:
                                self.last_face_detected_time = None
                                self.interpolation_start_time = None
                                self.interpolation_start_pose = None
                        # else: Keep current offsets (within 2s delay period)

                # Small sleep to prevent excessive CPU usage (same as main_works.py)
                time.sleep(0.04)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)  # Longer sleep on error

        # OpenCV cleanup
        self.destroy_preview_window()

        logger.debug("Camera worker thread exited")
