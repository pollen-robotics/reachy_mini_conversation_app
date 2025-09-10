"""Camera worker thread with frame buffering and face tracking.

Ported from main_works.py camera_worker() function to provide:
- 30Hz+ camera polling with thread-safe frame buffering
- Face tracking integration with smooth interpolation
- Latest frame always available for tools
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(
        self, camera: cv2.VideoCapture, reachy_mini: ReachyMini, head_tracker=None
    ):
        """Initialize."""
        self.camera = camera
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        # Thread-safe frame storage
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # Face tracking state
        self.is_head_tracking_enabled = True
        self.face_tracking_offsets = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()

        # Face tracking timing variables (same as main_works.py)
        self.last_face_detected_time: Optional[float] = None
        self.interpolation_start_time: Optional[float] = None
        self.interpolation_start_pose: Optional[np.ndarray] = None
        self.face_lost_delay = 2.0  # seconds to wait before starting interpolation
        self.interpolation_duration = 1.0  # seconds to interpolate back to neutral

        # Track state changes
        self.previous_head_tracking_state = self.is_head_tracking_enabled

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self.face_tracking_lock:
            return tuple(self.face_tracking_offsets)

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking."""
        self.is_head_tracking_enabled = enabled
        logger.info(f"Head tracking {'enabled' if enabled else 'disabled'}")

    async def enable(self, stop_event: asyncio.Event) -> None:
        """Enable the camera worker loop.

        Ported from main_works.py camera_worker() with same logic.
        """
        logger.info("Starting camera worker")

        # Initialize head tracker if available
        neutral_pose = np.eye(4)  # Neutral pose (identity matrix)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not stop_event.is_set():
            try:
                current_time = time.time()
                success, frame = self.camera.read()

                if success:
                    # Thread-safe frame storage
                    with self.frame_lock:
                        self.latest_frame = frame.copy()

                    # Check if face tracking was just disabled
                    if (
                        self.previous_head_tracking_state
                        and not self.is_head_tracking_enabled
                    ):
                        # Face tracking was just disabled - start interpolation to neutral
                        self.last_face_detected_time = (
                            current_time  # Trigger the face-lost logic
                        )
                        self.interpolation_start_time = (
                            None  # Will be set by the face-lost interpolation
                        )
                        self.interpolation_start_pose = None

                    # Update tracking state
                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    # Handle face tracking if enabled and head tracker available
                    if self.is_head_tracking_enabled and self.head_tracker is not None:
                        eye_center, _ = self.head_tracker.get_head_position(frame)

                        if eye_center is not None:
                            # Face detected - immediately switch to tracking
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = (
                                None  # Stop any interpolation
                            )

                            # Convert normalized coordinates to pixel coordinates
                            h, w, _ = frame.shape
                            eye_center_norm = (eye_center + 1) / 2
                            eye_center_pixels = [
                                eye_center_norm[0] * w,
                                eye_center_norm[1] * h,
                            ]

                            # Get the head pose needed to look at the target, but don't perform movement
                            target_pose = self.reachy_mini.look_at_image(
                                eye_center_pixels[0],
                                eye_center_pixels[1],
                                duration=0.0,
                                perform_movement=False,
                            )

                            # Extract translation and rotation from the target pose directly
                            translation = target_pose[:3, 3]
                            rotation = R.from_matrix(target_pose[:3, :3]).as_euler(
                                "xyz", degrees=False
                            )

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

                        else:
                            # No face detected while tracking enabled - set face lost timestamp
                            if (
                                self.last_face_detected_time is None
                                or self.last_face_detected_time == current_time
                            ):
                                # Only update if we haven't already set a face lost time
                                # (current_time check prevents overriding the disable-triggered timestamp)
                                pass

                    # Handle smooth interpolation (works for both face-lost and tracking-disabled cases)
                    if self.last_face_detected_time is not None:
                        time_since_face_lost = (
                            current_time - self.last_face_detected_time
                        )

                        if time_since_face_lost >= self.face_lost_delay:
                            # Start interpolation if not already started
                            if self.interpolation_start_time is None:
                                self.interpolation_start_time = current_time
                                # Capture current pose as start of interpolation
                                with self.face_tracking_lock:
                                    current_translation = self.face_tracking_offsets[:3]
                                    current_rotation_euler = self.face_tracking_offsets[
                                        3:
                                    ]
                                    # Convert to 4x4 pose matrix
                                    self.interpolation_start_pose = np.eye(4)
                                    self.interpolation_start_pose[:3, 3] = (
                                        current_translation
                                    )
                                    self.interpolation_start_pose[:3, :3] = (
                                        R.from_euler(
                                            "xyz", current_rotation_euler
                                        ).as_matrix()
                                    )

                            # Calculate interpolation progress (t from 0 to 1)
                            elapsed_interpolation = (
                                current_time - self.interpolation_start_time
                            )
                            t = min(
                                1.0, elapsed_interpolation / self.interpolation_duration
                            )

                            # Interpolate between current pose and neutral pose
                            interpolated_pose = linear_pose_interpolation(
                                self.interpolation_start_pose, neutral_pose, t
                            )

                            # Extract translation and rotation from interpolated pose
                            translation = interpolated_pose[:3, 3]
                            rotation = R.from_matrix(
                                interpolated_pose[:3, :3]
                            ).as_euler("xyz", degrees=False)

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
                await asyncio.sleep(0.001)
                # time.sleep(0.001)

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)  # Longer sleep on error

        logger.info("Camera worker thread exited")
