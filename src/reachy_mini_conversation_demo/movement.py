import time
import asyncio
import logging
import threading
import numpy as np
import scipy
import cv2

from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import IMAGE_SIZE
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)


class MovementManager:
    def __init__(
        self,
        current_robot: ReachyMini,
        head_tracker,
        camera: cv2.VideoCapture | None,
    ):
        self.current_robot = current_robot
        self.head_tracker = head_tracker
        self.camera = camera

        # default values
        self.current_head_pose = np.eye(4)
        self.moving_start = time.monotonic()
        self.moving_for = 0.0
        self.speech_head_offsets = [0.0] * 6
        self.movement_loop_sleep = 0.05  # seconds

    def set_offsets(self, offsets: list[float]) -> None:
        """Used by AudioSync callback to update speech offsets"""
        self.speech_head_offsets = list(offsets)

    def set_neutral(self) -> None:
        """Set neutral robot position"""
        self.speech_head_offsets = [0.0] * 6
        self.current_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.current_robot.set_target(head=self.current_head_pose, antennas=(0.0, 0.0))

    def reset_head_pose(self) -> None:
        self.current_head_pose = np.eye(4)

    async def enable(self, stop_event: threading.Event) -> None:
        logger.info("Starting head movement loop")
        debug_frame_count = 0
        while not stop_event.is_set():
            debug_frame_count += 1
            current_time = time.time()

            # Head tracking
            if self.head_tracker is not None:
                success, im = self.camera.read()
                if not success:
                    if current_time - last_log_ts > 1.5:
                        logger.warning("Camera read failed")
                        last_log_ts = current_time
                else:
                    eye_center, _ = self.head_tracker.get_head_position(
                        im
                    )  # as [-1, 1]

                    if eye_center is not None:
                        # Rescale target position into IMAGE_SIZE coordinates
                        w, h = IMAGE_SIZE
                        eye_center = (eye_center + 1) / 2
                        eye_center[0] *= w
                        eye_center[1] *= h

                        # Bounds checking
                        eye_center = np.clip(eye_center, [0, 0], [w - 1, h - 1])

                        current_head_pose = self.current_robot.look_at_image(
                            *eye_center, duration=0.0, perform_movement=False
                        )

                        self.current_head_pose = current_head_pose
            # Pose calculation
            try:
                current_x, current_y, current_z = self.current_head_pose[:3, 3]

                current_roll, current_pitch, current_yaw = (
                    scipy.spatial.transform.Rotation.from_matrix(
                        self.current_head_pose[:3, :3]
                    ).as_euler("xyz", degrees=False)
                )

                if debug_frame_count % 50 == 0:
                    logger.debug(
                        "Current pose XYZ: %.3f, %.3f, %.3f",
                        current_x,
                        current_y,
                        current_z,
                    )
                    logger.debug(
                        "Current angles: roll=%.3f, pitch=%.3f, yaw=%.3f",
                        current_roll,
                        current_pitch,
                        current_yaw,
                    )

            except Exception as e:
                logger.exception("Invalid pose; resetting")
                self.reset_head_pose()
                current_x, current_y, current_z = self.current_head_pose[:3, 3]
                current_roll = current_pitch = current_yaw = 0.0

            # Movement check
            is_moving = time.monotonic() - self.moving_start < self.moving_for

            if debug_frame_count % 50 == 0:
                logger.debug(f"Robot moving: {is_moving}")

            # Apply speech offsets when not moving
            if not is_moving:
                try:
                    head_pose = create_head_pose(
                        x=current_x + self.speech_head_offsets[0],
                        y=current_y + self.speech_head_offsets[1],
                        z=current_z + self.speech_head_offsets[2],
                        roll=current_roll + self.speech_head_offsets[3],
                        pitch=current_pitch + self.speech_head_offsets[4],
                        yaw=current_yaw + self.speech_head_offsets[5],
                        degrees=False,
                        mm=False,
                    )

                    if debug_frame_count % 50 == 0:
                        logger.debug(
                            "Final head pose with offsets: %s", head_pose[:3, 3]
                        )
                        logger.debug("Speech offsets: %s", self.speech_head_offsets)

                    self.current_robot.set_target(head=head_pose, antennas=(0.0, 0.0))

                    if debug_frame_count % 50 == 0:
                        logger.debug("Sent pose to robot successfully")

                except Exception as e:
                    logger.debug("Failed to set robot target: %s", e)

            await asyncio.sleep(self.movement_loop_sleep)

        logger.info("Exited head movement loop")
