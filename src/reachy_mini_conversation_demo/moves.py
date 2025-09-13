"""
Movement system with sequential primary moves and additive secondary moves.

This module implements the movement architecture from main_works.py:
- Primary moves (sequential): emotions, dances, goto, breathing
- Secondary moves (additive): speech offsets + face tracking
- Single set_target() control point with pose fusion
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import (
    compose_world_offset,
    linear_pose_interpolation,
)

logger = logging.getLogger(__name__)

# Type definitions
FullBodyPose = Tuple[
    np.ndarray, Tuple[float, float], float
]  # (head_pose_4x4, antennas, body_yaw)


class BreathingMove(Move):
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose: np.ndarray,
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ):
        """Initialize breathing move.

        Args:
            interpolation_start_pose: 4x4 matrix of current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions to interpolate from
            interpolation_duration: Duration of interpolation to neutral (seconds)
        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        # Neutral positions for breathing base
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])

        # Breathing parameters
        self.breathing_z_amplitude = 0.01  # 1cm gentle breathing
        self.breathing_frequency = 0.1  # Hz (6 breaths per minute)
        self.antenna_sway_amplitude = np.deg2rad(15)  # 15 degrees
        self.antenna_frequency = 0.5  # Hz (faster antenna sway)

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous breathing (never ends naturally)

    def evaluate(
        self, t: float
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        """Evaluate breathing move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to neutral base position
            interpolation_t = t / self.interpolation_duration

            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose, self.neutral_head_pose, interpolation_t
            )

            # Interpolate antennas
            antennas = (
                (1 - interpolation_t) * self.interpolation_start_antennas
                + interpolation_t * self.neutral_antennas
            )

        else:
            # Phase 2: Breathing patterns from neutral base
            breathing_time = t - self.interpolation_duration

            # Gentle z-axis breathing
            z_offset = self.breathing_z_amplitude * np.sin(
                2 * np.pi * self.breathing_frequency * breathing_time
            )
            head_pose = create_head_pose(
                x=0, y=0, z=z_offset, roll=0, pitch=0, yaw=0, degrees=True, mm=False
            )

            # Antenna sway (opposite directions)
            antenna_sway = self.antenna_sway_amplitude * np.sin(
                2 * np.pi * self.antenna_frequency * breathing_time
            )
            antennas = np.array([antenna_sway, -antenna_sway])

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        return (head_pose, antennas, 0.0)


def combine_full_body(
    primary_pose: FullBodyPose, secondary_pose: FullBodyPose
) -> FullBodyPose:
    """Combine primary and secondary full body poses.

    Args:
        primary_pose: (head_pose, antennas, body_yaw) - primary move
        secondary_pose: (head_pose, antennas, body_yaw) - secondary offsets

    Returns:
        Combined full body pose (head_pose, antennas, body_yaw)
    """
    primary_head, primary_antennas, primary_body_yaw = primary_pose
    secondary_head, secondary_antennas, secondary_body_yaw = secondary_pose

    # Combine head poses using compose_world_offset
    # primary_head is T_abs, secondary_head is T_off_world
    combined_head = compose_world_offset(
        primary_head, secondary_head, reorthonormalize=True
    )

    # Sum antennas and body_yaw
    combined_antennas = (
        primary_antennas[0] + secondary_antennas[0],
        primary_antennas[1] + secondary_antennas[1],
    )
    combined_body_yaw = primary_body_yaw + secondary_body_yaw

    return (combined_head, combined_antennas, combined_body_yaw)


@dataclass
class MovementState:
    """State tracking for the movement system"""

    # Primary move state
    current_move: Optional[Move] = None
    move_start_time: Optional[float] = None
    last_activity_time: float = 0.0

    # Secondary move state (offsets)
    speech_offsets: Tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    face_tracking_offsets: Tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    # Legacy movement state (for goto moves)
    moving_start: float = 0.0
    moving_for: float = 0.0

    # Status flags
    is_playing_move: bool = False
    is_moving: bool = False

    def update_activity(self) -> None:
        """Update the last activity time"""
        self.last_activity_time = time.time()


class MovementManager:
    """
    Enhanced movement manager that reproduces main_works.py behavior:
    - Sequential primary moves via queue
    - Additive secondary moves (speech + face tracking)
    - Single set_target control loop with pose fusion
    - Automatic breathing after inactivity
    """

    def __init__(
        self,
        current_robot: ReachyMini,
        head_tracker=None,
        camera=None,
        camera_worker=None,
    ):
        self.current_robot = current_robot
        self.head_tracker = head_tracker
        self.camera = camera
        self.camera_worker = camera_worker

        # Movement state
        self.state = MovementState()
        self.state.last_activity_time = time.time()

        # Move queue (primary moves)
        self.move_queue = deque()

        # Configuration
        self.breathing_inactivity_delay = 5.0  # seconds
        self.target_frequency = 50.0  # Hz
        self.target_period = 1.0 / self.target_frequency

    def queue_move(self, move: Move) -> None:
        """Add a move to the primary move queue"""
        self.move_queue.append(move)
        self.state.update_activity()
        logger.info(
            f"Queued move with duration {move.duration}s, queue size: {len(self.move_queue)}"
        )

    def clear_queue(self) -> None:
        """Clear all queued moves and stop current move"""
        self.move_queue.clear()
        self.state.current_move = None
        self.state.move_start_time = None
        self.state.is_playing_move = False
        logger.info("Cleared move queue and stopped current move")

    def set_speech_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Set speech head offsets (secondary move)"""
        self.state.speech_offsets = offsets

    def set_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Compatibility alias for set_speech_offsets"""
        self.set_speech_offsets(offsets)

    def set_face_tracking_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Set face tracking offsets (secondary move)"""
        self.state.face_tracking_offsets = offsets

    def set_moving_state(self, duration: float) -> None:
        """Set legacy moving state for goto moves"""
        self.state.moving_start = time.time()
        self.state.moving_for = duration
        self.state.update_activity()

    def _manage_move_queue(self, current_time: float) -> None:
        """Manage the primary move queue (sequential execution)"""
        # Check if current move is finished
        if self.state.current_move is None or (
            self.state.move_start_time is not None
            and current_time - self.state.move_start_time
            >= self.state.current_move.duration
        ):
            # Current move finished or no current move, get next from queue
            self.state.current_move = None
            self.state.move_start_time = None

            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                logger.info(
                    f"Starting new move, duration: {self.state.current_move.duration}s"
                )

    def _manage_breathing(self, current_time: float) -> None:
        """Manage automatic breathing when idle"""
        # Start breathing after inactivity delay if no moves in queue
        if self.state.current_move is None and not self.move_queue:
            time_since_activity = current_time - self.state.last_activity_time
            if time_since_activity >= self.breathing_inactivity_delay:
                # Start breathing move
                try:
                    _, current_antennas = (
                        self.current_robot.get_current_joint_positions()
                    )
                    current_head_pose = self.current_robot.get_current_head_pose()

                    breathing_move = BreathingMove(
                        interpolation_start_pose=current_head_pose,
                        interpolation_start_antennas=current_antennas,
                        interpolation_duration=1.0,
                    )
                    self.queue_move(breathing_move)
                    logger.info(
                        f"Started breathing after {time_since_activity:.1f}s of inactivity"
                    )
                except Exception as e:
                    logger.error(f"Failed to start breathing: {e}")

        # Stop breathing if new activity detected (queue has non-breathing moves)
        if (
            self.state.current_move is not None
            and isinstance(self.state.current_move, BreathingMove)
            and self.move_queue
        ):
            # There are new moves waiting, stop breathing immediately
            self.state.current_move = None
            self.state.move_start_time = None
            logger.info("Stopping breathing due to new move activity")

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get the primary full body pose from current move or neutral"""
        if (
            self.state.current_move is not None
            and self.state.move_start_time is not None
        ):
            move_time = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(move_time)

            # Convert official Move interface to FullBodyPose format
            # Handle None values from official interface
            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([0.0, 0.0])
            if body_yaw is None:
                body_yaw = 0.0

            # Convert antennas to tuple format for FullBodyPose
            antennas_tuple = (float(antennas[0]), float(antennas[1]))
            primary_full_body_pose = (head, antennas_tuple, float(body_yaw))

            self.state.is_playing_move = True
            self.state.is_moving = True
        else:
            # Default neutral pose when no move is playing
            self.state.is_playing_move = False
            self.state.is_moving = (
                time.time() - self.state.moving_start < self.state.moving_for
            )
            # Neutral primary pose
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            primary_full_body_pose = (neutral_head_pose, (0.0, 0.0), 0.0)

        return primary_full_body_pose

    def _get_secondary_pose(self) -> FullBodyPose:
        """Get the secondary full body pose from speech and face tracking offsets"""
        # Combine speech sway offsets + face tracking offsets for secondary pose
        secondary_offsets = [
            self.state.speech_offsets[0] + self.state.face_tracking_offsets[0],  # x
            self.state.speech_offsets[1] + self.state.face_tracking_offsets[1],  # y
            self.state.speech_offsets[2] + self.state.face_tracking_offsets[2],  # z
            self.state.speech_offsets[3] + self.state.face_tracking_offsets[3],  # roll
            self.state.speech_offsets[4] + self.state.face_tracking_offsets[4],  # pitch
            self.state.speech_offsets[5] + self.state.face_tracking_offsets[5],  # yaw
        ]

        secondary_head_pose = create_head_pose(
            x=secondary_offsets[0],
            y=secondary_offsets[1],
            z=secondary_offsets[2],
            roll=secondary_offsets[3],
            pitch=secondary_offsets[4],
            yaw=secondary_offsets[5],
            degrees=False,
            mm=False,
        )
        return (secondary_head_pose, (0, 0), 0)

    def _update_face_tracking(self, current_time: float) -> None:
        """Get face tracking offsets from camera worker thread"""
        if self.camera_worker is not None:
            # Get face tracking offsets from camera worker thread
            self.state.face_tracking_offsets = (
                self.camera_worker.get_face_tracking_offsets()
            )
        else:
            # No camera worker, use neutral offsets
            self.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def set_neutral(self) -> None:
        """Set neutral robot position"""
        self.state.speech_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.current_robot.set_target(head=neutral_head_pose, antennas=(0.0, 0.0))

    async def enable(self, stop_event: asyncio.Event) -> None:
        """
        Main movement control loop - reproduces main_works.py control architecture.
        Single set_target() call with pose fusion.
        """
        logger.info("Starting enhanced movement control loop (50Hz)")

        loop_count = 0
        last_print_time = time.time()

        while not stop_event.is_set():
            loop_start_time = time.time()
            loop_count += 1
            current_time = time.time()

            # 1. Manage move queue (sequential primary moves)
            self._manage_move_queue(current_time)

            # 2. Manage breathing (automatic idle behavior)
            self._manage_breathing(current_time)

            # 3. Update face tracking offsets
            self._update_face_tracking(current_time)

            # 4. Get primary pose from current move or neutral
            primary_full_body_pose = self._get_primary_pose(current_time)

            # 5. Get secondary pose from speech and face tracking offsets
            secondary_full_body_pose = self._get_secondary_pose()

            # 6. Combine primary and secondary poses
            global_full_body_pose = combine_full_body(
                primary_full_body_pose, secondary_full_body_pose
            )

            # 7. Extract pose components
            head, antennas, body_yaw = global_full_body_pose

            # 8. Single set_target call - the one and only place we control the robot
            try:
                self.current_robot.set_target(
                    head=head, antennas=antennas, body_yaw=body_yaw
                )
            except Exception as e:
                logger.error(f"Failed to set robot target: {e}")

            # 9. Calculate computation time and adjust sleep for 50Hz
            computation_time = time.time() - loop_start_time
            sleep_time = max(0, self.target_period - computation_time)

            # 10. Print frequency info every 100 loops (~2 seconds)
            if loop_count % 100 == 0:
                elapsed = current_time - last_print_time
                actual_freq = 100.0 / elapsed if elapsed > 0 else 0
                potential_freq = (
                    1.0 / computation_time if computation_time > 0 else float("inf")
                )
                logger.debug(
                    f"Loop freq - Actual: {actual_freq:.1f}Hz, Potential: {potential_freq:.1f}Hz, Target: {self.target_frequency:.1f}Hz"
                )
                print(
                    f"Loop freq - Actual: {actual_freq:.1f}Hz, Potential: {potential_freq:.1f}Hz, Target: {self.target_frequency:.1f}Hz"
                )
                last_print_time = current_time

            await asyncio.sleep(sleep_time)

        logger.info("Movement control loop stopped")
