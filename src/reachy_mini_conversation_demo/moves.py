"""Movement system with sequential primary moves and additive secondary moves.

This module implements the movement architecture from main_works.py:
- Primary moves (sequential): emotions, dances, goto, breathing
- Secondary moves (additive): speech offsets + face tracking
- Single set_target() control point with pose fusion
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

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
        self.breathing_z_amplitude = 0.005  # 5mm gentle breathing
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


def clone_full_body_pose(pose: FullBodyPose) -> FullBodyPose:
    """Create a deep copy of a full body pose tuple."""
    head, antennas, body_yaw = pose
    return (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))


@dataclass
class MovementState:
    """State tracking for the movement system."""

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
    last_primary_pose: Optional[FullBodyPose] = None

    def update_activity(self) -> None:
        """Update the last activity time."""
        self.last_activity_time = time.time()


class MovementManager:
    """Enhanced movement manager that reproduces main_works.py behavior.

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
        """Initialize movement manager."""
        self.current_robot = current_robot
        self.head_tracker = head_tracker
        self.camera = camera
        self.camera_worker = camera_worker

        # Movement state
        self.state = MovementState()
        self.state.last_activity_time = time.time()
        neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)

        # Move queue (primary moves)
        self.move_queue = deque()

        # Configuration
        self.idle_inactivity_delay = 5.0  # seconds
        self.target_frequency = 50.0  # Hz
        self.target_period = 1.0 / self.target_frequency

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.RLock()
        self._is_listening = False
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(
            self.state.last_primary_pose
        )
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4  # seconds to blend back after listening
        self._last_listening_blend_time = time.monotonic()

    def queue_move(self, move: Move) -> None:
        """Add a move to the primary move queue."""
        with self._state_lock:
            self.move_queue.append(move)
            self.state.update_activity()
            logger.info(
                f"Queued move with duration {move.duration}s, queue size: {len(self.move_queue)}"
            )

    def clear_queue(self) -> None:
        """Clear all queued moves and stop current move."""
        with self._state_lock:
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self.state.is_playing_move = False
            logger.info("Cleared move queue and stopped current move")

    def set_speech_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Set speech head offsets (secondary move)."""
        with self._state_lock:
            self.state.speech_offsets = offsets

    def set_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Compatibility alias for set_speech_offsets."""
        self.set_speech_offsets(offsets)

    def set_face_tracking_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Set face tracking offsets (secondary move)."""
        with self._state_lock:
            self.state.face_tracking_offsets = offsets

    def set_moving_state(self, duration: float) -> None:
        """Set legacy moving state for goto moves."""
        with self._state_lock:
            self.state.moving_start = time.time()
            self.state.moving_for = duration
            self.state.update_activity()

    def is_idle(self):
        """Check if the robot is idle based on inactivity delay."""
        with self._state_lock:
            if self._is_listening:
                return False
            current_time = time.time()
            time_since_activity = current_time - self.state.last_activity_time
            return time_since_activity >= self.idle_inactivity_delay

    def mark_user_activity(self) -> None:
        """Record recent user activity to delay idle behaviours."""
        with self._state_lock:
            self.state.update_activity()

    def set_listening(self, listening: bool) -> None:
        """Toggle listening mode, freezing antennas when active."""
        with self._state_lock:
            if self._is_listening == listening:
                return
            self._is_listening = listening
            self._last_listening_blend_time = time.monotonic()
            if listening:
                # Capture the last antenna command so we keep that pose during listening
                self._listening_antennas = (
                    float(self._last_commanded_pose[1][0]),
                    float(self._last_commanded_pose[1][1]),
                )
                self._antenna_unfreeze_blend = 0.0
            self.state.update_activity()

    def _manage_move_queue(self, current_time: float) -> None:
        """Manage the primary move queue (sequential execution)."""
        with self._state_lock:
            if self.state.current_move is None or (
                self.state.move_start_time is not None
                and current_time - self.state.move_start_time
                >= self.state.current_move.duration
            ):
                self.state.current_move = None
                self.state.move_start_time = None

                if self.move_queue:
                    self.state.current_move = self.move_queue.popleft()
                    self.state.move_start_time = current_time
                    logger.info(
                        f"Starting new move, duration: {self.state.current_move.duration}s"
                    )

    def _manage_breathing(self, current_time: float) -> None:
        """Manage automatic breathing when idle."""
        # Start breathing after inactivity delay if no moves in queue
        with self._state_lock:
            if self.state.current_move is None and not self.move_queue:
                time_since_activity = current_time - self.state.last_activity_time

                if self.is_idle():
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
                        self.move_queue.append(breathing_move)
                        self.state.update_activity()
                        logger.info(
                            f"Started breathing after {time_since_activity:.1f}s of inactivity"
                        )
                    except Exception as e:
                        logger.error(f"Failed to start breathing: {e}")

            if (
                self.state.current_move is not None
                and isinstance(self.state.current_move, BreathingMove)
                and self.move_queue
            ):
                self.state.current_move = None
                self.state.move_start_time = None
                logger.info("Stopping breathing due to new move activity")

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get the primary full body pose from current move or neutral."""
        with self._state_lock:
            # When a primary move is playing, sample it and cache the resulting pose
            if (
                self.state.current_move is not None
                and self.state.move_start_time is not None
            ):
                move_time = current_time - self.state.move_start_time
                head, antennas, body_yaw = self.state.current_move.evaluate(move_time)

                if head is None:
                    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
                if antennas is None:
                    antennas = np.array([0.0, 0.0])
                if body_yaw is None:
                    body_yaw = 0.0

                antennas_tuple = (float(antennas[0]), float(antennas[1]))
                head_copy = head.copy()
                primary_full_body_pose = (
                    head_copy,
                    antennas_tuple,
                    float(body_yaw),
                )

                self.state.is_playing_move = True
                self.state.is_moving = True
                self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)
            else:
                # Otherwise reuse the last primary pose so we avoid jumps between moves
                self.state.is_playing_move = False
                self.state.is_moving = (
                    time.time() - self.state.moving_start < self.state.moving_for
                )

                if self.state.last_primary_pose is not None:
                    primary_full_body_pose = clone_full_body_pose(
                        self.state.last_primary_pose
                    )
                else:
                    neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
                    primary_full_body_pose = (neutral_head_pose, (0.0, 0.0), 0.0)
                    self.state.last_primary_pose = clone_full_body_pose(
                        primary_full_body_pose
                    )

        return primary_full_body_pose

    def _get_secondary_pose(self) -> FullBodyPose:
        """Get the secondary full body pose from speech and face tracking offsets."""
        # Combine speech sway offsets + face tracking offsets for secondary pose
        with self._state_lock:
            secondary_offsets = [
                self.state.speech_offsets[0] + self.state.face_tracking_offsets[0],
                self.state.speech_offsets[1] + self.state.face_tracking_offsets[1],
                self.state.speech_offsets[2] + self.state.face_tracking_offsets[2],
                self.state.speech_offsets[3] + self.state.face_tracking_offsets[3],
                self.state.speech_offsets[4] + self.state.face_tracking_offsets[4],
                self.state.speech_offsets[5] + self.state.face_tracking_offsets[5],
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
        """Get face tracking offsets from camera worker thread."""
        if self.camera_worker is not None:
            # Get face tracking offsets from camera worker thread
            offsets = self.camera_worker.get_face_tracking_offsets()
            with self._state_lock:
                self.state.face_tracking_offsets = offsets
        else:
            # No camera worker, use neutral offsets
            with self._state_lock:
                self.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def start(self) -> None:
        """Start the move worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.info("Move worker started")

    def stop(self) -> None:
        """Stop the move worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.info("Move worker stopped")

    def working_loop(self) -> None:
        """Control loop main movements - reproduces main_works.py control architecture.

        Single set_target() call with pose fusion.
        """
        logger.debug("Starting enhanced movement control loop (50Hz)")

        loop_count = 0
        last_print_time = time.time()

        while not self._stop_event.is_set():
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
            now_monotonic = time.monotonic()
            with self._state_lock:
                listening = self._is_listening
                listening_antennas = self._listening_antennas
                blend = self._antenna_unfreeze_blend
                blend_duration = self._antenna_blend_duration
                last_update = self._last_listening_blend_time
                self._last_listening_blend_time = now_monotonic

            # Blend antenna outputs back to the live motion when leaving listening mode
            if listening:
                antennas_cmd = listening_antennas
                new_blend = 0.0
            else:
                dt = max(0.0, now_monotonic - last_update)
                if blend_duration <= 0:
                    new_blend = 1.0
                else:
                    new_blend = min(1.0, blend + dt / blend_duration)
                antennas_cmd = (
                    listening_antennas[0] * (1.0 - new_blend)
                    + antennas[0] * new_blend,
                    listening_antennas[1] * (1.0 - new_blend)
                    + antennas[1] * new_blend,
                )

            with self._state_lock:
                if listening:
                    self._antenna_unfreeze_blend = 0.0
                else:
                    self._antenna_unfreeze_blend = new_blend
                    if new_blend >= 1.0:
                        self._listening_antennas = (
                            float(antennas[0]),
                            float(antennas[1]),
                        )

            # 8. Single set_target call - the one and only place we control the robot
            try:
                self.current_robot.set_target(
                    head=head, antennas=antennas_cmd, body_yaw=body_yaw
                )
            except Exception as e:
                logger.error(f"Failed to set robot target: {e}")
            else:
                with self._state_lock:
                    self._last_commanded_pose = clone_full_body_pose(
                        (head, antennas_cmd, body_yaw)
                    )

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
                last_print_time = current_time

            time.sleep(sleep_time)

        logger.debug("Movement control loop stopped")
