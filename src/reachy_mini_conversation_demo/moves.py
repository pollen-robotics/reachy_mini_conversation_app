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
from queue import Empty, Queue
from typing import Any, Optional, Tuple

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
        self.last_activity_time = time.monotonic()


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
        camera_worker=None,
    ):
        """Initialize movement manager."""
        self.current_robot = current_robot
        self.head_tracker = head_tracker
        self.camera_worker = camera_worker

        # Single timing source for durations
        self._now = time.monotonic

        # Movement state
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)

        # Move queue (primary moves)
        self.move_queue = deque()

        # Configuration
        self.idle_inactivity_delay = 0.3  # seconds
        self.target_frequency = 100.0  # Hz
        self.target_period = 1.0 / self.target_frequency

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_listening = False
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(
            self.state.last_primary_pose
        )
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4  # seconds to blend back after listening
        self._last_listening_blend_time = self._now()
        self._breathing_active = False  # true when breathing move is running or queued

        # Cross-thread signalling
        self._command_queue: Queue[tuple[str, Any]] = Queue()
        self._speech_offsets_lock = threading.Lock()
        self._pending_speech_offsets: Tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._speech_offsets_dirty = False

        self._face_offsets_lock = threading.Lock()
        self._pending_face_offsets: Tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._face_offsets_dirty = False

        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self.state.last_activity_time
        self._shared_is_listening = self._is_listening

    def queue_move(self, move: Move) -> None:
        """Add a move to the primary move queue."""
        self._command_queue.put(("queue_move", move))

    def clear_queue(self) -> None:
        """Clear all queued moves and stop current move."""
        self._command_queue.put(("clear_queue", None))

    def set_speech_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Set speech head offsets (secondary move)."""
        with self._speech_offsets_lock:
            self._pending_speech_offsets = offsets
            self._speech_offsets_dirty = True

    def set_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Compatibility alias for set_speech_offsets."""
        self.set_speech_offsets(offsets)

    def set_face_tracking_offsets(
        self, offsets: Tuple[float, float, float, float, float, float]
    ) -> None:
        """Set face tracking offsets (secondary move)."""
        with self._face_offsets_lock:
            self._pending_face_offsets = offsets
            self._face_offsets_dirty = True

    def set_moving_state(self, duration: float) -> None:
        """Set legacy moving state for goto moves."""
        self._command_queue.put(("set_moving_state", duration))

    def is_idle(self):
        """Check if the robot is idle based on inactivity delay."""
        with self._shared_state_lock:
            last_activity = self._shared_last_activity_time
            listening = self._shared_is_listening

        if listening:
            return False

        current_time = self._now()
        time_since_activity = current_time - last_activity
        return time_since_activity >= self.idle_inactivity_delay

    def mark_user_activity(self) -> None:
        """Record recent user activity to delay idle behaviours."""
        self._command_queue.put(("mark_activity", None))

    def set_listening(self, listening: bool) -> None:
        """Toggle listening mode, freezing antennas when active."""
        with self._shared_state_lock:
            if self._shared_is_listening == listening:
                return
        self._command_queue.put(("set_listening", listening))

    def _poll_signals(self, current_time: float) -> None:
        """Apply queued commands and pending offset updates."""
        self._apply_pending_offsets()

        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload, current_time)

    def _apply_pending_offsets(self) -> None:
        """Apply the most recent speech/face offset updates."""
        speech_offsets: Optional[Tuple[float, float, float, float, float, float]] = None
        with self._speech_offsets_lock:
            if self._speech_offsets_dirty:
                speech_offsets = self._pending_speech_offsets
                self._speech_offsets_dirty = False

        if speech_offsets is not None:
            self.state.speech_offsets = speech_offsets
            self.state.update_activity()

        face_offsets: Optional[Tuple[float, float, float, float, float, float]] = None
        with self._face_offsets_lock:
            if self._face_offsets_dirty:
                face_offsets = self._pending_face_offsets
                self._face_offsets_dirty = False

        if face_offsets is not None:
            self.state.face_tracking_offsets = face_offsets
            self.state.update_activity()

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        """Handle a single cross-thread command."""
        if command == "queue_move":
            if isinstance(payload, Move):
                self.move_queue.append(payload)
                self.state.update_activity()
                duration = getattr(payload, "duration", None)
                if duration is not None:
                    try:
                        duration_str = f"{float(duration):.2f}"
                    except (TypeError, ValueError):
                        duration_str = str(duration)
                else:
                    duration_str = "?"
                logger.info(
                    "Queued move with duration %ss, queue size: %s",
                    duration_str,
                    len(self.move_queue),
                )
            else:
                logger.warning("Ignored queue_move command with invalid payload: %s", payload)
        elif command == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self.state.is_playing_move = False
            self._breathing_active = False
            logger.info("Cleared move queue and stopped current move")
        elif command == "set_moving_state":
            try:
                duration = float(payload)
            except (TypeError, ValueError):
                logger.warning("Invalid moving state duration: %s", payload)
                return
            self.state.moving_start = current_time
            self.state.moving_for = max(0.0, duration)
            self.state.update_activity()
        elif command == "mark_activity":
            self.state.update_activity()
        elif command == "set_listening":
            desired_state = bool(payload)
            if self._is_listening == desired_state:
                return
            self._is_listening = desired_state
            self._last_listening_blend_time = self._now()
            if desired_state:
                # Capture the last antenna command so we keep that pose during listening
                self._listening_antennas = (
                    float(self._last_commanded_pose[1][0]),
                    float(self._last_commanded_pose[1][1]),
                )
                self._antenna_unfreeze_blend = 0.0
            self.state.update_activity()
        else:
            logger.warning("Unknown command received by MovementManager: %s", command)

    def _publish_shared_state(self) -> None:
        """Expose idle-related state for external threads."""
        with self._shared_state_lock:
            self._shared_last_activity_time = self.state.last_activity_time
            self._shared_is_listening = self._is_listening

    def _manage_move_queue(self, current_time: float) -> None:
        """Manage the primary move queue (sequential execution)."""
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
                # Any real move cancels breathing mode flag
                self._breathing_active = isinstance(
                    self.state.current_move, BreathingMove
                )
                logger.info(
                    f"Starting new move, duration: {self.state.current_move.duration}s"
                )

    def _manage_breathing(self, current_time: float) -> None:
        """Manage automatic breathing when idle."""
        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._breathing_active
        ):
            idle_for = current_time - self.state.last_activity_time
            if idle_for >= self.idle_inactivity_delay:
                try:
                    _, current_antennas = self.current_robot.get_current_joint_positions()
                    current_head_pose = self.current_robot.get_current_head_pose()

                    self._breathing_active = True
                    self.state.update_activity()

                    breathing_move = BreathingMove(
                        interpolation_start_pose=current_head_pose,
                        interpolation_start_antennas=current_antennas,
                        interpolation_duration=1.0,
                    )
                    self.move_queue.append(breathing_move)
                    logger.info("Started breathing after %.1fs of inactivity", idle_for)
                except Exception as e:
                    self._breathing_active = False
                    logger.error("Failed to start breathing: %s", e)

        if (
            isinstance(self.state.current_move, BreathingMove)
            and self.move_queue
        ):
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.info("Stopping breathing due to new move activity")

        if self.state.current_move is not None and not isinstance(
            self.state.current_move, BreathingMove
        ):
            self._breathing_active = False

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get the primary full body pose from current move or neutral."""
        # When a primary move is playing, sample it and cache the resulting pose
        if self.state.current_move is not None and self.state.move_start_time is not None:
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
                current_time - self.state.moving_start < self.state.moving_for
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
            self.state.face_tracking_offsets = offsets
        else:
            # No camera worker, use neutral offsets
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
        logger.debug("Starting enhanced movement control loop (100Hz)")

        loop_count = 0
        prev_loop_start = self._now()
        freq_mean = 0.0
        freq_m2 = 0.0
        freq_min = float("inf")
        freq_count = 0
        last_freq = 0.0
        print_interval_loops = max(1, int(self.target_frequency * 2))
        potential_freq = 0.0

        while not self._stop_event.is_set():
            loop_start = self._now()
            loop_count += 1
            current_time = loop_start

            if loop_count > 1:
                period = loop_start - prev_loop_start
                if period > 0:
                    last_freq = 1.0 / period
                    freq_count += 1
                    delta = last_freq - freq_mean
                    freq_mean += delta / freq_count
                    freq_m2 += delta * (last_freq - freq_mean)
                    freq_min = min(freq_min, last_freq)
            prev_loop_start = loop_start

            # 1. Poll external signals and commands
            self._poll_signals(current_time)

            # 2. Sequential move management
            self._manage_move_queue(current_time)

            # 3. Automatic behaviours (breathing)
            self._manage_breathing(current_time)

            # 4. Update vision offsets
            self._update_face_tracking(current_time)

            # 5. Build pose snapshots
            primary_full_body_pose = self._get_primary_pose(current_time)
            secondary_full_body_pose = self._get_secondary_pose()
            global_full_body_pose = combine_full_body(
                primary_full_body_pose, secondary_full_body_pose
            )

            # 6. Blend listening state for antennas
            head, antennas, body_yaw = global_full_body_pose
            now_monotonic = self._now()
            listening = self._is_listening
            listening_antennas = self._listening_antennas
            blend = self._antenna_unfreeze_blend
            blend_duration = self._antenna_blend_duration
            last_update = self._last_listening_blend_time
            self._last_listening_blend_time = now_monotonic

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
                    listening_antennas[0] * (1.0 - new_blend) + antennas[0] * new_blend,
                    listening_antennas[1] * (1.0 - new_blend) + antennas[1] * new_blend,
                )

            if listening:
                self._antenna_unfreeze_blend = 0.0
            else:
                self._antenna_unfreeze_blend = new_blend
                if new_blend >= 1.0:
                    self._listening_antennas = (
                        float(antennas[0]),
                        float(antennas[1]),
                    )

            # 7. Single set_target call - the only control point
            try:
                self.current_robot.set_target(
                    head=head, antennas=antennas_cmd, body_yaw=body_yaw
                )
            except Exception as e:
                logger.error(f"Failed to set robot target: {e}")
            else:
                self._last_commanded_pose = clone_full_body_pose(
                    (head, antennas_cmd, body_yaw)
                )

            # 8. Timing bookkeeping + adaptive sleep for 100Hz
            computation_time = self._now() - loop_start
            potential_freq = (
                1.0 / computation_time if computation_time > 0 else float("inf")
            )
            sleep_time = max(0.0, self.target_period - computation_time)

            self._publish_shared_state()

            if sleep_time > 0:
                time.sleep(sleep_time)

            # 9. Periodic telemetry
            if loop_count % print_interval_loops == 0 and freq_count > 0:
                variance = freq_m2 / freq_count if freq_count > 0 else 0.0
                lowest = freq_min if freq_min != float("inf") else 0.0
                logger.debug(
                    "Loop freq - avg: %.2fHz, variance: %.4f, min: %.2fHz, last: %.2fHz, potential: %.2fHz, target: %.1fHz",
                    freq_mean,
                    variance,
                    lowest,
                    last_freq,
                    potential_freq,
                    self.target_frequency,
                )
                freq_mean = 0.0
                freq_m2 = 0.0
                freq_min = float("inf")
                freq_count = 0

        logger.debug("Movement control loop stopped")
