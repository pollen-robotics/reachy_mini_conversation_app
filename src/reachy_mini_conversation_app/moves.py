"""Movement system with sequential primary moves and additive secondary moves.

Design overview
- Primary moves (emotions, dances, goto, breathing) are mutually exclusive and run
  sequentially.
- Secondary moves (speech sway, face tracking) are additive offsets applied on top
  of the current primary pose.
- There is a single control point to the robot: `ReachyMini.set_target`.
- The control loop runs near 100 Hz and is phase-aligned via a monotonic clock.
- Idle behaviour starts an infinite `BreathingMove` after a short inactivity delay
  unless listening is active.

Threading model
- A dedicated worker thread owns all real-time state and issues `set_target`
  commands.
- Other threads communicate via a command queue (enqueue moves, mark activity,
  toggle listening).
- Secondary offset producers set pending values guarded by locks; the worker
  snaps them atomically.

Units and frames
- Secondary offsets are interpreted as metres for x/y/z and radians for
  roll/pitch/yaw in the world frame (unless noted by `compose_world_offset`).
- Antennas and `body_yaw` are in radians.
- Head pose composition uses `compose_world_offset(primary_head, secondary_head)`;
  the secondary offset must therefore be expressed in the world frame.

Safety
- Listening freezes antennas, then blends them back on unfreeze.
- Interpolations and blends are used to avoid jumps at all times.
- `set_target` errors are rate-limited in logs.
"""

from __future__ import annotations
import time
import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Tuple
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import (
    compose_world_offset,
    linear_pose_interpolation,
)


logger = logging.getLogger(__name__)

# Configuration constants
CONTROL_LOOP_FREQUENCY_HZ = 100.0  # Hz - Target frequency for the movement control loop

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]  # (head_pose_4x4, antennas, body_yaw)


class BreathingMove(Move):  # type: ignore
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
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

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate breathing move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to neutral base position
            interpolation_t = t / self.interpolation_duration

            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose, self.neutral_head_pose, interpolation_t,
            )

            # Interpolate antennas
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)

        else:
            # Phase 2: Breathing patterns from neutral base
            breathing_time = t - self.interpolation_duration

            # Gentle z-axis breathing
            z_offset = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)
            head_pose = create_head_pose(x=0, y=0, z=z_offset, roll=0, pitch=0, yaw=0, degrees=True, mm=False)

            # Antenna sway (opposite directions)
            antenna_sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        return (head_pose, antennas, 0.0)


def combine_full_body(primary_pose: FullBodyPose, secondary_pose: FullBodyPose) -> FullBodyPose:
    """Combine primary and secondary full body poses.

    Args:
        primary_pose: (head_pose, antennas, body_yaw) - primary move
        secondary_pose: (head_pose, antennas, body_yaw) - secondary offsets

    Returns:
        Combined full body pose (head_pose, antennas, body_yaw)

    """
    primary_head, primary_antennas, primary_body_yaw = primary_pose
    secondary_head, secondary_antennas, secondary_body_yaw = secondary_pose

    # Combine head poses using compose_world_offset; the secondary pose must be an
    # offset expressed in the world frame (T_off_world) applied to the absolute
    # primary transform (T_abs).
    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=True)

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
    current_move: Move | None = None
    move_start_time: float | None = None
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

    # Status flags
    last_primary_pose: FullBodyPose | None = None

    def update_activity(self) -> None:
        """Update the last activity time."""
        self.last_activity_time = time.monotonic()


@dataclass
class LoopFrequencyStats:
    """Track rolling loop frequency statistics."""

    mean: float = 0.0
    m2: float = 0.0
    min_freq: float = float("inf")
    count: int = 0
    last_freq: float = 0.0
    potential_freq: float = 0.0

    def reset(self) -> None:
        """Reset accumulators while keeping the last potential frequency."""
        self.mean = 0.0
        self.m2 = 0.0
        self.min_freq = float("inf")
        self.count = 0


class MovementManager:
    """Coordinate sequential moves, additive offsets, and robot output at 100 Hz.

    Responsibilities:
    - Own a real-time loop that samples the current primary move (if any), fuses
      secondary offsets, and calls `set_target` exactly once per tick.
    - Start an idle `BreathingMove` after `idle_inactivity_delay` when not
      listening and no moves are queued.
    - Expose thread-safe APIs so other threads can enqueue moves, mark activity,
      or feed secondary offsets without touching internal state.

    Timing:
    - All elapsed-time calculations rely on `time.monotonic()` through `self._now`
      to avoid wall-clock jumps.
    - The loop attempts 100 Hz

    Concurrency:
    - External threads communicate via `_command_queue` messages.
    - Secondary offsets are staged via dirty flags guarded by locks and consumed
      atomically inside the worker loop.
    """

    def __init__(
        self,
        current_robot: ReachyMini,
        camera_worker: "Any" = None,
    ):
        """Initialize movement manager."""
        self.current_robot = current_robot
        self.camera_worker = camera_worker

        # Single timing source for durations
        self._now = time.monotonic

        # Movement state
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)

        # Move queue (primary moves)
        self.move_queue: deque[Move] = deque()

        # Configuration
        self.idle_inactivity_delay = 0.3  # seconds
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_listening = False
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(self.state.last_primary_pose)
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4  # seconds to blend back after listening
        self._last_listening_blend_time = self._now()
        self._breathing_active = False  # true when breathing move is running or queued
        self._listening_debounce_s = 0.15
        self._last_listening_toggle_time = self._now()
        self._last_set_target_err = 0.0
        self._set_target_err_interval = 1.0  # seconds between error logs
        self._set_target_err_suppressed = 0

        # Cross-thread signalling
        self._command_queue: "Queue[Tuple[str, Any]]" = Queue()
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
        self._status_lock = threading.Lock()
        self._freq_stats = LoopFrequencyStats()
        self._freq_snapshot = LoopFrequencyStats()

    def queue_move(self, move: Move) -> None:
        """Queue a primary move to run after the currently executing one.

        Thread-safe: the move is enqueued via the worker command queue so the
        control loop remains the sole mutator of movement state.
        """
        self._command_queue.put(("queue_move", move))

    def clear_move_queue(self) -> None:
        """Stop the active move and discard any queued primary moves.

        Thread-safe: executed by the worker thread via the command queue.
        """
        self._command_queue.put(("clear_queue", None))

    def set_speech_offsets(self, offsets: Tuple[float, float, float, float, float, float]) -> None:
        """Update speech-induced secondary offsets (x, y, z, roll, pitch, yaw).

        Offsets are interpreted as metres for translation and radians for
        rotation in the world frame. Thread-safe via a pending snapshot.
        """
        with self._speech_offsets_lock:
            self._pending_speech_offsets = offsets
            self._speech_offsets_dirty = True

    def set_moving_state(self, duration: float) -> None:
        """Mark the robot as actively moving for the provided duration.

        Legacy hook used by goto helpers to keep inactivity and breathing logic
        aware of manual motions. Thread-safe via the command queue.
        """
        self._command_queue.put(("set_moving_state", duration))

    def is_idle(self) -> bool:
        """Return True when the robot has been inactive longer than the idle delay."""
        with self._shared_state_lock:
            last_activity = self._shared_last_activity_time
            listening = self._shared_is_listening

        if listening:
            return False

        return self._now() - last_activity >= self.idle_inactivity_delay

    def set_listening(self, listening: bool) -> None:
        """Enable or disable listening mode without touching shared state directly.

        While listening:
        - Antenna positions are frozen at the last commanded values.
        - Blending is reset so that upon unfreezing the antennas return smoothly.
        - Idle breathing is suppressed.

        Thread-safe: the change is posted to the worker command queue.
        """
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
        speech_offsets: Tuple[float, float, float, float, float, float] | None = None
        with self._speech_offsets_lock:
            if self._speech_offsets_dirty:
                speech_offsets = self._pending_speech_offsets
                self._speech_offsets_dirty = False

        if speech_offsets is not None:
            self.state.speech_offsets = speech_offsets
            self.state.update_activity()

        face_offsets: Tuple[float, float, float, float, float, float] | None = None
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
                logger.debug(
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
            self._breathing_active = False
            logger.info("Cleared move queue and stopped current move")
        elif command == "set_moving_state":
            try:
                duration = float(payload)
            except (TypeError, ValueError):
                logger.warning("Invalid moving state duration: %s", payload)
                return
            self.state.update_activity()
        elif command == "mark_activity":
            self.state.update_activity()
        elif command == "set_listening":
            desired_state = bool(payload)
            now = self._now()
            if now - self._last_listening_toggle_time < self._listening_debounce_s:
                return
            self._last_listening_toggle_time = now

            if self._is_listening == desired_state:
                return

            self._is_listening = desired_state
            self._last_listening_blend_time = now
            if desired_state:
                # Freeze: snapshot current commanded antennas and reset blend
                self._listening_antennas = (
                    float(self._last_commanded_pose[1][0]),
                    float(self._last_commanded_pose[1][1]),
                )
                self._antenna_unfreeze_blend = 0.0
            else:
                # Unfreeze: restart blending from frozen pose
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
            and current_time - self.state.move_start_time >= self.state.current_move.duration
        ):
            self.state.current_move = None
            self.state.move_start_time = None

            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                # Any real move cancels breathing mode flag
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)
                logger.debug(f"Starting new move, duration: {self.state.current_move.duration}s")

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
                    # These 2 functions return the latest available sensor data from the robot, but don't perform I/O synchronously.
                    # Therefore, we accept calling them inside the control loop.
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
                    logger.debug("Started breathing after %.1fs of inactivity", idle_for)
                except Exception as e:
                    self._breathing_active = False
                    logger.error("Failed to start breathing: %s", e)

        if isinstance(self.state.current_move, BreathingMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.debug("Stopping breathing due to new move activity")

        if self.state.current_move is not None and not isinstance(self.state.current_move, BreathingMove):
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

            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)
        # Otherwise reuse the last primary pose so we avoid jumps between moves
        elif self.state.last_primary_pose is not None:
            primary_full_body_pose = clone_full_body_pose(self.state.last_primary_pose)
        else:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            primary_full_body_pose = (neutral_head_pose, (0.0, 0.0), 0.0)
            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)

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
        return (secondary_head_pose, (0.0, 0.0), 0.0)

    def _compose_full_body_pose(self, current_time: float) -> FullBodyPose:
        """Compose primary and secondary poses into a single command pose."""
        primary = self._get_primary_pose(current_time)
        secondary = self._get_secondary_pose()
        return combine_full_body(primary, secondary)

    def _update_primary_motion(self, current_time: float) -> None:
        """Advance queue state and idle behaviours for this tick."""
        self._manage_move_queue(current_time)
        self._manage_breathing(current_time)

    def _calculate_blended_antennas(self, target_antennas: Tuple[float, float]) -> Tuple[float, float]:
        """Blend target antennas with listening freeze state and update blending."""
        now = self._now()
        listening = self._is_listening
        listening_antennas = self._listening_antennas
        blend = self._antenna_unfreeze_blend
        blend_duration = self._antenna_blend_duration
        last_update = self._last_listening_blend_time
        self._last_listening_blend_time = now

        if listening:
            antennas_cmd = listening_antennas
            new_blend = 0.0
        else:
            dt = max(0.0, now - last_update)
            if blend_duration <= 0:
                new_blend = 1.0
            else:
                new_blend = min(1.0, blend + dt / blend_duration)
            antennas_cmd = (
                listening_antennas[0] * (1.0 - new_blend) + target_antennas[0] * new_blend,
                listening_antennas[1] * (1.0 - new_blend) + target_antennas[1] * new_blend,
            )

        if listening:
            self._antenna_unfreeze_blend = 0.0
        else:
            self._antenna_unfreeze_blend = new_blend
            if new_blend >= 1.0:
                self._listening_antennas = (
                    float(target_antennas[0]),
                    float(target_antennas[1]),
                )

        return antennas_cmd

    def _issue_control_command(self, head: NDArray[np.float32], antennas: Tuple[float, float], body_yaw: float) -> None:
        """Send the fused pose to the robot with throttled error logging."""
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
        except Exception as e:
            now = self._now()
            if now - self._last_set_target_err >= self._set_target_err_interval:
                msg = f"Failed to set robot target: {e}"
                if self._set_target_err_suppressed:
                    msg += f" (suppressed {self._set_target_err_suppressed} repeats)"
                    self._set_target_err_suppressed = 0
                logger.error(msg)
                self._last_set_target_err = now
            else:
                self._set_target_err_suppressed += 1
        else:
            with self._status_lock:
                self._last_commanded_pose = clone_full_body_pose((head, antennas, body_yaw))

    def _update_frequency_stats(
        self, loop_start: float, prev_loop_start: float, stats: LoopFrequencyStats,
    ) -> LoopFrequencyStats:
        """Update frequency statistics based on the current loop start time."""
        period = loop_start - prev_loop_start
        if period > 0:
            stats.last_freq = 1.0 / period
            stats.count += 1
            delta = stats.last_freq - stats.mean
            stats.mean += delta / stats.count
            stats.m2 += delta * (stats.last_freq - stats.mean)
            stats.min_freq = min(stats.min_freq, stats.last_freq)
        return stats

    def _schedule_next_tick(self, loop_start: float, stats: LoopFrequencyStats) -> Tuple[float, LoopFrequencyStats]:
        """Compute sleep time to maintain target frequency and update potential freq."""
        computation_time = self._now() - loop_start
        stats.potential_freq = 1.0 / computation_time if computation_time > 0 else float("inf")
        sleep_time = max(0.0, self.target_period - computation_time)
        return sleep_time, stats

    def _record_frequency_snapshot(self, stats: LoopFrequencyStats) -> None:
        """Store a thread-safe snapshot of current frequency statistics."""
        with self._status_lock:
            self._freq_snapshot = LoopFrequencyStats(
                mean=stats.mean,
                m2=stats.m2,
                min_freq=stats.min_freq,
                count=stats.count,
                last_freq=stats.last_freq,
                potential_freq=stats.potential_freq,
            )

    def _maybe_log_frequency(self, loop_count: int, print_interval_loops: int, stats: LoopFrequencyStats) -> None:
        """Emit frequency telemetry when enough loops have elapsed."""
        if loop_count % print_interval_loops != 0 or stats.count == 0:
            return

        variance = stats.m2 / stats.count if stats.count > 0 else 0.0
        lowest = stats.min_freq if stats.min_freq != float("inf") else 0.0
        logger.debug(
            "Loop freq - avg: %.2fHz, variance: %.4f, min: %.2fHz, last: %.2fHz, potential: %.2fHz, target: %.1fHz",
            stats.mean,
            variance,
            lowest,
            stats.last_freq,
            stats.potential_freq,
            self.target_frequency,
        )
        stats.reset()

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
        """Start the worker thread that drives the 100 Hz control loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Move worker already running; start() ignored")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Move worker started")

    def stop(self) -> None:
        """Request the worker thread to stop and wait for it to exit.

        Before stopping, resets the robot to a neutral position.
        """
        if self._thread is None or not self._thread.is_alive():
            logger.debug("Move worker not running; stop() ignored")
            return

        logger.info("Stopping movement manager and resetting to neutral position...")

        # Clear any queued moves and stop current move
        self.clear_move_queue()

        # Stop the worker thread first so it doesn't interfere
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        logger.debug("Move worker stopped")

        # Reset to neutral position using goto_target (same approach as wake_up)
        try:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            neutral_antennas = [0.0, 0.0]
            neutral_body_yaw = 0.0

            # Use goto_target directly on the robot
            self.current_robot.goto_target(
                head=neutral_head_pose,
                antennas=neutral_antennas,
                duration=2.0,
                body_yaw=neutral_body_yaw,
            )

            logger.info("Reset to neutral position completed")

        except Exception as e:
            logger.error(f"Failed to reset to neutral position: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return a lightweight status snapshot for observability."""
        with self._status_lock:
            pose_snapshot = clone_full_body_pose(self._last_commanded_pose)
            freq_snapshot = LoopFrequencyStats(
                mean=self._freq_snapshot.mean,
                m2=self._freq_snapshot.m2,
                min_freq=self._freq_snapshot.min_freq,
                count=self._freq_snapshot.count,
                last_freq=self._freq_snapshot.last_freq,
                potential_freq=self._freq_snapshot.potential_freq,
            )

        head_matrix = pose_snapshot[0].tolist() if pose_snapshot else None
        antennas = pose_snapshot[1] if pose_snapshot else None
        body_yaw = pose_snapshot[2] if pose_snapshot else None

        return {
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "breathing_active": self._breathing_active,
            "last_commanded_pose": {
                "head": head_matrix,
                "antennas": antennas,
                "body_yaw": body_yaw,
            },
            "loop_frequency": {
                "last": freq_snapshot.last_freq,
                "mean": freq_snapshot.mean,
                "min": freq_snapshot.min_freq,
                "potential": freq_snapshot.potential_freq,
                "samples": freq_snapshot.count,
            },
        }

    def working_loop(self) -> None:
        """Control loop main movements - reproduces main_works.py control architecture.

        Single set_target() call with pose fusion.
        """
        logger.debug("Starting enhanced movement control loop (100Hz)")

        loop_count = 0
        prev_loop_start = self._now()
        print_interval_loops = max(1, int(self.target_frequency * 2))
        freq_stats = self._freq_stats

        while not self._stop_event.is_set():
            loop_start = self._now()
            loop_count += 1

            if loop_count > 1:
                freq_stats = self._update_frequency_stats(loop_start, prev_loop_start, freq_stats)
            prev_loop_start = loop_start

            # 1) Poll external commands and apply pending offsets (atomic snapshot)
            self._poll_signals(loop_start)

            # 2) Manage the primary move queue (start new move, end finished move, breathing)
            self._update_primary_motion(loop_start)

            # 3) Update vision-based secondary offsets
            self._update_face_tracking(loop_start)

            # 4) Build primary and secondary full-body poses, then fuse them
            head, antennas, body_yaw = self._compose_full_body_pose(loop_start)

            # 5) Apply listening antenna freeze or blend-back
            antennas_cmd = self._calculate_blended_antennas(antennas)

            # 6) Single set_target call - the only control point
            self._issue_control_command(head, antennas_cmd, body_yaw)

            # 7) Adaptive sleep to align to next tick, then publish shared state
            sleep_time, freq_stats = self._schedule_next_tick(loop_start, freq_stats)
            self._publish_shared_state()
            self._record_frequency_snapshot(freq_stats)

            # 8) Periodic telemetry on loop frequency
            self._maybe_log_frequency(loop_count, print_interval_loops, freq_stats)

            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Movement control loop stopped")
