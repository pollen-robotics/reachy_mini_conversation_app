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
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import (
    compose_world_offset,
    linear_pose_interpolation,
    time_trajectory,
    InterpolationTechnique,
)


logger = logging.getLogger(__name__)

# Configuration constants
CONTROL_LOOP_FREQUENCY_HZ = 100.0  # Hz - Target frequency for the movement control loop

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]  # (head_pose_4x4, antennas, body_yaw)


class AnchorState(Enum):
    """State machine for body yaw anchoring system."""
    ANCHORED = "anchored"        # Body locked at anchor point
    SYNCING = "syncing"          # Body moving to match head
    STABILIZING = "stabilizing"  # Waiting for head to stabilize


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

        # Breathing parameters - subtle meditative sway motion
        self.sway_amplitude = 0.006  # 6mm side-to-side drift (garnish, not side dish)
        self.roll_amplitude = np.deg2rad(-5)  # -5 degrees tilt for visible breathing
        self.breathing_frequency = 0.2  # Hz (5 second cycle)
        self.antenna_sway_amplitude = np.deg2rad(12)  # 12 degrees antenna sway
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
            # Phase 2: Gentle breathing - ONLY roll, no translation
            # Sway causes camera motion which makes stationary faces appear to move,
            # triggering false corrections and creating feedback loops
            breathing_time = t - self.interpolation_duration

            # Roll tilt for breathing motion
            roll_tilt = self.roll_amplitude * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)

            # Create breathing pose: ONLY roll, no translation or yaw/pitch
            # Yaw=0 and pitch=0 prevent breathing from resetting face tracking orientation
            # Face tracking controls yaw/pitch, breathing adds roll tilt
            head_pose = create_head_pose(
                x=0.0,
                y=0.0,          # NO sway - causes camera motion artifacts
                z=0.01,         # Z translation (lift slightly)
                roll=roll_tilt, # Roll tilt for breathing motion
                pitch=0.0,      # NO pitch - let face tracking control this
                yaw=0.0,        # NO yaw - let face tracking control this
                degrees=False,
                mm=False
            )

            # Antenna sway (opposite directions) - keep existing implementation
            antenna_sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        # Return None for body_yaw to preserve current position (avoid locking to center)
        return (head_pose, antennas, None)


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

        # Debug logging rate limiting
        self._last_breathing_debug = 0.0
        self._breathing_debug_interval = 2.0  # seconds between debug messages

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_listening = False
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(self.state.last_primary_pose)
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4  # seconds to blend back after listening
        self._last_listening_blend_time = self._now()
        self._breathing_active = False  # true when breathing move is running or queued
        self._breathing_paused_external = False  # true when external coordinator pauses breathing
        self._breathing_pause_lock = threading.Lock()
        self._listening_debounce_s = 0.15
        self._last_listening_toggle_time = self._now()
        self._last_set_target_err = 0.0
        self._set_target_err_interval = 1.0  # seconds between error logs
        self._set_target_err_suppressed = 0

        # Face tracking suppression for primary moves
        self._suppress_face_tracking = False

        # External control flag (for Claude Code mood plugin coordination)
        self._external_control_active = False
        self._external_control_lock = threading.Lock()

        # Body follow state tracking (legacy - now using anchor system)
        self._last_head_yaw_deg = 0.0
        self._head_stable_since = None  # Time when head became stable
        self._body_follow_threshold_deg = 20.0
        self._head_stability_threshold_deg = 5.0  # Max change to be considered stable
        self._body_follow_duration = 1.0  # Duration for smooth body follow interpolation (1 second)
        self._body_follow_deadband_deg = 2.0  # Ignore adjustments smaller than this
        self._body_follow_start_yaw = 0.0  # Starting body yaw for interpolation
        self._body_follow_target_yaw = 0.0  # Target body yaw for interpolation
        self._body_follow_start_time = None  # When current follow motion started

        # Anchor-based body yaw control
        self._anchor_state = AnchorState.ANCHORED
        self._body_anchor_yaw = 0.0              # Current anchor point (temp_zero)
        self._strain_threshold_deg = 13.0        # 20% of 65° max head-body difference
        self._stability_duration_s = 3.0         # 3 seconds for head stabilization before anchor lock
        self._stability_threshold_deg = 2.0      # 2 degrees max movement to be considered stable

        # Pitch rate limiting (applied in moves.py after clamp, not in camera_worker)
        self._last_commanded_pitch = 0.0         # Previous commanded pitch for rate limiting
        self._max_pitch_change_per_frame = np.deg2rad(5.0)  # Max 5°/frame (~150°/sec at 30Hz)

        # Oscillation detection and recovery
        self._pitch_direction_changes = 0       # Count of direction reversals
        self._last_pitch_change_sign = 0        # Sign of last change (+1, -1, or 0)
        self._oscillation_recovery_mode = False # In recovery mode flag
        self._oscillation_recovery_start = None # When recovery started
        self._oscillation_recovery_duration = 5.0  # Hold at 0° for 5 seconds
        self._oscillation_threshold = 2         # Max direction changes before recovery

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

    def set_external_control(self, active: bool) -> None:
        """Set external control flag for coordination with Claude Code mood plugin.

        When active, face tracking is suppressed to allow external control.
        Thread-safe via lock.
        """
        with self._external_control_lock:
            self._external_control_active = active
            logger.info(f"External control {'enabled' if active else 'disabled'} - face tracking {'suppressed' if active else 'resumed'}")

        # If starting external control, stop any active breathing move
        if active and isinstance(self.state.current_move, BreathingMove):
            self._command_queue.put(("clear_queue", None))
            logger.info("Stopped breathing move for external control")

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

    def pause_breathing(self) -> None:
        """Pause breathing animation for external move coordination.

        Called by daemon before executing moves from desktop viewer or other sources.
        Prevents breathing from starting, clears any active breathing, and stops
        the control loop from calling set_target() to avoid race conditions.
        """
        with self._breathing_pause_lock:
            self._breathing_paused_external = True

        # Relinquish control - stop control loop from calling set_target()
        with self._external_control_lock:
            self._external_control_active = True

        # Clear any active breathing
        if self._breathing_active:
            self._command_queue.put(("clear_queue", None))
        logger.info("Breathing paused and control relinquished for external move")

    def resume_breathing(self) -> None:
        """Resume breathing animation after external move completes.

        Called by daemon after move execution finishes.
        Allows breathing to start again when idle and resumes control loop commands.
        """
        with self._breathing_pause_lock:
            self._breathing_paused_external = False

        # Reclaim control - resume control loop set_target() calls
        with self._external_control_lock:
            self._external_control_active = False

        logger.info("Breathing resumed and control reclaimed")

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
            logger.info(f"[DEBUG] Speech offsets updated, activity reset")

        face_offsets: Tuple[float, float, float, float, float, float] | None = None
        with self._face_offsets_lock:
            if self._face_offsets_dirty:
                face_offsets = self._pending_face_offsets
                self._face_offsets_dirty = False

        if face_offsets is not None:
            self.state.face_tracking_offsets = face_offsets
            # Face tracking is secondary motion - don't reset activity timer
            logger.info(f"[DEBUG] Face offsets updated (no activity reset)")

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
            self._suppress_face_tracking = False  # Resume face tracking when queue cleared
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
            # Clear face tracking suppression when move completes
            self._suppress_face_tracking = False

            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                # Any real move cancels breathing mode flag
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)
                # Suppress face tracking for primary moves (except BreathingMove)
                self._suppress_face_tracking = not isinstance(self.state.current_move, BreathingMove)
                logger.debug(f"Starting new move, duration: {self.state.current_move.duration}s, face tracking suppressed: {self._suppress_face_tracking}")

    def _manage_breathing(self, current_time: float) -> None:
        """Manage automatic breathing when idle."""
        # Check external control flag
        with self._external_control_lock:
            external_control = self._external_control_active

        # Check external breathing pause flag
        with self._breathing_pause_lock:
            breathing_paused = self._breathing_paused_external

        # Calculate idle time for both debugging and logic
        idle_for = current_time - self.state.last_activity_time

        # Periodic debug (every 2 seconds) - show breathing state
        if current_time - self._last_breathing_debug >= self._breathing_debug_interval:
            self._last_breathing_debug = current_time

            # Only log when breathing is NOT active (avoid spam when already breathing)
            if not self._breathing_active:
                conditions = {
                    "no_move": self.state.current_move is None,
                    "queue_empty": not self.move_queue,
                    "not_listening": not self._is_listening,
                    "no_external": not external_control,
                    "not_paused": not breathing_paused,
                }

                blocking = [k for k, v in conditions.items() if not v]
                idle_ready = idle_for >= self.idle_inactivity_delay

                if blocking or not idle_ready:
                    status = "idle_time" if not idle_ready and not blocking else blocking
                    logger.info(f"Breathing check: idle={idle_for:.2f}s/{self.idle_inactivity_delay}s, blocking={status}")

        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._breathing_active
            and not external_control
            and not breathing_paused
        ):
            if idle_for >= self.idle_inactivity_delay:
                # Set breathing active IMMEDIATELY to prevent repeated attempts
                self._breathing_active = True
                try:
                    # These 2 functions return the latest available sensor data from the robot, but don't perform I/O synchronously.
                    # Therefore, we accept calling them inside the control loop.
                    _, current_joints = self.current_robot.get_current_joint_positions()
                    current_head_pose = self.current_robot.get_current_head_pose()

                    # Joint array should be [body_yaw, left_antenna, right_antenna] but may only have antennas
                    if len(current_joints) >= 2:
                        # Only start breathing if robot is fully initialized (need at least antennas)
                        if len(current_joints) == 3:
                            current_antennas = (current_joints[1], current_joints[2])
                        else:
                            # Only antenna values returned
                            current_antennas = (current_joints[0], current_joints[1])

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

        if isinstance(self.state.current_move, BreathingMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            # Note: face tracking suppression will be set by the new move in _manage_move_queue
            logger.debug("Stopping breathing due to new move activity")

        if self.state.current_move is not None and not isinstance(self.state.current_move, BreathingMove):
            self._breathing_active = False

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get the primary full body pose from current move or neutral."""
        # When a primary move is playing, sample it and cache the resulting pose
        if self.state.current_move is not None and self.state.move_start_time is not None:
            # Skip breathing move evaluation when external control is active
            with self._external_control_lock:
                external_active = self._external_control_active

            # Pause breathing during external control or body syncing
            suppress_breathing = (
                (isinstance(self.state.current_move, BreathingMove) and external_active) or
                (isinstance(self.state.current_move, BreathingMove) and self._anchor_state == AnchorState.SYNCING)
            )

            if suppress_breathing:
                # Return neutral pose with z=0.01 lift to maintain "alive" appearance
                head = create_head_pose(0, 0, 0.01, 0, 0, 0, degrees=False, mm=False)
                antennas = np.array([0.0, 0.0])
                body_yaw = 0.0
            else:
                move_time = current_time - self.state.move_start_time
                head, antennas, body_yaw = self.state.current_move.evaluate(move_time)

            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([0.0, 0.0])
            if body_yaw is None:
                # Preserve last commanded body_yaw (includes body_follow calculation)
                # Breathing doesn't care about body orientation - let body_follow manage it
                body_yaw = self._last_commanded_pose[2]

            antennas_tuple = (float(antennas[0]), float(antennas[1]))
            head_copy = head.copy()

            # Make breathing relative to current anchor position
            # Apply anchor yaw offset so breathing happens around the anchor, not zero
            if isinstance(self.state.current_move, BreathingMove):
                from scipy.spatial.transform import Rotation as R

                # Extract current yaw from breathing pose (should be 0)
                head_rotation = R.from_matrix(head_copy[:3, :3])
                head_roll, head_pitch, head_yaw_local = head_rotation.as_euler("xyz")

                # Add anchor offset to make breathing relative to anchor position
                anchor_yaw_rad = np.deg2rad(self._body_anchor_yaw)
                absolute_yaw = head_yaw_local + anchor_yaw_rad

                # Reconstruct rotation matrix with anchor-relative yaw
                new_rotation = R.from_euler("xyz", [head_roll, head_pitch, absolute_yaw])
                head_copy[:3, :3] = new_rotation.as_matrix()

            # Rotate breathing Y sway by current HEAD orientation (not body)
            # This makes breathing sway relative to where head is pointing
            if isinstance(self.state.current_move, BreathingMove):
                # Extract head yaw from rotation matrix
                from scipy.spatial.transform import Rotation as R
                head_rotation = R.from_matrix(head_copy[:3, :3])
                head_roll, head_pitch, head_yaw = head_rotation.as_euler("xyz")

                # Neutral pose with 10mm elevation
                neutral_pose = np.eye(4, dtype=np.float32)
                neutral_pose[2, 3] = 0.01

                # Calculate breathing sway delta from neutral
                dx = head_copy[0, 3] - neutral_pose[0, 3]
                dy = head_copy[1, 3] - neutral_pose[1, 3]
                dz = head_copy[2, 3] - neutral_pose[2, 3]

                # Rotate delta by head yaw
                cos_yaw = np.cos(head_yaw)
                sin_yaw = np.sin(head_yaw)

                dx_rotated = dx * cos_yaw - dy * sin_yaw
                dy_rotated = dx * sin_yaw + dy * cos_yaw

                # Apply rotated delta to neutral position
                head_copy[0, 3] = neutral_pose[0, 3] + dx_rotated
                head_copy[1, 3] = neutral_pose[1, 3] + dy_rotated
                head_copy[2, 3] = neutral_pose[2, 3] + dz

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


    def _extract_yaw_from_pose(self, head_pose: NDArray[np.float32]) -> float:
        """Extract yaw angle in radians from 4x4 transformation matrix."""
        # Extract rotation matrix (top-left 3x3)
        R = head_pose[:3, :3]
        # Calculate yaw from rotation matrix (assuming ZYX euler convention)
        # yaw = atan2(R[1,0], R[0,0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return yaw

    def _apply_body_follow(self, pose: FullBodyPose) -> FullBodyPose:
        """Apply anchor-based body yaw control with strain threshold.

        State machine:
        - ANCHORED: Body locked at anchor point until strain exceeds threshold
        - SYNCING: Body smoothly interpolating to match head position
        - STABILIZING: Waiting for head to stabilize before setting new anchor

        Uses EASE_IN_OUT for normal tracking, CARTOON during external plugin control
        for more expressive movements.

        Args:
            pose: Full body pose (head_pose, antennas, body_yaw)

        Returns:
            Adjusted full body pose with anchor-based body yaw control
        """
        head_pose, antennas, body_yaw = pose

        # Extract current head yaw (absolute)
        head_yaw_rad = self._extract_yaw_from_pose(head_pose)
        head_yaw_deg = np.rad2deg(head_yaw_rad)
        body_yaw_deg = np.rad2deg(body_yaw)

        now = self._now()

        # Calculate strain (difference between head and anchor point)
        strain = head_yaw_deg - self._body_anchor_yaw

        # Normalize strain to [-180, 180]
        while strain > 180:
            strain -= 360
        while strain < -180:
            strain += 360

        # Debug logging every 100 ticks (~1 second at 100Hz)
        if not hasattr(self, '_body_follow_log_counter'):
            self._body_follow_log_counter = 0
        self._body_follow_log_counter += 1
        if self._body_follow_log_counter % 100 == 0:
            logger.info(f"Body follow INPUT: head_yaw={head_yaw_deg:.1f}°, body_yaw_input={body_yaw_deg:.1f}°")
            logger.info(f"Body follow: state={self._anchor_state.value}, anchor={self._body_anchor_yaw:.1f}°, strain={strain:.1f}°")

        # STATE MACHINE
        if self._anchor_state == AnchorState.ANCHORED:
            # Check if strain exceeds threshold
            if abs(strain) > self._strain_threshold_deg:
                # Trigger sync
                logger.debug(f"Anchor: Strain {strain:.1f}° exceeds {self._strain_threshold_deg:.1f}° threshold, syncing body to head")
                self._anchor_state = AnchorState.SYNCING
                self._body_follow_start_time = now
                self._body_follow_start_yaw = self._body_anchor_yaw
                self._body_follow_target_yaw = head_yaw_deg
            else:
                # Stay anchored - lock body at anchor point
                body_yaw_deg = self._body_anchor_yaw

        elif self._anchor_state == AnchorState.SYNCING:
            # Interpolate body toward head
            elapsed = now - self._body_follow_start_time
            t_normalized = min(1.0, elapsed / self._body_follow_duration)

            # Choose interpolation method based on external control
            with self._external_control_lock:
                is_external = self._external_control_active

            if is_external:
                # CARTOON for expressive plugin movements
                interp_method = InterpolationTechnique.CARTOON
            else:
                # EASE_IN_OUT for gentle natural tracking
                interp_method = InterpolationTechnique.EASE_IN_OUT

            # Apply interpolation curve
            t_curved = time_trajectory(t_normalized, interp_method)

            # Interpolate body yaw
            body_yaw_deg = (
                (1 - t_curved) * self._body_follow_start_yaw +
                t_curved * self._body_follow_target_yaw
            )

            # Check if sync complete
            if t_normalized >= 1.0:
                logger.debug("Anchor: Sync complete, entering stabilization phase")
                self._anchor_state = AnchorState.STABILIZING
                self._head_stable_since = None

        elif self._anchor_state == AnchorState.STABILIZING:
            # Body matches head, wait for stability
            body_yaw_deg = head_yaw_deg

            # Track head movement
            head_change = abs(head_yaw_deg - self._last_head_yaw_deg)

            if head_change > self._stability_threshold_deg:
                # Head moved significantly, reset stability timer
                self._head_stable_since = None
            elif self._head_stable_since is None:
                # Head just became stable, start timer
                self._head_stable_since = now
            else:
                # Check if stable long enough
                stable_duration = now - self._head_stable_since
                if stable_duration >= self._stability_duration_s:
                    # Establish new anchor!
                    logger.debug(f"Anchor: Head stable for {stable_duration:.1f}s, setting new anchor at {head_yaw_deg:.1f}°")
                    self._body_anchor_yaw = head_yaw_deg
                    self._anchor_state = AnchorState.ANCHORED

        # Update last head yaw for stability tracking
        self._last_head_yaw_deg = head_yaw_deg

        # Log OUTPUT body yaw (what's actually being sent)
        if self._body_follow_log_counter % 100 == 0:
            logger.info(f"Body follow OUTPUT: body_yaw={body_yaw_deg:.1f}° ({np.deg2rad(body_yaw_deg):.3f} rad)")

        return (head_pose, antennas, np.deg2rad(body_yaw_deg))

    def _compose_full_body_pose(self, current_time: float) -> FullBodyPose:
        """Compose primary and secondary poses into a single command pose.

        Architecture:
        - Primary: Breathing or explicit move
        - Secondary: Face tracking + speech sway (additive offsets)
        - Composition: compose_world_offset(primary, secondary)

        Face tracking and breathing work TOGETHER via composition, not as alternatives.
        """
        # Get the primary pose (breathing, other move, or neutral)
        primary = self._get_primary_pose(current_time)

        # Get the secondary pose (face tracking + speech sway combined)
        secondary = self._get_secondary_pose()

        # Debug logging for composition
        if not hasattr(self, '_compose_log_counter'):
            self._compose_log_counter = 0
        self._compose_log_counter += 1
        if self._compose_log_counter % 100 == 0:
            from scipy.spatial.transform import Rotation as R
            primary_body_yaw_deg = np.rad2deg(primary[2])
            secondary_body_yaw_deg = np.rad2deg(secondary[2])

            # Extract head yaw from secondary pose to see face tracking offset
            secondary_head_rotation = R.from_matrix(secondary[0][:3, :3])
            _, _, secondary_head_yaw = secondary_head_rotation.as_euler("xyz")
            secondary_head_yaw_deg = np.rad2deg(secondary_head_yaw)

            logger.info(f"Composition: primary_body_yaw={primary_body_yaw_deg:.1f}°, secondary_body_yaw={secondary_body_yaw_deg:.1f}°, secondary_head_yaw={secondary_head_yaw_deg:.1f}°")

        # --- Composition Logic ---
        # Face tracking controls yaw/pitch, breathing adds roll and translation
        # Must decompose and recompose to prevent roll from becoming yaw
        primary_head, primary_antennas, primary_body_yaw = primary
        secondary_head, secondary_antennas, secondary_body_yaw = secondary

        from scipy.spatial.transform import Rotation as R_scipy

        # Extract face tracking yaw/pitch (secondary has yaw/pitch, no roll)
        R_face = R_scipy.from_matrix(secondary_head[:3, :3])
        _, face_pitch, face_yaw = R_face.as_euler("xyz", degrees=False)

        # Clamp pitch to mechanical limits (camera_worker calculates raw desired pitch)
        # Positive = up, negative = down
        max_pitch_up = np.deg2rad(15.0)
        max_pitch_down = np.deg2rad(-15.0)
        clamped_pitch = np.clip(face_pitch, max_pitch_down, max_pitch_up)

        # Check if in oscillation recovery mode
        if self._oscillation_recovery_mode:
            # Force pitch to 0° and check if recovery period complete
            current_time = self._now()
            elapsed = current_time - self._oscillation_recovery_start

            if elapsed >= self._oscillation_recovery_duration:
                # Recovery complete - resume normal tracking
                self._oscillation_recovery_mode = False
                self._pitch_direction_changes = 0
                self._last_pitch_change_sign = 0
                logger.info("Oscillation recovery complete, resuming pitch tracking")
                # Continue with normal rate limiting below
            else:
                # Still in recovery - hold at 0°
                rate_limited_pitch = 0.0
                self._last_commanded_pitch = 0.0
                if not hasattr(self, '_recovery_log_counter'):
                    self._recovery_log_counter = 0
                self._recovery_log_counter += 1
                if self._recovery_log_counter % 100 == 0:
                    logger.info(f"Oscillation recovery: holding at 0° ({elapsed:.1f}s / {self._oscillation_recovery_duration:.1f}s)")

        if not self._oscillation_recovery_mode:
            # Normal rate limiting
            pitch_change = clamped_pitch - self._last_commanded_pitch
            rate_limited_pitch = self._last_commanded_pitch + np.clip(
                pitch_change,
                -self._max_pitch_change_per_frame,
                self._max_pitch_change_per_frame
            )

            # Detect oscillation by tracking direction changes
            if abs(pitch_change) > np.deg2rad(1.0):  # Ignore tiny changes
                current_sign = 1 if pitch_change > 0 else -1

                if self._last_pitch_change_sign != 0 and current_sign != self._last_pitch_change_sign:
                    # Direction reversed
                    self._pitch_direction_changes += 1

                    if self._pitch_direction_changes > self._oscillation_threshold:
                        # Too many oscillations - enter recovery mode
                        self._oscillation_recovery_mode = True
                        self._oscillation_recovery_start = self._now()
                        self._pitch_direction_changes = 0
                        logger.warning(f"Pitch oscillation detected! Entering recovery mode (0° for {self._oscillation_recovery_duration}s)")
                        rate_limited_pitch = 0.0

                self._last_pitch_change_sign = current_sign

            self._last_commanded_pitch = rate_limited_pitch

        # Log pitch values every 100 ticks for debugging
        if not hasattr(self, '_pitch_log_counter'):
            self._pitch_log_counter = 0
        self._pitch_log_counter += 1
        if self._pitch_log_counter % 100 == 0:
            logger.info(
                f"Pitch tracking: raw={np.rad2deg(face_pitch):.1f}°, "
                f"clamped={np.rad2deg(clamped_pitch):.1f}°, "
                f"rate_limited={np.rad2deg(rate_limited_pitch):.1f}°"
            )

        # Extract breathing roll (primary has roll, yaw/pitch are zero)
        R_breathing = R_scipy.from_matrix(primary_head[:3, :3])
        breathing_roll, _, _ = R_breathing.as_euler("xyz", degrees=False)

        # Combine: use face tracking's yaw/pitch + breathing's roll
        # This keeps roll as pure roll regardless of yaw
        R_combined = R_scipy.from_euler("xyz", [breathing_roll, rate_limited_pitch, face_yaw], degrees=False)

        # Use face tracking translation (from look_at_image IK calculation)
        # This includes the z-raise needed when looking up/down
        T_face = secondary_head[:3, 3]

        # Create the new combined head pose
        combined_head = np.eye(4, dtype=np.float32)
        combined_head[:3, :3] = R_combined.as_matrix().astype(np.float32)
        combined_head[:3, 3] = T_face

        # Sum antennas and body_yaw as before
        combined_antennas = (
            primary_antennas[0] + secondary_antennas[0],
            primary_antennas[1] + secondary_antennas[1],
        )
        combined_body_yaw = primary_body_yaw + secondary_body_yaw

        combined = (combined_head, combined_antennas, combined_body_yaw)

        # Log combined head yaw
        if self._compose_log_counter % 100 == 0:
            from scipy.spatial.transform import Rotation as R
            combined_head_rotation = R.from_matrix(combined[0][:3, :3])
            _, _, combined_head_yaw = combined_head_rotation.as_euler("xyz")
            combined_head_yaw_deg = np.rad2deg(combined_head_yaw)
            logger.info(f"Combined head yaw after composition: {combined_head_yaw_deg:.1f}°")

        # Apply body follow logic to avoid extreme neck angles
        return self._apply_body_follow(combined)

    def _get_secondary_pose(self) -> FullBodyPose:
        """Get the secondary full body pose from speech and face tracking offsets.

        Both face tracking and speech sway are secondary movements that compose
        additively with the primary movement (breathing or explicit move).
        """
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

                # Initialize anchor on first successful command
                if self._anchor_state == AnchorState.ANCHORED and self._body_anchor_yaw == 0.0:
                    body_yaw_deg = np.rad2deg(body_yaw)
                    self._body_anchor_yaw = body_yaw_deg
                    logger.debug(f"Anchor: Initialized anchor at {body_yaw_deg:.1f}°")

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
        """Update camera worker with current breathing pose and fetch face tracking data.

        Face tracking offsets are RELATIVE (delta from breathing pose) and compose
        additively with breathing. When suppressed, offsets are set to (0,0,0,0,0,0)
        which means "no offset" in the composition.
        """
        # Check external control flag with thread safety
        with self._external_control_lock:
            external_control = self._external_control_active


        # Check if face tracking is suppressed (primary move OR external control active)
        if self._suppress_face_tracking or external_control:
            # Use neutral offsets when suppressed (means "no offset" in composition)
            self.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        elif self.camera_worker is not None:
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
        """Request the worker thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        logger.debug("Move worker stopped")

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
        logger.info("Starting enhanced movement control loop (100Hz)")

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

            # 6) Single set_target call - SKIP if external control is active
            with self._external_control_lock:
                external_control = self._external_control_active

            if not external_control:
                # Only send commands when we have control
                self._issue_control_command(head, antennas_cmd, body_yaw)
            # else: External system has full control, send nothing

            # 7) Adaptive sleep to align to next tick, then publish shared state
            sleep_time, freq_stats = self._schedule_next_tick(loop_start, freq_stats)
            self._publish_shared_state()
            self._record_frequency_snapshot(freq_stats)

            # 8) Periodic telemetry on loop frequency
            self._maybe_log_frequency(loop_count, print_interval_loops, freq_stats)

            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Movement control loop stopped")
