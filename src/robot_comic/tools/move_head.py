import os
import math
import time
import logging
import threading
from typing import Any, Dict, Tuple, Literal, Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini.utils import create_head_pose
from robot_comic.tools.core_tools import Tool, ToolDependencies
from robot_comic.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Direction = Literal["left", "right", "up", "down", "front"]


# Default cooldown between consecutive move_head invocations (seconds).
# Hardware finding 2026-05-16: rapid LLM-driven sequences ("left, left, left")
# stack stale GotoQueueMoves whose interpolation start poses no longer match
# the live head pose. The MovementManager's per-tick velocity cap absorbs the
# resulting step into a ~0.2 s reversal, but at near-cowling yaw the reversal
# is enough to slam against the surround. A 0.6 s cooldown gives the typical
# 1 s move time to finish (or nearly finish) before another is accepted.
DEFAULT_MIN_INTERVAL_S = 0.6

# Default tolerance for "head is already at the previous target" dedupe check.
# 5 degrees ≈ 0.087 rad. If the largest per-axis delta between current head
# pose and the previous tool target is smaller than this, a same-direction
# repeat is treated as a no-op rather than re-queueing another GotoQueueMove.
DEFAULT_AT_TARGET_TOL_RAD = math.radians(5.0)


def _read_min_interval_s() -> float:
    """Read the cooldown window from the environment at call time.

    Read on every invocation (not at import) so test fixtures that patch the
    env var with monkeypatch take effect without re-importing the module.
    Falls back to ``DEFAULT_MIN_INTERVAL_S`` for missing or invalid values.
    """
    raw = os.getenv("REACHY_MINI_MOVE_HEAD_MIN_INTERVAL_S")
    if raw is None:
        return DEFAULT_MIN_INTERVAL_S
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning(
            "Invalid REACHY_MINI_MOVE_HEAD_MIN_INTERVAL_S=%r; using default %.2f",
            raw,
            DEFAULT_MIN_INTERVAL_S,
        )
        return DEFAULT_MIN_INTERVAL_S
    # Negative values are nonsensical — treat as 0 (cooldown disabled).
    return max(0.0, value)


def _yaw_pitch_roll(pose: NDArray[np.floating]) -> Tuple[float, float, float]:
    """Extract (roll, pitch, yaw) radians from a 4x4 head pose, robustly.

    Returns (0, 0, 0) on any decomposition failure so the dedupe check stays
    inert rather than raising into the tool's hot path.
    """
    try:
        from scipy.spatial.transform import Rotation

        r = Rotation.from_matrix(pose[:3, :3])
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)
        return (float(roll), float(pitch), float(yaw))
    except Exception:
        return (0.0, 0.0, 0.0)


class MoveHead(Tool):
    """Move head in a given direction."""

    name = "move_head"
    description = (
        "Move your head in a given direction: left, right, up, down or front. "
        "Use down only when the user explicitly asks you to look or move down."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
                "description": (
                    "Head direction. Use down only when the user explicitly asks "
                    "you to look or move down; avoid it for normal conversation, "
                    "idle behavior, scanning, jokes, and emotional beats."
                ),
            },
        },
        "required": ["direction"],
    }

    # mapping: direction -> args for create_head_pose
    DELTAS: Dict[str, Tuple[int, int, int, int, int, int]] = {
        "left": (0, 0, 0, 0, 0, 40),
        "right": (0, 0, 0, 0, 0, -40),
        "up": (0, 0, 0, 0, -30, 0),
        "down": (0, 0, 0, 0, 30, 0),
        "front": (0, 0, 0, 0, 0, 0),
    }

    def __init__(
        self,
        monotonic_clock: Callable[[], float] | None = None,
        at_target_tol_rad: float = DEFAULT_AT_TARGET_TOL_RAD,
    ) -> None:
        """Create a MoveHead tool.

        Parameters
        ----------
        monotonic_clock:
            Optional clock function for deterministic tests. Defaults to
            ``time.monotonic``.
        at_target_tol_rad:
            Per-axis tolerance (radians) used by the same-direction dedupe
            check. Defaults to ~5 degrees.

        """
        self._clock = monotonic_clock or time.monotonic
        self._at_target_tol_rad = at_target_tol_rad
        # Tool state used for rate-limit + dedupe. Tool instances are created
        # once at registry init and shared across the dispatcher, so this
        # state is process-global per tool instance.
        self._state_lock = threading.Lock()
        self._last_call_monotonic: float | None = None
        self._last_direction: str | None = None
        self._last_target_rpy: Tuple[float, float, float] | None = None

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Move head in a given direction."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str):
            return {"error": "direction must be a string"}
        direction: Direction = direction_raw  # type: ignore[assignment]
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)
        target_rpy = _yaw_pitch_roll(target)

        # ── Admission control: cooldown + dedupe ──────────────────────────
        # Both checks are evaluated under the state lock so concurrent
        # dispatches from BackgroundToolManager cannot race past the gate.
        min_interval_s = _read_min_interval_s()
        now = self._clock()

        with self._state_lock:
            last_call = self._last_call_monotonic
            last_direction = self._last_direction
            last_target_rpy = self._last_target_rpy

        if last_call is not None and min_interval_s > 0.0:
            elapsed = now - last_call
            if elapsed < min_interval_s:
                # Same-direction dedupe takes precedence within the cooldown
                # window — if the head is already near the previous target,
                # this is an obvious no-op and we tell the LLM so.
                if direction == last_direction and self._head_is_at_target(deps, last_target_rpy):
                    logger.info(
                        "move_head dedupe: direction=%s already at target; skipping duplicate queue (elapsed=%.2fs)",
                        direction,
                        elapsed,
                    )
                    return {"status": f"already looking {direction}"}

                remaining = max(0.0, min_interval_s - elapsed)
                logger.warning(
                    "move_head rate-limited: direction=%s elapsed=%.2fs "
                    "min_interval=%.2fs (chassis safety guard; #308 hardware "
                    "finding 2026-05-16)",
                    direction,
                    elapsed,
                    min_interval_s,
                )
                return {
                    "error": (
                        f"move_head rate-limited: try again in {remaining:.2f}s "
                        f"(chassis safety cooldown to prevent cowling impact)"
                    )
                }

        # ── Same-direction dedupe outside cooldown ────────────────────────
        # Even past the cooldown, if the last direction was the same and the
        # head is sitting at that target, queueing another GotoQueueMove just
        # repeats work. Skip it.
        if direction == last_direction and self._head_is_at_target(deps, last_target_rpy):
            logger.info(
                "move_head dedupe: direction=%s already at target; skipping",
                direction,
            )
            return {"status": f"already looking {direction}"}

        # ── Queue the move ────────────────────────────────────────────────
        try:
            movement_manager = deps.movement_manager

            # Get current antennas now (no queueing concern for antennas).
            # Head pose start is captured LAZILY at dequeue time via callable
            # so that rapid back-to-back move_head calls don't all snapshot the
            # same mid-swing pose (the "head whip" / ~11° snap bug).
            _, current_antennas = deps.reachy_mini.get_current_joint_positions()

            # Create goto move
            try:
                speed = float(getattr(movement_manager, "speed_factor", 1.0))
            except (TypeError, ValueError):
                speed = 1.0
            goto_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=deps.reachy_mini.get_current_head_pose,  # callable — lazy dequeue-time capture
                target_antennas=(0, 0),  # Reset antennas to default
                start_antennas=(
                    current_antennas[0],
                    current_antennas[1],
                ),  # Skip body_yaw
                target_body_yaw=0,  # Reset body yaw
                start_body_yaw=current_antennas[0],  # body_yaw is first in joint positions
                duration=deps.motion_duration_s,
                speed_factor=speed,
                # Cubic smoothstep ramp (#264): zero start/end velocity removes
                # the visible snap when consecutive move_head calls chain in
                # quick succession (greet scan sweep, etc).
                ease=True,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.set_moving_state(goto_move.duration)

            with self._state_lock:
                self._last_call_monotonic = now
                self._last_direction = direction
                self._last_target_rpy = target_rpy

            return {"status": f"looking {direction}"}

        except Exception as e:
            logger.error("move_head failed")
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}

    def _head_is_at_target(
        self,
        deps: ToolDependencies,
        target_rpy: Tuple[float, float, float] | None,
    ) -> bool:
        """Return True when the live head pose is within tolerance of *target_rpy*.

        Used to dedupe same-direction repeat calls. On any sensor-read or
        decomposition error, returns False so we err on the side of queueing
        the move (the cooldown still gates rate; the dedupe is a best-effort
        optimisation).
        """
        if target_rpy is None:
            return False
        try:
            current_head_pose = deps.reachy_mini.get_current_head_pose()
        except Exception:
            return False
        if current_head_pose is None:
            return False
        current_rpy = _yaw_pitch_roll(current_head_pose)
        for cur, tgt in zip(current_rpy, target_rpy):
            if abs(cur - tgt) > self._at_target_tol_rad:
                return False
        return True
