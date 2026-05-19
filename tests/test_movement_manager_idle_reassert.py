"""Tests for MovementManager idle re-assert (issue #479).

When no primary move is queued, the 60 Hz control loop must continuously
re-assert the last-commanded full-body target so the motor-controller PID
always has an explicit hold target.
"""

from __future__ import annotations
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np


def _make_manager(*, idle_animation_enabled: bool = False) -> tuple[Any, MagicMock]:
    """Return (MovementManager, robot_mock) with a minimal robot stub."""
    from robot_comic.moves import MovementManager

    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, (0.0, 0.0))
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    robot.set_target.return_value = None
    robot.enable_motors.return_value = None

    manager = MovementManager(current_robot=robot)
    manager.idle_animation_enabled = idle_animation_enabled
    return manager, robot


def _run_loop_ticks(manager: Any, seconds: float = 0.15) -> None:
    """Start the loop, let it run for *seconds*, then stop it."""
    manager.start()
    time.sleep(seconds)
    manager._stop_event.set()
    if manager._thread:
        manager._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Core idle re-assert behaviour
# ---------------------------------------------------------------------------


def test_idle_reasserts_last_target_on_subsequent_ticks() -> None:
    """With no primary move queued, set_target is called repeatedly with the same args."""
    manager, robot = _make_manager()

    # Record every set_target call.
    calls: list[tuple[Any, ...]] = []
    robot.set_target.side_effect = lambda **kw: calls.append(
        (kw.get("head"), tuple(kw.get("antennas", (0.0, 0.0))), kw.get("body_yaw"))
    )

    _run_loop_ticks(manager, seconds=0.15)

    # At 60 Hz for 0.15 s we expect roughly 9 ticks.  Require at least 5.
    assert len(calls) >= 5, f"expected ≥5 set_target calls, got {len(calls)}"

    # All calls after the first must use the same antennas and body_yaw as the
    # first call (no drift between ticks in true idle).
    first_antennas = calls[0][1]
    first_body_yaw = calls[0][2]
    for i, (_, antennas, body_yaw) in enumerate(calls[1:], start=1):
        assert antennas == first_antennas, f"tick {i}: antennas changed from {first_antennas} to {antennas}"
        assert body_yaw == first_body_yaw, f"tick {i}: body_yaw changed from {first_body_yaw} to {body_yaw}"


def test_idle_target_initialised_to_neutral() -> None:
    """_last_idle_target starts as the neutral pose on construction."""
    from robot_comic.moves import MovementManager

    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, (0.0, 0.0))
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)

    manager = MovementManager(current_robot=robot)

    _, antennas, body_yaw = manager._last_idle_target
    assert antennas == (0.0, 0.0), f"expected neutral antennas (0,0), got {antennas}"
    assert body_yaw == 0.0, f"expected neutral body_yaw 0.0, got {body_yaw}"


# ---------------------------------------------------------------------------
# Primary-move-active: idle re-assert must NOT fire
# ---------------------------------------------------------------------------


def test_primary_move_suppresses_idle_reassert() -> None:
    """While a primary move is executing the loop must NOT re-assert the idle target."""
    from reachy_mini.motion.move import Move

    class _FixedMove(Move):
        """Move that holds a fixed, recognisable antenna target for its duration."""

        duration = 0.5  # type: ignore[assignment]

        def evaluate(self, t: float) -> tuple:  # type: ignore[override]
            import numpy as np

            from reachy_mini.utils import create_head_pose

            head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (head, np.array([0.99, -0.99]), 0.0)

    manager, robot = _make_manager()

    recorded_antennas: list[tuple[float, float]] = []
    robot.set_target.side_effect = lambda **kw: recorded_antennas.append(
        tuple(float(x) for x in kw.get("antennas", (0.0, 0.0)))  # type: ignore[misc]
    )

    manager.queue_move(_FixedMove())
    _run_loop_ticks(manager, seconds=0.2)

    # During the move window, antennas should reflect the move's output (0.99, -0.99),
    # NOT the idle target (0.0, 0.0).
    move_frames = [a for a in recorded_antennas if abs(a[0] - 0.99) < 0.05]
    assert move_frames, (
        "Expected at least one set_target call with the move's antenna target (0.99, -0.99); "
        f"recorded: {recorded_antennas[:10]}"
    )


# ---------------------------------------------------------------------------
# _last_idle_target is updated when a new target is successfully sent
# ---------------------------------------------------------------------------


def test_last_idle_target_updated_after_successful_set_target() -> None:
    """_last_idle_target reflects the most recently sent set_target args."""
    from reachy_mini.utils import create_head_pose

    manager, robot = _make_manager()

    custom_head = create_head_pose(0, 0, 0.05, 0, 0, 0, degrees=False, mm=False)
    custom_antennas = (0.3, -0.3)
    custom_body_yaw = 0.42

    # Directly call _issue_control_command to simulate a successfully sent target.
    manager._issue_control_command(custom_head, custom_antennas, custom_body_yaw)

    _, antennas, body_yaw = manager._last_idle_target
    assert abs(antennas[0] - custom_antennas[0]) < 1e-6, (
        f"antenna[0] mismatch: expected {custom_antennas[0]}, got {antennas[0]}"
    )
    assert abs(antennas[1] - custom_antennas[1]) < 1e-6, (
        f"antenna[1] mismatch: expected {custom_antennas[1]}, got {antennas[1]}"
    )
    assert abs(body_yaw - custom_body_yaw) < 1e-6, f"body_yaw mismatch: expected {custom_body_yaw}, got {body_yaw}"


def test_last_idle_target_not_updated_on_failed_set_target() -> None:
    """A set_target failure must NOT update _last_idle_target."""
    from robot_comic.moves import clone_full_body_pose

    manager, robot = _make_manager()

    initial_target = clone_full_body_pose(manager._last_idle_target)

    robot.set_target.side_effect = RuntimeError("bus error")

    from reachy_mini.utils import create_head_pose

    bad_head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    manager._issue_control_command(bad_head, (0.99, -0.99), 1.23)

    # _last_idle_target must be unchanged
    _, antennas, body_yaw = manager._last_idle_target
    _, init_antennas, init_body_yaw = initial_target
    assert antennas == init_antennas
    assert body_yaw == init_body_yaw


# ---------------------------------------------------------------------------
# Queue-non-empty: idle re-assert must also be suppressed while moves queued
# ---------------------------------------------------------------------------


def test_idle_reassert_suppressed_when_moves_queued() -> None:
    """Even if current_move is None, a non-empty queue means a move is imminent
    and the idle path must not fire.
    """
    from reachy_mini.motion.move import Move

    class _LongMove(Move):
        duration = 5.0  # type: ignore[assignment]

        def evaluate(self, t: float) -> tuple:  # type: ignore[override]
            import numpy as np

            from reachy_mini.utils import create_head_pose

            return (create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), np.array([0.5, -0.5]), 0.0)

    manager, robot = _make_manager()

    # Pre-load a move into the internal deque BEFORE the loop starts so the
    # loop picks it up immediately on the first tick.
    manager.move_queue.append(_LongMove())

    recorded_antennas: list[tuple[float, float]] = []
    robot.set_target.side_effect = lambda **kw: recorded_antennas.append(
        tuple(float(x) for x in kw.get("antennas", (0.0, 0.0)))  # type: ignore[misc]
    )

    _run_loop_ticks(manager, seconds=0.15)

    move_frames = [a for a in recorded_antennas if abs(a[0] - 0.5) < 0.1]
    assert move_frames, (
        f"Expected set_target calls reflecting the queued move's antenna target; recorded: {recorded_antennas[:10]}"
    )
