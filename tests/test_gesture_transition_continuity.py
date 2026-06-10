"""Tests for gesture transition continuity (#521, jerkiness source #1).

Gestures previously hardcoded their interpolation start at the neutral pose
(head), (0, 0) (antennas) and 0.0 (body yaw). Queued while the robot is
off-neutral — mid-breathing sway, after an emotion, with a yawed body — the
first evaluate() snapped the commanded pose to the neutral-based path: a C0
discontinuity. The fix extends GotoQueueMove's lazy callable-sentinel pattern
(already used for the head, see test_goto_queue_move_lazy_start.py) to
antennas and body yaw, and makes the gesture _goto helper start every gesture
from the manager's last commanded full-body pose with smoothstep easing.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np

from robot_comic.dance_emotion_moves import GotoQueueMove


def _make_pose(yaw_deg: float) -> "np.ndarray":
    from reachy_mini.utils import create_head_pose

    return create_head_pose(0, 0, 0, 0, 0, yaw_deg, degrees=True)


# ---------------------------------------------------------------------------
# GotoQueueMove: callable start_antennas / start_body_yaw
# ---------------------------------------------------------------------------


def test_callable_start_antennas_resolved_on_first_evaluate() -> None:
    calls = 0

    def antennas_provider() -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return (0.2, -0.1)

    move = GotoQueueMove(
        target_head_pose=_make_pose(25.0),
        target_antennas=(0.0, 0.0),
        start_antennas=antennas_provider,
        duration=1.0,
    )
    assert calls == 0, "callable must not be invoked at construction"

    _, antennas, _ = move.evaluate(0.0)
    assert calls == 1
    np.testing.assert_allclose(antennas, [0.2, -0.1], atol=1e-6)

    move.evaluate(0.5)
    assert calls == 1, "callable must be cached after first evaluate"


def test_callable_start_body_yaw_resolved_on_first_evaluate() -> None:
    calls = 0

    def body_yaw_provider() -> float:
        nonlocal calls
        calls += 1
        return 0.3

    move = GotoQueueMove(
        target_head_pose=_make_pose(25.0),
        target_body_yaw=0.0,
        start_body_yaw=body_yaw_provider,
        duration=1.0,
    )
    assert calls == 0

    _, _, body_yaw = move.evaluate(0.0)
    assert calls == 1
    assert body_yaw is not None
    np.testing.assert_allclose(body_yaw, 0.3, atol=1e-6)


def test_callable_antennas_exception_falls_back_to_neutral() -> None:
    def bad_provider() -> tuple[float, float]:
        raise RuntimeError("status unavailable")

    move = GotoQueueMove(
        target_head_pose=_make_pose(25.0),
        start_antennas=bad_provider,
        start_body_yaw=bad_provider,
        duration=1.0,
    )
    head, antennas, body_yaw = move.evaluate(0.0)
    assert head is not None
    np.testing.assert_allclose(antennas, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(body_yaw, 0.0, atol=1e-6)


def test_tuple_start_antennas_unchanged() -> None:
    """Backward compat: explicit tuples behave exactly as before."""
    move = GotoQueueMove(
        target_head_pose=_make_pose(25.0),
        target_antennas=(0.0, 0.0),
        start_antennas=(0.1, 0.1),
        start_body_yaw=0.2,
        duration=1.0,
    )
    _, antennas, body_yaw = move.evaluate(0.0)
    np.testing.assert_allclose(antennas, [0.1, 0.1], atol=1e-6)
    np.testing.assert_allclose(body_yaw, 0.2, atol=1e-6)


# ---------------------------------------------------------------------------
# MovementManager.last_commanded_full_pose accessor
# ---------------------------------------------------------------------------


def test_manager_exposes_last_commanded_full_pose() -> None:
    from robot_comic.moves import MovementManager

    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, (0.0, 0.0))
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)

    manager = MovementManager(current_robot=robot)
    head, antennas, body_yaw = manager.last_commanded_full_pose()
    assert head.shape == (4, 4)
    assert len(antennas) == 2
    assert isinstance(float(body_yaw), float)


# ---------------------------------------------------------------------------
# Gesture _goto: starts from the live commanded pose, eased
# ---------------------------------------------------------------------------


def test_gesture_goto_starts_from_live_commanded_pose() -> None:
    """A gesture queued while the robot is off-neutral must begin its
    interpolation exactly at the manager's last commanded pose — no snap."""
    from robot_comic.gestures.gestures import _goto

    live_head = _make_pose(15.0)
    live_antennas = (0.2, -0.1)
    live_body_yaw = 0.3

    manager = MagicMock()
    manager.speed_factor = 1.0
    manager.last_commanded_full_pose.return_value = (live_head, live_antennas, live_body_yaw)

    _goto(manager, yaw=40.0, duration=0.4)

    move = manager.queue_move.call_args[0][0]
    # Nothing resolved at queue time — capture happens at dequeue (evaluate).
    assert manager.last_commanded_full_pose.call_count == 0

    head, antennas, body_yaw = move.evaluate(0.0)
    np.testing.assert_allclose(head, live_head, atol=1e-5)
    np.testing.assert_allclose(antennas, list(live_antennas), atol=1e-6)
    np.testing.assert_allclose(body_yaw, live_body_yaw, atol=1e-6)


def test_gesture_goto_uses_smoothstep_easing() -> None:
    """Gesture moves ease in/out (zero endpoint velocity) instead of the
    legacy linear ramp."""
    from robot_comic.gestures.gestures import _goto

    manager = MagicMock()
    manager.speed_factor = 1.0
    manager.last_commanded_full_pose.return_value = (_make_pose(0.0), (0.0, 0.0), 0.0)

    _goto(manager, yaw=40.0, duration=0.4)
    move = manager.queue_move.call_args[0][0]
    assert move.ease is True
