"""Tests for GotoQueueMove lazy (dequeue-time) start pose capture.

These tests verify the callable-sentinel fix for the "head whip" bug where
multiple queued GotoQueueMoves captured a mid-swing pose as their start at
call time rather than waiting until the move actually executes.

Tests are ordered TDD-style: they describe the desired behavior before the
implementation is in place.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eye4() -> "np.ndarray[tuple[int, int], np.dtype[np.float32]]":
    return np.eye(4, dtype=np.float32)


def _make_target() -> "np.ndarray[tuple[int, int], np.dtype[np.float32]]":
    """A target pose distinct from identity (25° yaw left)."""
    from reachy_mini.utils import create_head_pose

    return create_head_pose(0, 0, 0, 0, 0, 25, degrees=True)


def _make_start(yaw_deg: float) -> "np.ndarray[tuple[int, int], np.dtype[np.float32]]":
    from reachy_mini.utils import create_head_pose

    return create_head_pose(0, 0, 0, 0, 0, yaw_deg, degrees=True)


# ---------------------------------------------------------------------------
# Test 1: explicit array start — unchanged backward-compat regression guard
# ---------------------------------------------------------------------------


def test_call_with_array_start_works_as_before() -> None:
    """Passing an explicit numpy array start pose behaves exactly as before."""
    from robot_comic.dance_emotion_moves import GotoQueueMove

    start = _make_start(10.0)
    target = _make_target()
    move = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=start,
        duration=1.0,
    )

    # start_head_pose stored as the array (not a callable)
    assert callable(move.start_head_pose) is False

    # evaluate at t=0 returns a head pose (not None)
    head_pose, _, _ = move.evaluate(0.0)
    assert head_pose is not None

    # evaluate at t=1 returns a head pose
    head_pose_end, _, _ = move.evaluate(1.0)
    assert head_pose_end is not None


# ---------------------------------------------------------------------------
# Test 2: None start — neutral-pose fallback regression guard (gesture path)
# ---------------------------------------------------------------------------


def test_call_with_none_falls_back_to_neutral() -> None:
    """None start_head_pose still falls back to neutral pose (gesture path)."""
    from reachy_mini.utils import create_head_pose
    from robot_comic.dance_emotion_moves import GotoQueueMove

    neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    target = _make_target()
    move = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=None,
        duration=1.0,
    )

    # At t=0, result should be close to neutral
    head_pose, _, _ = move.evaluate(0.0)
    assert head_pose is not None
    np.testing.assert_allclose(head_pose, neutral, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: callable is NOT called at construction; IS called on first evaluate
# ---------------------------------------------------------------------------


def test_callable_start_invoked_on_first_evaluate() -> None:
    """Callable start_head_pose is NOT invoked at construction; IS invoked on
    first evaluate() call and cached — subsequent calls do not re-invoke."""
    from robot_comic.dance_emotion_moves import GotoQueueMove

    call_count = 0

    def start_provider() -> "np.ndarray":
        nonlocal call_count
        call_count += 1
        return _make_start(10.0)

    target = _make_target()
    move = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=start_provider,
        duration=1.0,
    )

    # Not invoked at construction
    assert call_count == 0, "callable must not be called at construction"

    # Invoked on first evaluate
    move.evaluate(0.0)
    assert call_count == 1, "callable must be called exactly once on first evaluate"

    # Not re-invoked on subsequent evaluates
    move.evaluate(0.5)
    move.evaluate(1.0)
    assert call_count == 1, "callable must not be called again after first evaluate"


# ---------------------------------------------------------------------------
# Test 4: headline — callable captures dequeue-time pose, not call-time pose
# ---------------------------------------------------------------------------


def test_callable_start_returns_dequeue_time_pose_not_call_time_pose() -> None:
    """The headline regression test.

    Two GotoQueueMoves are constructed while the mock 'current pose' reads
    as pose_A.  Before the second move evaluates, the mock pose advances to
    pose_B (simulating the first move completing).  The second move's start
    must be pose_B (dequeue time), not pose_A (call time).
    """
    from robot_comic.dance_emotion_moves import GotoQueueMove

    # Mutable state simulating the robot's live head pose
    live_pose = {"value": _make_start(0.0)}  # starts at 0° yaw

    def get_current_head_pose() -> "np.ndarray":
        return live_pose["value"].copy()

    target = _make_target()

    # Construct both moves while the head is at 0°
    move1 = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=get_current_head_pose,
        duration=1.0,
    )
    move2 = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=get_current_head_pose,
        duration=1.0,
    )

    # move1 evaluates (dequeues) — head is still at 0°
    move1.evaluate(0.0)
    captured_start_1 = move1._resolved_start_pose  # type: ignore[attr-defined]

    # Simulate move1 completing: robot is now at 25° (call-time pose_A → pose_B)
    live_pose["value"] = _make_start(25.0)

    # move2 dequeues NOW — it should capture 25°, not the 0° it was constructed with
    move2.evaluate(0.0)
    captured_start_2 = move2._resolved_start_pose  # type: ignore[attr-defined]

    np.testing.assert_allclose(
        captured_start_1,
        _make_start(0.0),
        atol=1e-4,
        err_msg="move1 start should be 0° (pose at its dequeue time)",
    )
    np.testing.assert_allclose(
        captured_start_2,
        _make_start(25.0),
        atol=1e-4,
        err_msg="move2 start should be 25° (dequeue-time pose), not 0° (call-time pose)",
    )


# ---------------------------------------------------------------------------
# Test 5: callable exception falls back to neutral — worker-loop safety
# ---------------------------------------------------------------------------


def test_callable_exception_does_not_kill_move() -> None:
    """If the callable raises, GotoQueueMove falls back to neutral pose and
    does NOT propagate the exception — the move worker loop must not crash."""
    from reachy_mini.utils import create_head_pose
    from robot_comic.dance_emotion_moves import GotoQueueMove

    neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    def bad_provider() -> "np.ndarray":
        raise RuntimeError("robot sensor unavailable")

    target = _make_target()
    move = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=bad_provider,
        duration=1.0,
    )

    # Must not raise
    head_pose, _, _ = move.evaluate(0.0)
    assert head_pose is not None

    # Should use neutral fallback
    np.testing.assert_allclose(head_pose, neutral, atol=1e-5)
