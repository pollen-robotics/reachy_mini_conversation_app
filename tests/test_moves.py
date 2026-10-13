import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import (
    BreathingMove,
    MovementManager,
    LoopFrequencyStats,
    clone_full_body_pose,
)
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove, EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = MagicMock()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)


def test_clone_full_body_pose_is_a_deep_copy() -> None:
    """Cloning a pose must not alias the head-pose array of the original."""
    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    original = (head, (1.0, 2.0), 3.0)
    clone = clone_full_body_pose(original)
    head[0, 0] = 999.0
    assert clone[0][0, 0] != 999.0
    assert clone[1] == (1.0, 2.0)
    assert clone[2] == 3.0


def test_loop_frequency_stats_reset_keeps_last_potential() -> None:
    """Reset clears accumulators but preserves last/potential frequency."""
    stats = LoopFrequencyStats(mean=5.0, m2=2.0, min_freq=1.0, count=10, last_freq=59.0, potential_freq=61.0)
    stats.reset()
    assert stats.mean == 0.0
    assert stats.m2 == 0.0
    assert stats.count == 0
    assert stats.min_freq == float("inf")
    assert stats.last_freq == 59.0
    assert stats.potential_freq == 61.0


def test_breathing_move_interpolates_then_breathes() -> None:
    """Phase 1 starts at the given antennas; phase 2 keeps body yaw neutral."""
    move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(0.3, -0.3),
        interpolation_duration=1.0,
    )
    head_start, antennas_start, body_yaw_start = move.evaluate(0.0)
    assert head_start is not None
    np.testing.assert_allclose(antennas_start, [0.3, -0.3])
    assert body_yaw_start == 0.0

    head_breathe, antennas_breathe, body_yaw_breathe = move.evaluate(5.0)
    assert head_breathe is not None
    assert antennas_breathe is not None and antennas_breathe.shape == (2,)
    assert body_yaw_breathe == 0.0


def test_is_idle_reflects_listening_and_activity() -> None:
    """is_idle is False while listening and True once past the inactivity delay."""
    manager = MovementManager(MagicMock())

    manager._shared_is_listening = True
    assert manager.is_idle() is False

    manager._shared_is_listening = False
    manager._shared_last_activity_time = manager._now()
    assert manager.is_idle() is False

    manager._shared_last_activity_time = manager._now() - 10.0
    assert manager.is_idle() is True


def test_handle_command_queue_and_clear() -> None:
    """queue_move appends real moves, ignores bad payloads, and clear empties the queue."""
    manager = MovementManager(MagicMock())
    now = manager._now()
    move = GotoQueueMove(target_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True))

    manager._handle_command("queue_move", move, now)
    assert list(manager.move_queue) == [move]

    manager._handle_command("queue_move", "not-a-move", now)
    assert list(manager.move_queue) == [move]

    manager._handle_command("clear_queue", None, now)
    assert len(manager.move_queue) == 0
    assert manager.state.current_move is None
