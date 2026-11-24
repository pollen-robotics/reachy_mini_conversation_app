import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import (
    REST_ANTENNAS,
    SETTLE_MAX_SECONDS,
    SETTLE_MIN_SECONDS,
    SettleMove,
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


def test_listening_applies_immediately_and_preserves_queued_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listening starts immediately, and rapid queued changes retain their order."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    monkeypatch.setattr(manager, "_now", lambda: 0.0)
    manager.set_listening(True)
    manager._poll_signals(0.0)
    assert manager.get_status()["is_listening"]
    assert manager._calculate_blended_antennas((0.7, -0.7)) == (0.0, 0.0)

    manager.set_listening(False)
    manager.set_listening(True)
    manager.set_listening(False)
    manager._poll_signals(0.0)
    assert not manager.get_status()["is_listening"]


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


def test_gesture_yields_to_other_moves_and_cancels_alone() -> None:
    """A gesture never stacks on another move, and cancelling it leaves that move in place."""
    manager = MovementManager(MagicMock())
    now = manager._now()
    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    gesture, dance = _FakeMove(head), _FakeMove(head)

    manager._handle_command("play_gesture", gesture, now)
    manager._manage_move_queue(now)
    assert manager.state.current_move is gesture

    manager._handle_command("queue_move", GotoQueueMove(target_head_pose=head), now)
    manager._handle_command("cancel_gesture", None, now)
    assert manager.state.current_move is None
    assert len(manager.move_queue) == 1

    manager.move_queue.clear()
    manager.state.current_move = dance
    manager._handle_command("play_gesture", gesture, now)
    assert len(manager.move_queue) == 0
    manager._handle_command("cancel_gesture", None, now)
    assert manager.state.current_move is dance


def _tilted_pose(pitch_degrees: float, antennas: tuple[float, float] = REST_ANTENNAS) -> tuple:
    return (create_head_pose(0, 0, 0, 0, pitch_degrees, 0, degrees=True), antennas, 0.0)


def test_settle_glides_to_rest_gently_in_a_time_that_grows_with_distance() -> None:
    """The glide starts at the held pose, ends at rest with no jolt, and takes longer the further it has to go."""
    small, large = SettleMove(_tilted_pose(12.0)), SettleMove(_tilted_pose(24.0, (2.4, -2.4)))

    assert SETTLE_MIN_SECONDS <= small.duration < large.duration <= SETTLE_MAX_SECONDS
    head, antennas, body_yaw = large.evaluate(0.0)
    np.testing.assert_allclose(head, _tilted_pose(24.0)[0], atol=1e-9)
    np.testing.assert_allclose(antennas, (2.4, -2.4))
    head, antennas, body_yaw = large.evaluate(large.duration)
    np.testing.assert_allclose(head, np.eye(4), atol=1e-9)
    np.testing.assert_allclose(antennas, REST_ANTENNAS)
    assert body_yaw == 0.0
    # Minimum jerk: nearly still just after starting and just before arriving.
    early = large.evaluate(0.02 * large.duration)[1]
    assert abs(early[0] - 2.4) < 0.01 * abs(2.4 - REST_ANTENNAS[0])
    assert SettleMove(_tilted_pose(0.0)).duration == 0.0


def test_listening_settles_a_held_pose_back_to_rest() -> None:
    """A gesture cut off by the user speaking glides back to rest, antennas included, instead of freezing mid-move."""
    manager = MovementManager(MagicMock())
    now = manager._now()
    manager._last_commanded_pose = clone_full_body_pose(_tilted_pose(20.0, (1.5, -1.5)))
    manager._handle_command("set_listening", True, now)

    manager._update_primary_motion(now)
    settle = manager.state.current_move
    assert isinstance(settle, SettleMove)

    # Halfway: the head is between the held tilt and level, and the antennas follow the glide despite listening.
    head, antennas, _ = manager._get_primary_pose(now + settle.duration / 2)
    assert not np.allclose(head, np.eye(4), atol=1e-3)
    assert not np.allclose(head, _tilted_pose(20.0)[0], atol=1e-3)
    commanded = manager._calculate_blended_antennas(antennas)
    assert commanded == pytest.approx(tuple(antennas))
    assert REST_ANTENNAS[0] < commanded[0] < 1.5

    # The last tick of the glide reaches rest; after it, nothing glides again and the antennas stay frozen there.
    head, antennas, body_yaw = manager._get_primary_pose(now + settle.duration)
    manager._last_commanded_pose = (head, manager._calculate_blended_antennas(antennas), body_yaw)
    manager._update_primary_motion(now + settle.duration + 0.01)
    assert manager.state.current_move is None
    assert manager._calculate_blended_antennas((0.7, -0.7)) == pytest.approx(REST_ANTENNAS, abs=1e-6)


def test_settle_waits_for_listening_and_gives_way_to_any_move() -> None:
    """Nothing glides while the robot isn't listening, and a gesture or other move replaces a glide at once."""
    manager = MovementManager(MagicMock())
    now = manager._now()
    manager._last_commanded_pose = clone_full_body_pose(_tilted_pose(20.0))
    manager._update_primary_motion(now)
    assert not isinstance(manager.state.current_move, SettleMove)

    manager.state.current_move = None
    manager.move_queue.clear()
    manager._handle_command("set_listening", True, now)
    manager._update_primary_motion(now)
    assert isinstance(manager.state.current_move, SettleMove)

    gesture = _FakeMove(create_head_pose(0, 0, 0, 0, 0, 0, degrees=True))
    manager._handle_command("play_gesture", gesture, now)
    manager._update_primary_motion(now + 0.1)
    assert manager.state.current_move is gesture
