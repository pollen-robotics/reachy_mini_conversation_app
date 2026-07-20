from unittest.mock import MagicMock

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove, DanceQueueMove


def test_goto_evaluate_at_start_returns_start_state() -> None:
    """At t=0 the goto move sits at its start antennas and body yaw."""
    move = GotoQueueMove(
        target_head_pose=create_head_pose(0, 0, 0, 0, 0, 40, degrees=True),
        start_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        target_antennas=(1.0, 1.0),
        start_antennas=(0.0, 0.0),
        target_body_yaw=10.0,
        start_body_yaw=0.0,
        duration=2.0,
    )
    _, antennas, body_yaw = move.evaluate(0.0)
    assert antennas is not None
    np.testing.assert_allclose(antennas, [0.0, 0.0])
    assert body_yaw == 0.0


def test_goto_evaluate_midpoint_interpolates_linearly() -> None:
    """Halfway through the duration antennas and body yaw are halfway to target."""
    move = GotoQueueMove(
        target_head_pose=create_head_pose(0, 0, 0, 0, 0, 40, degrees=True),
        target_antennas=(1.0, 1.0),
        start_antennas=(0.0, 0.0),
        target_body_yaw=10.0,
        start_body_yaw=0.0,
        duration=2.0,
    )
    _, antennas, body_yaw = move.evaluate(1.0)
    assert antennas is not None
    np.testing.assert_allclose(antennas, [0.5, 0.5])
    assert body_yaw == 5.0


def test_goto_evaluate_clamps_past_duration() -> None:
    """Past the duration t is clamped so antennas reach the target."""
    move = GotoQueueMove(
        target_head_pose=create_head_pose(0, 0, 0, 0, 0, 40, degrees=True),
        target_antennas=(1.0, 1.0),
        start_antennas=(0.0, 0.0),
        target_body_yaw=10.0,
        start_body_yaw=0.0,
        duration=2.0,
    )
    _, antennas, body_yaw = move.evaluate(99.0)
    assert antennas is not None
    np.testing.assert_allclose(antennas, [1.0, 1.0])
    assert body_yaw == 10.0


def test_dance_queue_move_falls_back_to_neutral_on_error() -> None:
    """A failing wrapped dance move yields a neutral pose instead of raising."""
    move = DanceQueueMove.__new__(DanceQueueMove)
    move.move_name = "broken"
    move.dance_move = MagicMock()
    move.dance_move.evaluate.side_effect = RuntimeError("boom")
    head_pose, antennas, body_yaw = move.evaluate(0.5)
    assert head_pose is not None
    np.testing.assert_allclose(antennas, [0.0, 0.0])
    assert body_yaw == 0.0


def test_dance_queue_move_converts_tuple_antennas() -> None:
    """Tuple antennas from the wrapped move are converted to a numpy array."""
    move = DanceQueueMove.__new__(DanceQueueMove)
    move.move_name = "ok"
    move.dance_move = MagicMock()
    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    move.dance_move.evaluate.return_value = (head, (0.1, 0.2), 0.5)
    _, antennas, body_yaw = move.evaluate(0.0)
    assert isinstance(antennas, np.ndarray)
    np.testing.assert_allclose(antennas, [0.1, 0.2])
    assert body_yaw == 0.5
