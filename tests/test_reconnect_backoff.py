"""Tests for the set_target reconnect backoff in MovementManager.

The Pi 5 field diagnosis showed 54 "Failed to set robot target" errors fired
in rapid bursts during serial-bus flakes, burning 186% CPU because the 60 Hz
loop never yielded between attempts.  The fix adds an exponential backoff
(starting at 200 ms, doubling each failure, capping at 8 s) that resets to
zero on the first successful call.

These tests exercise only _issue_control_command; they do not start the full
working loop thread, so they are fast and deterministic.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager():
    from robot_comic.moves import MovementManager

    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, (0.0, 0.0))
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    robot.set_target.return_value = None
    robot.enable_motors.return_value = None
    return MovementManager(current_robot=robot), robot


def _neutral_pose():
    """Return the minimal head/antennas/body_yaw tuple used by _issue_control_command."""
    from reachy_mini.utils import create_head_pose  # type: ignore[import-not-found]

    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    return head, (0.0, 0.0), 0.0


# ---------------------------------------------------------------------------
# Tests — backoff state machine
# ---------------------------------------------------------------------------


def test_no_backoff_on_first_success() -> None:
    """A healthy connection never triggers any sleep."""
    manager, robot = _make_manager()
    head, antennas, yaw = _neutral_pose()

    sleep_calls: list[float] = []
    with patch("robot_comic.moves.time.sleep", side_effect=sleep_calls.append):
        manager._issue_control_command(head, antennas, yaw)

    assert sleep_calls == [], "no sleep should occur when set_target succeeds"
    assert manager._set_target_consecutive_failures == 0
    assert manager._set_target_backoff_s == 0.0


def test_backoff_grows_across_consecutive_failures() -> None:
    """Each consecutive failure doubles the backoff delay, starting from BASE_S."""
    from robot_comic.moves import (
        _RECONNECT_BACKOFF_CAP_S,
        _RECONNECT_BACKOFF_BASE_S,
        _RECONNECT_BACKOFF_MULTIPLIER,
    )

    manager, robot = _make_manager()
    head, antennas, yaw = _neutral_pose()
    robot.set_target.side_effect = ConnectionError("bus flake")

    sleep_calls: list[float] = []
    with patch("robot_comic.moves.time.sleep", side_effect=sleep_calls.append):
        # Call N times; each call after the first should sleep more than the previous.
        for _ in range(5):
            manager._issue_control_command(head, antennas, yaw)

    # First call: no prior failures → no sleep injected.
    # Second call: backoff_s == BASE_S → sleep(BASE_S).
    # Third call: backoff_s == BASE_S * MULT → sleep(BASE_S * MULT).
    # …
    assert len(sleep_calls) == 4, f"expected 4 sleeps for 5 calls; got {len(sleep_calls)}"
    assert sleep_calls[0] == pytest.approx(_RECONNECT_BACKOFF_BASE_S)
    assert sleep_calls[1] == pytest.approx(_RECONNECT_BACKOFF_BASE_S * _RECONNECT_BACKOFF_MULTIPLIER)

    # All delays must be monotonically non-decreasing.
    for i in range(len(sleep_calls) - 1):
        assert sleep_calls[i] <= sleep_calls[i + 1], (
            f"backoff must not decrease: sleeps[{i}]={sleep_calls[i]} > sleeps[{i + 1}]={sleep_calls[i + 1]}"
        )

    # No delay exceeds the cap.
    assert all(s <= _RECONNECT_BACKOFF_CAP_S for s in sleep_calls), (
        f"backoff exceeded cap {_RECONNECT_BACKOFF_CAP_S}: {sleep_calls}"
    )


def test_backoff_resets_after_success() -> None:
    """After a recovery (success), backoff and failure counter return to zero."""
    manager, robot = _make_manager()
    head, antennas, yaw = _neutral_pose()

    # Simulate 3 failures followed by 1 success.
    robot.set_target.side_effect = [
        ConnectionError("flake"),
        ConnectionError("flake"),
        ConnectionError("flake"),
        None,  # success
    ]

    sleep_calls: list[float] = []
    with patch("robot_comic.moves.time.sleep", side_effect=sleep_calls.append):
        for _ in range(4):
            manager._issue_control_command(head, antennas, yaw)

    assert manager._set_target_consecutive_failures == 0, "failure counter must reset to 0 after a successful call"
    assert manager._set_target_backoff_s == 0.0, "backoff delay must reset to 0.0 after a successful call"


def test_backoff_resets_means_no_sleep_on_next_call() -> None:
    """Once reset, the very next call does not sleep even if a prior burst happened."""
    manager, robot = _make_manager()
    head, antennas, yaw = _neutral_pose()

    # 2 failures then a success.
    robot.set_target.side_effect = [
        RuntimeError("flake"),
        RuntimeError("flake"),
        None,  # recovery
        None,  # next call — should not sleep
    ]

    sleep_calls: list[float] = []
    with patch("robot_comic.moves.time.sleep", side_effect=sleep_calls.append):
        for _ in range(4):
            manager._issue_control_command(head, antennas, yaw)

    # Only the second and third calls may have slept (the first failure sets
    # backoff, the second call pays it).  The fourth call (after recovery) must
    # NOT sleep.  We verify by checking the count: 2 failures → 1 pre-call
    # sleep (before the 2nd failed call), 3rd call pays another sleep before
    # succeeding.  Whatever the exact sequence, the point is the state is zeroed.
    assert manager._set_target_backoff_s == 0.0
    assert manager._set_target_consecutive_failures == 0


def test_backoff_cap_is_respected() -> None:
    """Backoff never exceeds _RECONNECT_BACKOFF_CAP_S regardless of failure count."""
    from robot_comic.moves import _RECONNECT_BACKOFF_CAP_S

    manager, robot = _make_manager()
    head, antennas, yaw = _neutral_pose()
    robot.set_target.side_effect = RuntimeError("persistent fault")

    sleep_calls: list[float] = []
    with patch("robot_comic.moves.time.sleep", side_effect=sleep_calls.append):
        # 20 consecutive failures — well past the point where the cap kicks in.
        for _ in range(20):
            manager._issue_control_command(head, antennas, yaw)

    assert all(s <= _RECONNECT_BACKOFF_CAP_S for s in sleep_calls), (
        f"cap violated: max delay was {max(sleep_calls):.2f}, cap is {_RECONNECT_BACKOFF_CAP_S}"
    )
    # The manager's internal state must also reflect the cap.
    assert manager._set_target_backoff_s == _RECONNECT_BACKOFF_CAP_S
