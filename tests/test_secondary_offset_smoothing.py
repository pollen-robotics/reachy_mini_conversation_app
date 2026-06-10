"""Tests for secondary-offset smoothing (#521, jerkiness sources #2 and #5).

Speech-sway and face-tracking offsets are additive on top of the primary
pose, but their producers step them instantaneously: HeadWobbler.reset()
zeroes the sway the moment speech ends, and face tracking snaps on at full
amplitude when a face appears. Smoothing the *combined* secondary offsets
with a short exponential low-pass inside the manager's 60 Hz loop removes
every such step in one place. ``REACHY_MINI_SECONDARY_SMOOTHING_TAU_S``
tunes the time constant; ``0`` disables smoothing (legacy passthrough).
"""

from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np

from robot_comic.moves import MovementManager


def _make_manager() -> MovementManager:
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, (0.0, 0.0))
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    return MovementManager(current_robot=robot)


_STEP = (0.0, 0.0, 0.01, 0.0, 0.0, 0.3)  # 10 mm z + 0.3 rad yaw


def _yaw_of(manager: MovementManager, t: float) -> float:
    """Sample the secondary pose at time t and extract its yaw angle."""
    head, _, _ = manager._get_secondary_pose(t)
    # yaw from rotation matrix: atan2(R[1,0], R[0,0])
    return float(np.arctan2(head[1, 0], head[0, 0]))


def test_step_input_is_smoothed_not_instant() -> None:
    manager = _make_manager()
    manager._secondary_smoothing_tau = 0.1

    manager._get_secondary_pose(0.0)  # establish t=0 with zero offsets
    manager.state.speech_offsets = _STEP

    yaw_first = _yaw_of(manager, 1 / 60)
    assert 0.0 < yaw_first < 0.3 * 0.5, f"first tick must be a partial step, got {yaw_first}"

    # Converges monotonically toward the target
    yaw_later = _yaw_of(manager, 0.2)
    assert yaw_later > yaw_first
    yaw_settled = _yaw_of(manager, 1.0)
    np.testing.assert_allclose(yaw_settled, 0.3, atol=0.01)


def test_reset_to_zero_blends_out() -> None:
    """The HeadWobbler.reset() hard-stop: offsets dropping to zero must ramp."""
    manager = _make_manager()
    manager._secondary_smoothing_tau = 0.1

    manager._get_secondary_pose(0.0)
    manager.state.speech_offsets = _STEP
    yaw_settled = _yaw_of(manager, 1.0)
    np.testing.assert_allclose(yaw_settled, 0.3, atol=0.01)

    # Speech ends: wobbler zeroes the offsets instantly.
    manager.state.speech_offsets = (0.0,) * 6

    yaw_after = _yaw_of(manager, 1.0 + 1 / 60)
    assert yaw_after > 0.15, f"offset must blend out, not snap to zero (got {yaw_after})"
    yaw_done = _yaw_of(manager, 2.0)
    np.testing.assert_allclose(yaw_done, 0.0, atol=0.01)


def test_tau_zero_disables_smoothing() -> None:
    """Legacy passthrough: tau=0 applies offsets instantaneously."""
    manager = _make_manager()
    manager._secondary_smoothing_tau = 0.0

    manager._get_secondary_pose(0.0)
    manager.state.speech_offsets = _STEP
    yaw = _yaw_of(manager, 1 / 60)
    np.testing.assert_allclose(yaw, 0.3, atol=1e-6)


def test_smoothed_offsets_settle_exactly_to_target() -> None:
    """After convergence the smoothed value snaps to the target so the
    unchanged-offsets cache can re-engage (no asymptotic float churn)."""
    manager = _make_manager()
    manager._secondary_smoothing_tau = 0.05

    manager._get_secondary_pose(0.0)
    manager.state.speech_offsets = _STEP
    manager._get_secondary_pose(5.0)  # plenty of settle time

    assert manager._smoothed_secondary_offsets == manager.state.speech_offsets


def test_config_knob_default_and_refresh(monkeypatch) -> None:
    from robot_comic import config as config_mod

    assert config_mod.config.SECONDARY_SMOOTHING_TAU_S == 0.06

    monkeypatch.setenv("REACHY_MINI_SECONDARY_SMOOTHING_TAU_S", "0.25")
    config_mod.refresh_runtime_config_from_env()
    try:
        assert config_mod.config.SECONDARY_SMOOTHING_TAU_S == 0.25
    finally:
        monkeypatch.delenv("REACHY_MINI_SECONDARY_SMOOTHING_TAU_S")
        config_mod.refresh_runtime_config_from_env()
