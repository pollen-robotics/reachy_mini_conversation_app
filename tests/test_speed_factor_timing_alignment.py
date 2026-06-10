"""Tests for speed-factor timing alignment (#521, jerkiness sources #3 and #4).

The global movement speed factor scales goto/emotion/dance durations, but two
fixed-time blends ignored it: the breathing move's ease-in-to-neutral (1.0 s)
and the antenna listening-unfreeze blend (0.4 s). At slow speed factors those
ran at full speed against globally slowed primary motion — a visible tempo
mismatch at exactly the transition moments #521 is about.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np

from robot_comic.moves import BreathingMove, MovementManager


def _make_manager() -> MovementManager:
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, (0.0, 0.0))
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    return MovementManager(current_robot=robot)


# ---------------------------------------------------------------------------
# BreathingMove ease-in scales with speed factor
# ---------------------------------------------------------------------------


def test_breathing_interpolation_scales_with_speed_factor() -> None:
    move = BreathingMove(
        interpolation_start_pose=np.eye(4, dtype=np.float32),
        interpolation_start_antennas=(0.0, 0.0),
        interpolation_duration=1.0,
        speed_factor=0.5,
    )
    assert move.interpolation_duration == 2.0


def test_breathing_speed_factor_default_preserves_legacy() -> None:
    move = BreathingMove(
        interpolation_start_pose=np.eye(4, dtype=np.float32),
        interpolation_start_antennas=(0.0, 0.0),
        interpolation_duration=1.0,
    )
    assert move.interpolation_duration == 1.0


def test_breathing_speed_factor_is_clamped() -> None:
    move = BreathingMove(
        interpolation_start_pose=np.eye(4, dtype=np.float32),
        interpolation_start_antennas=(0.0, 0.0),
        interpolation_duration=1.0,
        speed_factor=0.01,  # below the 0.1 clamp
    )
    assert move.interpolation_duration == 10.0


def test_manager_passes_speed_factor_to_breathing() -> None:
    manager = _make_manager()
    manager.speed_factor = 0.5
    manager.idle_inactivity_delay = 0.0
    manager.state.last_activity_time = -100.0

    manager._manage_breathing(0.0)

    assert len(manager.move_queue) == 1
    breathing = manager.move_queue[0]
    assert isinstance(breathing, BreathingMove)
    assert breathing.interpolation_duration == 2.0


# ---------------------------------------------------------------------------
# Antenna unfreeze blend scales with speed factor
# ---------------------------------------------------------------------------


def _simulate_unfreeze(manager: MovementManager, dt: float) -> float:
    """Run one unfreeze blend step of dt seconds; return the new blend value."""
    manager._now = lambda: dt  # type: ignore[method-assign]
    manager._is_listening = False
    manager._listening_antennas = (0.5, -0.5)
    manager._antenna_unfreeze_blend = 0.0
    manager._last_listening_blend_time = 0.0
    manager._calculate_blended_antennas((0.0, 0.0))
    return manager._antenna_unfreeze_blend


def test_antenna_unfreeze_blend_scales_with_speed_factor() -> None:
    manager = _make_manager()

    manager.speed_factor = 1.0
    blend_full_speed = _simulate_unfreeze(manager, 0.2)
    np.testing.assert_allclose(blend_full_speed, 0.5, atol=1e-6)  # 0.2 / 0.4

    manager.speed_factor = 0.5
    blend_half_speed = _simulate_unfreeze(manager, 0.2)
    np.testing.assert_allclose(blend_half_speed, 0.25, atol=1e-6)  # 0.2 / (0.4/0.5)
