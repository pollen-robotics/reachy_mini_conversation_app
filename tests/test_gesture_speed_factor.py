"""Tests that _goto forwards MovementManager.speed_factor to GotoQueueMove.

All seven canonical gestures route through _goto(), which must read
``speed_factor`` from the manager and pass it through so the admin UI
movement-speed slider actually affects gesture playback speed.
"""

from __future__ import annotations
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(speed_factor: float | None = None):
    """Return a minimal fake manager that captures queued moves.

    If *speed_factor* is None, returns a bare SimpleNamespace WITHOUT the
    attribute so we can test the getattr default-to-1.0 path.
    """
    queued: list = []

    if speed_factor is None:
        manager = SimpleNamespace(queue_move=queued.append)
    else:
        manager = SimpleNamespace(speed_factor=speed_factor, queue_move=queued.append)

    return manager, queued


# ---------------------------------------------------------------------------
# shrug with speed_factor = 0.5 → durations should be doubled
# ---------------------------------------------------------------------------


class TestShrugSpeedFactor:
    """shrug() enqueues 3 GotoQueueMoves; their durations must scale with speed_factor."""

    # Design-intent baseline durations for shrug (from gestures.py):
    # move 1: 0.20 s, move 2: 0.25 s, move 3 (return): 0.20 s
    BASELINE = (0.20, 0.25, 0.20)

    def test_speed_factor_half_doubles_all_durations(self) -> None:
        """speed_factor=0.5 → each duration = baseline / 0.5 = doubled."""
        from robot_comic.gestures.gestures import shrug

        manager, queued = _make_manager(speed_factor=0.5)
        shrug(manager)

        assert len(queued) == 3, f"Expected 3 queued moves, got {len(queued)}"
        expected = [d / 0.5 for d in self.BASELINE]  # [0.40, 0.50, 0.40]
        for i, (move, exp) in enumerate(zip(queued, expected)):
            assert move.duration == pytest.approx(exp, rel=1e-3), (
                f"Move {i}: expected duration {exp:.3f}, got {move.duration:.3f}"
            )

    def test_speed_factor_one_preserves_baseline_durations(self) -> None:
        """speed_factor=1.0 → durations are the design-intent baseline."""
        from robot_comic.gestures.gestures import shrug

        manager, queued = _make_manager(speed_factor=1.0)
        shrug(manager)

        assert len(queued) == 3
        for i, (move, exp) in enumerate(zip(queued, self.BASELINE)):
            assert move.duration == pytest.approx(exp, rel=1e-3), (
                f"Move {i}: expected {exp:.3f}, got {move.duration:.3f}"
            )

    def test_missing_speed_factor_attribute_defaults_to_one(self) -> None:
        """A manager without speed_factor should behave identically to speed_factor=1.0."""
        from robot_comic.gestures.gestures import shrug

        manager, queued = _make_manager(speed_factor=None)  # no attribute
        shrug(manager)

        assert len(queued) == 3
        for i, (move, exp) in enumerate(zip(queued, self.BASELINE)):
            assert move.duration == pytest.approx(exp, rel=1e-3), (
                f"Move {i}: expected {exp:.3f}, got {move.duration:.3f}"
            )
