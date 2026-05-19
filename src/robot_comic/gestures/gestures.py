"""Canonical gesture implementations for Reachy Mini.

Each public function in this module implements one gesture by building a
sequence of ``GotoQueueMove`` objects and enqueuing them on the supplied
``MovementManager``.  The safety layer (cowling clamps + velocity cap)
already gates every ``set_target`` call inside ``MovementManager``, so
gesture functions need not duplicate those checks.

Gesture design targets
----------------------
- Total duration: 0.5 – 2.0 s.
- Gestures are *primary* moves: they run exclusively via the manager's
  move queue and interrupt any active breathing/idle move.
- Head pose values use degrees for readability; ``create_head_pose`` is
  called with ``degrees=True``.

Coordinate convention (matches ``MoveHead`` tool)
---------------------------------------------------
- yaw   > 0  → turn left
- yaw   < 0  → turn right
- pitch > 0  → look down
- pitch < 0  → look up
- roll  > 0  → tilt right ear down
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from reachy_mini.utils import create_head_pose
from robot_comic.dance_emotion_moves import GotoQueueMove


if TYPE_CHECKING:
    from robot_comic.moves import MovementManager
    from robot_comic.gestures.registry import GestureRegistry


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neutral reference pose
# ---------------------------------------------------------------------------

_NEUTRAL_POSE = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
_NEUTRAL_ANTENNAS = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _goto(
    manager: "MovementManager",
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    duration: float = 0.4,
) -> None:
    """Queue a single goto move to the given head pose (degrees)."""
    target = create_head_pose(0, 0, 0, roll, pitch, yaw, degrees=True)
    speed = getattr(manager, "speed_factor", 1.0)
    move = GotoQueueMove(
        target_head_pose=target,
        start_head_pose=None,  # manager fills from last pose
        target_antennas=_NEUTRAL_ANTENNAS,
        start_antennas=_NEUTRAL_ANTENNAS,
        target_body_yaw=0.0,
        start_body_yaw=0.0,
        duration=duration,
        speed_factor=speed,
    )
    manager.queue_move(move)


def _return_to_neutral(manager: "MovementManager", duration: float = 0.35) -> None:
    """Queue a short return-to-neutral move."""
    _goto(manager, yaw=0.0, pitch=0.0, roll=0.0, duration=duration)


# ---------------------------------------------------------------------------
# Canonical gestures
# ---------------------------------------------------------------------------


def shrug(manager: "MovementManager") -> None:
    """Quick head-tilt suggesting a shrug.

    Sequence:
      1. Tilt head slightly (roll left + pitch down) — 0.20 s
      2. Hold for 0.25 s (extra goto at same pose, short duration)
      3. Return to neutral — 0.20 s
    Total approx 0.65 s.
    """
    _goto(manager, roll=10.0, pitch=8.0, duration=0.20)
    _goto(manager, roll=10.0, pitch=8.0, duration=0.25)
    _return_to_neutral(manager, duration=0.20)


def nod_yes(manager: "MovementManager") -> None:
    """Double nod downward — affirmative.

    Sequence (two nods):
      down → neutral → down → neutral
    Total ≈ 1.0 s
    """
    for _ in range(2):
        _goto(manager, pitch=18.0, duration=0.22)
        _return_to_neutral(manager, duration=0.22)


def nod_no(manager: "MovementManager") -> None:
    """Double side-to-side shake — negative/dismissive.

    Sequence: right → left → right → neutral
    Total ≈ 1.0 s
    """
    _goto(manager, yaw=-22.0, duration=0.22)
    _goto(manager, yaw=22.0, duration=0.22)
    _goto(manager, yaw=-22.0, duration=0.22)
    _return_to_neutral(manager, duration=0.22)


def point_left(manager: "MovementManager") -> None:
    """Turn head decisively to the left and hold — "over there".

    Sequence:
      1. Snap to left — 0.30 s
      2. Hold — 0.50 s
      3. Return to neutral — 0.30 s
    Total ≈ 1.1 s
    """
    _goto(manager, yaw=35.0, duration=0.30)
    _goto(manager, yaw=35.0, duration=0.50)
    _return_to_neutral(manager, duration=0.30)


def point_right(manager: "MovementManager") -> None:
    """Turn head decisively to the right and hold — "over there".

    Sequence:
      1. Snap to right — 0.30 s
      2. Hold — 0.50 s
      3. Return to neutral — 0.30 s
    Total ≈ 1.1 s
    """
    _goto(manager, yaw=-35.0, duration=0.30)
    _goto(manager, yaw=-35.0, duration=0.50)
    _return_to_neutral(manager, duration=0.30)


def scan(manager: "MovementManager") -> None:
    """Slow theatrical sweep left → right → neutral — checking the audience.

    Total ≈ 1.8 s
    """
    _goto(manager, yaw=30.0, duration=0.50)
    _goto(manager, yaw=-30.0, duration=0.70)
    _return_to_neutral(manager, duration=0.40)


def lean_in(manager: "MovementManager") -> None:
    """Lean forward (pitch down) with a slight hold — "lemme tell ya...".

    Sequence:
      1. Pitch forward — 0.25 s
      2. Hold — 0.35 s
      3. Return — 0.25 s
    Total ≈ 0.85 s
    """
    _goto(manager, pitch=20.0, duration=0.25)
    _goto(manager, pitch=20.0, duration=0.35)
    _return_to_neutral(manager, duration=0.25)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

_CANONICAL_GESTURES = {
    "shrug": shrug,
    "nod_yes": nod_yes,
    "nod_no": nod_no,
    "point_left": point_left,
    "point_right": point_right,
    "scan": scan,
    "lean_in": lean_in,
}


def register_all(registry: "GestureRegistry") -> None:
    """Register all canonical gestures into *registry*."""
    for name, fn in _CANONICAL_GESTURES.items():
        registry.register(name, fn)
