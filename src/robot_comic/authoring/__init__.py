"""Offline authoring + safety-screening toolchain for custom recorded moves (#538).

This package is the *offline* half of the custom-move training loop: it turns a
small declarative keyframe spec into reachy_mini ``RecordedMove`` JSON and runs
a kinematic + dynamics safety pre-screen on the result, all without touching
the robot or the daemon.  Live verification (daemon playback, ``robot_measure_motion``,
bonk watch) is a separate phase.

Public API
----------
- :func:`robot_comic.authoring.move_authoring.build_recorded_move`
- :func:`robot_comic.authoring.move_screen.screen_recorded_move`
"""

from robot_comic.authoring.move_screen import (
    ScreenLimits,
    ScreenResult,
    screen_recorded_move,
)
from robot_comic.authoring.move_authoring import (
    Keyframe,
    KeyframeSpec,
    build_recorded_move,
)


__all__ = [
    "Keyframe",
    "KeyframeSpec",
    "build_recorded_move",
    "ScreenLimits",
    "ScreenResult",
    "screen_recorded_move",
]
