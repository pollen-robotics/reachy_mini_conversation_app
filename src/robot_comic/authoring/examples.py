"""Example custom moves authored with the #538 toolchain.

These are *generic* demonstration gestures — no persona/comedian content — that
show how to drive :func:`robot_comic.authoring.move_authoring.build_recorded_move`
and validate the output with
:func:`robot_comic.authoring.move_screen.screen_recorded_move`.

Run as a script to author each example, screen it, and (optionally) write the
``RecordedMove`` JSON to disk::

    python -m robot_comic.authoring.examples --out-dir build/custom_moves
"""

from __future__ import annotations
import os
import json
import argparse
from typing import Any, Dict

from robot_comic.authoring.move_screen import ScreenResult, screen_recorded_move
from robot_comic.authoring.move_authoring import (
    Keyframe,
    KeyframeSpec,
    build_recorded_move,
)


def double_take_spec() -> KeyframeSpec:
    """Build a comic *double take*: glance away, take back with a look-up, settle.

    Stays well inside the safe head envelope (yaw +-18 deg, pitch -14..+4 deg)
    and uses minimum-jerk easing so velocity is zero at every waypoint.  The
    segment timings are sized so the minimum-jerk peak velocity (~1.875 *
    range / duration) stays under the conservative on-robot velocity cap
    (1.5 rad/s) — it passes the offline safety screen with headroom.

    Timeline (~2.85 s):
      0.00 s  neutral
      0.55 s  casual glance to the left (yaw +18), antennas relax
      0.90 s  still looking away
      1.55 s  ease back through neutral toward the right (the "take")
      2.15 s  surprised look up + slight tilt (pitch -14, roll +6), antennas perk
      2.55 s  ease back toward neutral
      2.85 s  neutral
    """
    return KeyframeSpec(
        name="double_take",
        description=(
            "A comic double take: a casual glance away, then a take back "
            "with a surprised look-up before settling to neutral."
        ),
        fps=100.0,
        easing="minjerk",
        keyframes=[
            Keyframe(t=0.00, yaw=0.0, pitch=0.0, roll=0.0, antennas=(0.0, 0.0)),
            Keyframe(t=0.55, yaw=18.0, pitch=3.0, antennas=(-0.1, -0.1)),
            Keyframe(t=0.90, yaw=18.0, pitch=3.0, antennas=(-0.1, -0.1)),
            Keyframe(t=1.55, yaw=-8.0, pitch=1.0, antennas=(0.1, 0.1)),
            Keyframe(t=2.15, yaw=-2.0, pitch=-14.0, roll=6.0, antennas=(0.4, 0.4)),
            Keyframe(t=2.55, yaw=0.0, pitch=-3.0, roll=2.0, antennas=(0.1, 0.1)),
            Keyframe(t=2.85, yaw=0.0, pitch=0.0, roll=0.0, antennas=(0.0, 0.0)),
        ],
    )


def head_tilt_emphasis_spec() -> KeyframeSpec:
    """Build a slow, curious *head tilt* used to emphasise a beat, then return.

    A single gentle roll-and-hold (right ear down) with a touch of look-up —
    the quizzical "oh really?" tilt.  Gentle minimum-jerk motion, deep inside
    the envelope.
    """
    return KeyframeSpec(
        name="head_tilt_emphasis",
        description=(
            "A slow curious head tilt to emphasise a beat: roll the head over, "
            "hold the quizzical pose, then ease back to neutral."
        ),
        fps=100.0,
        easing="minjerk",
        keyframes=[
            Keyframe(t=0.00, roll=0.0, pitch=0.0),
            Keyframe(t=0.45, roll=15.0, pitch=-6.0),
            Keyframe(t=0.95, roll=15.0, pitch=-6.0),
            Keyframe(t=1.40, roll=0.0, pitch=0.0),
        ],
    )


# Registry of named example specs.
EXAMPLES = {
    "double_take": double_take_spec,
    "head_tilt_emphasis": head_tilt_emphasis_spec,
}


def author_and_screen(name: str) -> tuple[Dict[str, Any], ScreenResult]:
    """Build a named example move and run it through the offline safety screen."""
    spec = EXAMPLES[name]()
    move = build_recorded_move(spec)
    result = screen_recorded_move(move)
    return move, result


def main() -> None:
    """CLI: author every example, screen it, optionally dump RecordedMove JSON."""
    parser = argparse.ArgumentParser(description="Author + screen example custom moves (#538)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="if set, write each <name>.json RecordedMove here",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma list of example names (default: all)",
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.only.split(",")] if args.only else sorted(EXAMPLES)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    exit_code = 0
    for name in names:
        move, result = author_and_screen(name)
        verdict = "PASS" if result.passed else "REJECT"
        print(
            f"[{verdict}] {name}: "
            f"vel={result.metrics['max_velocity_rad_s']:.2f} rad/s "
            f"acc={result.metrics['max_acceleration_rad_s2']:.1f} rad/s^2 "
            f"jerk={result.metrics['max_jerk_rad_s3']:.0f} rad/s^3 "
            f"({len(move['time'])} frames)"
        )
        for reason in result.reasons:
            print(f"    - {reason}")
            exit_code = 1
        if args.out_dir:
            path = os.path.join(args.out_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(move, fh, indent=2)
            print(f"    wrote {path}")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
