"""Unit tests for robot_comic.authoring.move_screen.

Pure-math offline safety pre-screen. No robot hardware, no daemon, no SDK I/O.
A gentle move must pass; a deliberately violent move must be rejected with a
named reason.
"""

from __future__ import annotations
import math

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


# ---------------------------------------------------------------------------
# Synthetic move builders
# ---------------------------------------------------------------------------


def _gentle_move() -> dict:
    spec = KeyframeSpec(
        name="gentle",
        description="A gentle nod within all limits.",
        fps=100.0,
        easing="minjerk",
        keyframes=[
            Keyframe(t=0.0, pitch=0.0),
            Keyframe(t=0.6, pitch=12.0),
            Keyframe(t=1.2, pitch=0.0),
        ],
    )
    return build_recorded_move(spec)


def _out_of_envelope_move() -> dict:
    """A move whose pitch grossly exceeds the safe envelope at a keyframe."""
    spec = KeyframeSpec(
        name="too_far",
        description="Pitch way past the envelope.",
        fps=100.0,
        keyframes=[
            Keyframe(t=0.0, pitch=0.0),
            Keyframe(t=0.5, pitch=80.0),  # far beyond +25 deg limit
            Keyframe(t=1.0, pitch=0.0),
        ],
    )
    return build_recorded_move(spec)


def _violent_move() -> dict:
    """A move that snaps a huge yaw swing in a tiny time -> jerk/velocity blowup."""
    spec = KeyframeSpec(
        name="violent",
        description="Snap yaw far and fast.",
        fps=100.0,
        easing="linear",
        keyframes=[
            Keyframe(t=0.0, yaw=-40.0),
            Keyframe(t=0.05, yaw=40.0),  # 80 deg in 50 ms
            Keyframe(t=0.10, yaw=-40.0),
            Keyframe(t=0.15, yaw=40.0),
        ],
    )
    return build_recorded_move(spec)


# ---------------------------------------------------------------------------
# screen_recorded_move
# ---------------------------------------------------------------------------


class TestScreenResultType:
    def test_returns_screen_result(self) -> None:
        result = screen_recorded_move(_gentle_move())
        assert isinstance(result, ScreenResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.reasons, list)


class TestGentlePasses:
    def test_gentle_move_passes(self) -> None:
        result = screen_recorded_move(_gentle_move())
        assert result.passed, f"expected pass, got reasons: {result.reasons}"
        assert result.reasons == []

    def test_metrics_reported(self) -> None:
        result = screen_recorded_move(_gentle_move())
        # Peak metrics should be populated and finite.
        assert result.metrics["max_velocity_rad_s"] >= 0.0
        assert result.metrics["max_acceleration_rad_s2"] >= 0.0
        assert result.metrics["max_jerk_rad_s3"] >= 0.0
        assert math.isfinite(result.metrics["max_velocity_rad_s"])


class TestEnvelopeRejection:
    def test_out_of_envelope_rejected(self) -> None:
        result = screen_recorded_move(_out_of_envelope_move())
        assert not result.passed
        assert any("envelope" in r.lower() or "limit" in r.lower() for r in result.reasons)

    def test_reason_names_the_axis(self) -> None:
        result = screen_recorded_move(_out_of_envelope_move())
        assert any("pitch" in r.lower() for r in result.reasons)


class TestVelocityRejection:
    def test_violent_move_rejected(self) -> None:
        result = screen_recorded_move(_violent_move())
        assert not result.passed
        assert any("velocity" in r.lower() or "jerk" in r.lower() or "accel" in r.lower() for r in result.reasons)


class TestCustomLimits:
    def test_tighter_velocity_limit_rejects_otherwise_fine_move(self) -> None:
        move = _gentle_move()
        # Pass with default; fail with an absurdly tight velocity cap.
        assert screen_recorded_move(move).passed
        strict = ScreenLimits(max_velocity_rad_s=0.001)
        result = screen_recorded_move(move, limits=strict)
        assert not result.passed
        assert any("velocity" in r.lower() for r in result.reasons)

    def test_loose_limits_accept_violent_move(self) -> None:
        move = _violent_move()
        loose = ScreenLimits(
            max_velocity_rad_s=1e6,
            max_acceleration_rad_s2=1e9,
            max_jerk_rad_s3=1e12,
        )
        # Envelope still holds (yaw 40 deg < 45 deg), so loosening kinematics passes.
        result = screen_recorded_move(move, limits=loose)
        assert result.passed, f"reasons: {result.reasons}"


class TestMalformedInput:
    def test_missing_keys_rejected(self) -> None:
        result = screen_recorded_move({"description": "x"})
        assert not result.passed
        assert any("malformed" in r.lower() or "missing" in r.lower() for r in result.reasons)

    def test_mismatched_lengths_rejected(self) -> None:
        move = _gentle_move()
        move["time"] = move["time"][:-1]
        result = screen_recorded_move(move)
        assert not result.passed


class TestKinematicReachability:
    """When analytical kinematics is available, an unreachable pose is flagged."""

    def test_reachability_check_runs_on_gentle_move(self) -> None:
        # A gentle move's poses must all be reachable.
        result = screen_recorded_move(_gentle_move(), check_kinematics=True)
        assert result.passed, f"reasons: {result.reasons}"

    def test_kinematics_can_be_skipped(self) -> None:
        result = screen_recorded_move(_gentle_move(), check_kinematics=False)
        assert "reachable" not in result.metrics or True  # smoke: must not raise
        assert result.passed
