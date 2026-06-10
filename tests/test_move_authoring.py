"""Unit tests for robot_comic.authoring.move_authoring.

Pure-math: no robot hardware, no SDK I/O. Exercises the keyframe -> RecordedMove
JSON authoring pipeline and round-trip validity against reachy_mini's
``RecordedMove`` loader.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_comic.authoring.move_authoring import (
    Keyframe,
    KeyframeSpec,
    build_recorded_move,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rpy_of_head(head: list[list[float]]) -> tuple[float, float, float]:
    """Extract (roll, pitch, yaw) degrees from a 4x4 head matrix (as nested lists)."""
    mat = np.array(head, dtype=np.float64)
    roll, pitch, yaw = Rotation.from_matrix(mat[:3, :3]).as_euler("xyz", degrees=True)
    return float(roll), float(pitch), float(yaw)


def _simple_spec() -> KeyframeSpec:
    """A gentle two-segment nod: neutral -> pitch down -> neutral."""
    return KeyframeSpec(
        name="test_nod",
        description="A gentle test nod.",
        fps=100.0,
        keyframes=[
            Keyframe(t=0.0, pitch=0.0),
            Keyframe(t=0.5, pitch=15.0),
            Keyframe(t=1.0, pitch=0.0),
        ],
    )


# ---------------------------------------------------------------------------
# KeyframeSpec / Keyframe validation
# ---------------------------------------------------------------------------


class TestSpecValidation:
    def test_requires_at_least_two_keyframes(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            KeyframeSpec(name="x", description="d", keyframes=[Keyframe(t=0.0)]).validate()

    def test_first_keyframe_must_be_zero(self) -> None:
        with pytest.raises(ValueError, match="first keyframe"):
            KeyframeSpec(
                name="x",
                description="d",
                keyframes=[Keyframe(t=0.1), Keyframe(t=1.0)],
            ).validate()

    def test_keyframes_must_be_strictly_increasing(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            KeyframeSpec(
                name="x",
                description="d",
                keyframes=[Keyframe(t=0.0), Keyframe(t=0.5), Keyframe(t=0.5)],
            ).validate()

    def test_positive_fps_required(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            KeyframeSpec(
                name="x",
                description="d",
                fps=0.0,
                keyframes=[Keyframe(t=0.0), Keyframe(t=1.0)],
            ).validate()

    def test_unknown_easing_rejected(self) -> None:
        with pytest.raises(ValueError, match="easing"):
            KeyframeSpec(
                name="x",
                description="d",
                easing="bogus",
                keyframes=[Keyframe(t=0.0), Keyframe(t=1.0)],
            ).validate()


# ---------------------------------------------------------------------------
# build_recorded_move — JSON shape and round trip
# ---------------------------------------------------------------------------


class TestBuildRecordedMoveShape:
    def test_top_level_keys(self) -> None:
        move = build_recorded_move(_simple_spec())
        assert set(move.keys()) == {"description", "time", "set_target_data"}
        assert move["description"] == "A gentle test nod."

    def test_time_and_data_aligned(self) -> None:
        move = build_recorded_move(_simple_spec())
        assert len(move["time"]) == len(move["set_target_data"])
        assert len(move["time"]) > 1

    def test_time_starts_at_zero_and_is_monotonic(self) -> None:
        move = build_recorded_move(_simple_spec())
        times = move["time"]
        assert times[0] == pytest.approx(0.0)
        assert all(b > a for a, b in zip(times, times[1:]))

    def test_frame_count_matches_fps_and_duration(self) -> None:
        # 1.0 s at 100 fps -> 101 frames (inclusive of endpoints).
        move = build_recorded_move(_simple_spec())
        assert len(move["time"]) == 101

    def test_each_frame_has_required_fields(self) -> None:
        move = build_recorded_move(_simple_spec())
        for frame in move["set_target_data"]:
            assert set(frame.keys()) >= {"head", "antennas", "body_yaw"}
            head = np.array(frame["head"], dtype=np.float64)
            assert head.shape == (4, 4)
            np.testing.assert_array_almost_equal(head[3, :], [0.0, 0.0, 0.0, 1.0])
            assert len(frame["antennas"]) == 2

    def test_head_matrices_are_valid_rotations(self) -> None:
        move = build_recorded_move(_simple_spec())
        for frame in move["set_target_data"]:
            rot = np.array(frame["head"], dtype=np.float64)[:3, :3]
            # Orthonormal: R R^T == I and det == 1
            np.testing.assert_array_almost_equal(rot @ rot.T, np.eye(3), decimal=5)
            assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-4)

    def test_json_serializable(self) -> None:
        import json

        move = build_recorded_move(_simple_spec())
        text = json.dumps(move)
        assert json.loads(text)["description"] == "A gentle test nod."


class TestKeyframeInterpolation:
    def test_keyframe_poses_hit_at_their_timestamps(self) -> None:
        move = build_recorded_move(_simple_spec())
        times = move["time"]
        data = move["set_target_data"]

        # The peak at t=0.5 should be ~15 deg pitch.
        idx_mid = min(range(len(times)), key=lambda i: abs(times[i] - 0.5))
        _, pitch_mid, _ = _rpy_of_head(data[idx_mid]["head"])
        assert pitch_mid == pytest.approx(15.0, abs=0.5)

        # Endpoints are neutral.
        _, pitch_start, _ = _rpy_of_head(data[0]["head"])
        _, pitch_end, _ = _rpy_of_head(data[-1]["head"])
        assert pitch_start == pytest.approx(0.0, abs=0.2)
        assert pitch_end == pytest.approx(0.0, abs=0.2)

    def test_translation_interpolated(self) -> None:
        spec = KeyframeSpec(
            name="lift",
            description="lift head",
            fps=100.0,
            keyframes=[
                Keyframe(t=0.0, z=0.0),
                Keyframe(t=1.0, z=0.01),
            ],
        )
        move = build_recorded_move(spec)
        z_start = move["set_target_data"][0]["head"][2][3]
        z_end = move["set_target_data"][-1]["head"][2][3]
        assert z_start == pytest.approx(0.0, abs=1e-6)
        assert z_end == pytest.approx(0.01, abs=1e-6)

    def test_antennas_and_body_yaw_interpolated(self) -> None:
        spec = KeyframeSpec(
            name="ant",
            description="antenna wiggle",
            fps=100.0,
            keyframes=[
                Keyframe(t=0.0, antennas=(0.0, 0.0), body_yaw=0.0),
                Keyframe(t=1.0, antennas=(0.5, -0.5), body_yaw=0.3),
            ],
        )
        move = build_recorded_move(spec)
        assert move["set_target_data"][0]["antennas"] == pytest.approx([0.0, 0.0])
        assert move["set_target_data"][-1]["antennas"] == pytest.approx([0.5, -0.5])
        assert move["set_target_data"][-1]["body_yaw"] == pytest.approx(0.3)

    def test_linear_easing_is_linear_at_midpoint(self) -> None:
        spec = KeyframeSpec(
            name="lin",
            description="linear yaw ramp",
            fps=100.0,
            easing="linear",
            keyframes=[Keyframe(t=0.0, yaw=0.0), Keyframe(t=1.0, yaw=20.0)],
        )
        move = build_recorded_move(spec)
        times = move["time"]
        idx_mid = min(range(len(times)), key=lambda i: abs(times[i] - 0.5))
        _, _, yaw_mid = _rpy_of_head(move["set_target_data"][idx_mid]["head"])
        assert yaw_mid == pytest.approx(10.0, abs=0.3)


class TestRecordedMoveRoundTrip:
    """The authored JSON must load and evaluate through reachy_mini's RecordedMove."""

    def test_loads_in_recorded_move(self) -> None:
        from reachy_mini.motion.recorded_move import RecordedMove

        move = build_recorded_move(_simple_spec())
        rm = RecordedMove(move)
        assert rm.duration > 0.0

    def test_evaluate_returns_pose(self) -> None:
        from reachy_mini.motion.recorded_move import RecordedMove

        move = build_recorded_move(_simple_spec())
        rm = RecordedMove(move)
        head, antennas, body_yaw = rm.evaluate(0.5)
        assert head.shape == (4, 4)
        assert antennas.shape == (2,)
        # Pitch near peak at the middle keyframe.
        _, pitch, _ = Rotation.from_matrix(head[:3, :3]).as_euler("xyz", degrees=True)
        assert pitch == pytest.approx(15.0, abs=1.0)
