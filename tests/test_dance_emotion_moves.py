"""Unit tests for dance_emotion_moves module."""

from __future__ import annotations
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Store original modules before any mocking
_ORIGINAL_MODULES: dict[str, Any] = {}
_MODULES_TO_MOCK = [
    "reachy_mini",
    "reachy_mini.motion",
    "reachy_mini.motion.move",
    "reachy_mini.motion.recorded_move",
    "reachy_mini.utils",
    "reachy_mini.utils.interpolation",
    "reachy_mini_dances_library",
    "reachy_mini_dances_library.dance_move",
]


@pytest.fixture(autouse=True)
def mock_dance_emotion_dependencies() -> Any:
    """Mock heavy dependencies for dance_emotion_moves tests."""
    # Save originals
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in sys.modules:
            _ORIGINAL_MODULES[mod_name] = sys.modules[mod_name]

    # Create proper mock for Move base class that can be subclassed
    class MockMove:
        pass

    # Install mocks
    mock_reachy_mini = MagicMock()
    mock_motion = MagicMock()
    mock_motion_move = MagicMock()
    mock_motion_move.Move = MockMove
    mock_recorded_move = MagicMock()
    mock_utils = MagicMock()
    mock_utils.create_head_pose = MagicMock(return_value=np.eye(4, dtype=np.float32))
    mock_interpolation = MagicMock()
    mock_interpolation.linear_pose_interpolation = MagicMock(
        return_value=np.eye(4, dtype=np.float64)
    )
    mock_dances_lib = MagicMock()
    mock_dance_move_mod = MagicMock()

    sys.modules["reachy_mini"] = mock_reachy_mini
    sys.modules["reachy_mini.motion"] = mock_motion
    sys.modules["reachy_mini.motion.move"] = mock_motion_move
    sys.modules["reachy_mini.motion.recorded_move"] = mock_recorded_move
    sys.modules["reachy_mini.utils"] = mock_utils
    sys.modules["reachy_mini.utils.interpolation"] = mock_interpolation
    sys.modules["reachy_mini_dances_library"] = mock_dances_lib
    sys.modules["reachy_mini_dances_library.dance_move"] = mock_dance_move_mod

    yield {
        "MockMove": MockMove,
        "mock_utils": mock_utils,
        "mock_interpolation": mock_interpolation,
        "mock_dance_move_mod": mock_dance_move_mod,
    }

    # Restore originals
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in _ORIGINAL_MODULES:
            sys.modules[mod_name] = _ORIGINAL_MODULES[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]

    # Clear cached module imports
    mods_to_clear = [
        k for k in sys.modules if k.startswith("reachy_mini_conversation_app.dance_emotion")
    ]
    for mod_name in mods_to_clear:
        del sys.modules[mod_name]


class TestDanceQueueMoveInit:
    """Tests for DanceQueueMove initialization."""

    def test_init_creates_dance_move(self) -> None:
        """Test that init creates a DanceMove with the given name."""
        mock_dance_move = MagicMock()
        mock_dance_move.duration = 3.0

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini_dances_library.dance_move": MagicMock(
                    DanceMove=MagicMock(return_value=mock_dance_move)
                ),
            },
        ):
            # Force reimport
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

            move = DanceQueueMove("test_dance")

            assert move.move_name == "test_dance"
            assert move.dance_move is mock_dance_move

    def test_init_stores_move_name(self) -> None:
        """Test that init stores the move name."""
        mock_dance_move = MagicMock()
        mock_dance_move.duration = 2.5

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini_dances_library.dance_move": MagicMock(
                    DanceMove=MagicMock(return_value=mock_dance_move)
                ),
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

            move = DanceQueueMove("happy_dance")

            assert move.move_name == "happy_dance"


class TestDanceQueueMoveDuration:
    """Tests for DanceQueueMove duration property."""

    def test_duration_returns_dance_move_duration(self) -> None:
        """Test that duration property returns the dance move's duration."""
        mock_dance_move = MagicMock()
        mock_dance_move.duration = 4.5

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini_dances_library.dance_move": MagicMock(
                    DanceMove=MagicMock(return_value=mock_dance_move)
                ),
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

            move = DanceQueueMove("test_dance")

            assert move.duration == 4.5

    def test_duration_converts_to_float(self) -> None:
        """Test that duration is converted to float."""
        mock_dance_move = MagicMock()
        mock_dance_move.duration = 3  # Integer

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini_dances_library.dance_move": MagicMock(
                    DanceMove=MagicMock(return_value=mock_dance_move)
                ),
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

            move = DanceQueueMove("test_dance")

            assert isinstance(move.duration, float)
            assert move.duration == 3.0


class TestDanceQueueMoveEvaluate:
    """Tests for DanceQueueMove evaluate method."""

    def test_evaluate_calls_dance_move_evaluate(self) -> None:
        """Test that evaluate calls the underlying dance move's evaluate."""
        mock_dance_move = MagicMock()
        mock_dance_move.duration = 2.0
        head_pose = np.eye(4, dtype=np.float64)
        antennas = np.array([0.1, 0.2])
        body_yaw = 0.5
        mock_dance_move.evaluate.return_value = (head_pose, antennas, body_yaw)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini_dances_library.dance_move": MagicMock(
                    DanceMove=MagicMock(return_value=mock_dance_move)
                ),
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

            move = DanceQueueMove("test_dance")
            result = move.evaluate(0.5)

            mock_dance_move.evaluate.assert_called_once_with(0.5)
            assert result[0] is head_pose
            np.testing.assert_array_equal(result[1], antennas)
            assert result[2] == body_yaw

    def test_evaluate_converts_tuple_antennas_to_array(self) -> None:
        """Test that evaluate converts tuple antennas to numpy array."""
        mock_dance_move = MagicMock()
        mock_dance_move.duration = 2.0
        head_pose = np.eye(4, dtype=np.float64)
        antennas_tuple = (0.3, 0.4)  # Tuple instead of array
        body_yaw = 0.0
        mock_dance_move.evaluate.return_value = (head_pose, antennas_tuple, body_yaw)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini_dances_library.dance_move": MagicMock(
                    DanceMove=MagicMock(return_value=mock_dance_move)
                ),
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

            move = DanceQueueMove("test_dance")
            result = move.evaluate(1.0)

            assert isinstance(result[1], np.ndarray)
            np.testing.assert_array_equal(result[1], np.array([0.3, 0.4]))

    def test_evaluate_returns_neutral_on_error(self) -> None:
        """Test that evaluate returns neutral pose on error."""
        mock_dance_move = MagicMock()
        mock_dance_move.duration = 2.0
        mock_dance_move.evaluate.side_effect = RuntimeError("Dance error")

        neutral_pose = np.eye(4, dtype=np.float64)
        mock_create_head_pose = MagicMock(return_value=neutral_pose)
        mock_utils = MagicMock()
        mock_utils.create_head_pose = mock_create_head_pose

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini_dances_library.dance_move": MagicMock(
                    DanceMove=MagicMock(return_value=mock_dance_move)
                ),
                "reachy_mini.utils": mock_utils,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

            move = DanceQueueMove("test_dance")
            result = move.evaluate(0.5)

            # Use array equality (not identity) since .astype() creates a copy
            np.testing.assert_array_equal(result[0], neutral_pose)
            np.testing.assert_array_equal(result[1], np.array([0.0, 0.0]))
            assert result[2] == 0.0


class TestEmotionQueueMoveInit:
    """Tests for EmotionQueueMove initialization."""

    def test_init_gets_emotion_from_recorded_moves(self) -> None:
        """Test that init gets emotion move from RecordedMoves."""
        from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

        mock_emotion_move = MagicMock()
        mock_emotion_move.duration = 1.5
        mock_recorded_moves = MagicMock()
        mock_recorded_moves.get.return_value = mock_emotion_move

        move = EmotionQueueMove("happy", mock_recorded_moves)

        mock_recorded_moves.get.assert_called_once_with("happy")
        assert move.emotion_name == "happy"
        assert move.emotion_move is mock_emotion_move

    def test_init_stores_emotion_name(self) -> None:
        """Test that init stores the emotion name."""
        from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

        mock_emotion_move = MagicMock()
        mock_emotion_move.duration = 2.0
        mock_recorded_moves = MagicMock()
        mock_recorded_moves.get.return_value = mock_emotion_move

        move = EmotionQueueMove("sad", mock_recorded_moves)

        assert move.emotion_name == "sad"


class TestEmotionQueueMoveDuration:
    """Tests for EmotionQueueMove duration property."""

    def test_duration_returns_emotion_move_duration(self) -> None:
        """Test that duration property returns the emotion move's duration."""
        from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

        mock_emotion_move = MagicMock()
        mock_emotion_move.duration = 3.0
        mock_recorded_moves = MagicMock()
        mock_recorded_moves.get.return_value = mock_emotion_move

        move = EmotionQueueMove("happy", mock_recorded_moves)

        assert move.duration == 3.0

    def test_duration_converts_to_float(self) -> None:
        """Test that duration is converted to float."""
        from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

        mock_emotion_move = MagicMock()
        mock_emotion_move.duration = 2  # Integer
        mock_recorded_moves = MagicMock()
        mock_recorded_moves.get.return_value = mock_emotion_move

        move = EmotionQueueMove("happy", mock_recorded_moves)

        assert isinstance(move.duration, float)
        assert move.duration == 2.0


class TestEmotionQueueMoveEvaluate:
    """Tests for EmotionQueueMove evaluate method."""

    def test_evaluate_calls_emotion_move_evaluate(self) -> None:
        """Test that evaluate calls the underlying emotion move's evaluate."""
        from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

        mock_emotion_move = MagicMock()
        mock_emotion_move.duration = 1.5
        head_pose = np.eye(4, dtype=np.float64)
        antennas = np.array([0.2, 0.3])
        body_yaw = 0.1
        mock_emotion_move.evaluate.return_value = (head_pose, antennas, body_yaw)
        mock_recorded_moves = MagicMock()
        mock_recorded_moves.get.return_value = mock_emotion_move

        move = EmotionQueueMove("happy", mock_recorded_moves)
        result = move.evaluate(0.75)

        mock_emotion_move.evaluate.assert_called_once_with(0.75)
        assert result[0] is head_pose
        np.testing.assert_array_equal(result[1], antennas)
        assert result[2] == body_yaw

    def test_evaluate_converts_tuple_antennas_to_array(self) -> None:
        """Test that evaluate converts tuple antennas to numpy array."""
        from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

        mock_emotion_move = MagicMock()
        mock_emotion_move.duration = 1.5
        head_pose = np.eye(4, dtype=np.float64)
        antennas_tuple = (0.5, 0.6)  # Tuple instead of array
        body_yaw = 0.0
        mock_emotion_move.evaluate.return_value = (head_pose, antennas_tuple, body_yaw)
        mock_recorded_moves = MagicMock()
        mock_recorded_moves.get.return_value = mock_emotion_move

        move = EmotionQueueMove("happy", mock_recorded_moves)
        result = move.evaluate(1.0)

        assert isinstance(result[1], np.ndarray)
        np.testing.assert_array_equal(result[1], np.array([0.5, 0.6]))

    def test_evaluate_returns_neutral_on_error(self) -> None:
        """Test that evaluate returns neutral pose on error."""
        mock_emotion_move = MagicMock()
        mock_emotion_move.duration = 1.5
        mock_emotion_move.evaluate.side_effect = RuntimeError("Emotion error")
        mock_recorded_moves = MagicMock()
        mock_recorded_moves.get.return_value = mock_emotion_move

        neutral_pose = np.eye(4, dtype=np.float64)
        mock_create_head_pose = MagicMock(return_value=neutral_pose)
        mock_utils = MagicMock()
        mock_utils.create_head_pose = mock_create_head_pose

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

            move = EmotionQueueMove("happy", mock_recorded_moves)
            result = move.evaluate(0.5)

            # Use array equality (not identity) since .astype() creates a copy
            np.testing.assert_array_equal(result[0], neutral_pose)
            np.testing.assert_array_equal(result[1], np.array([0.0, 0.0]))
            assert result[2] == 0.0


class TestGotoQueueMoveInit:
    """Tests for GotoQueueMove initialization."""

    def test_init_with_required_params(self) -> None:
        """Test initialization with only required parameters."""
        from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

        target_pose = np.eye(4, dtype=np.float32)

        move = GotoQueueMove(target_head_pose=target_pose)

        np.testing.assert_array_equal(move.target_head_pose, target_pose)
        assert move.start_head_pose is None
        assert move.target_antennas == (0, 0)
        assert move.start_antennas == (0, 0)
        assert move.target_body_yaw == 0
        assert move.start_body_yaw == 0
        assert move._duration == 1.0

    def test_init_with_all_params(self) -> None:
        """Test initialization with all parameters."""
        from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

        target_pose = np.eye(4, dtype=np.float32)
        start_pose = np.eye(4, dtype=np.float32) * 0.5

        move = GotoQueueMove(
            target_head_pose=target_pose,
            start_head_pose=start_pose,
            target_antennas=(0.5, 0.6),
            start_antennas=(0.1, 0.2),
            target_body_yaw=0.3,
            start_body_yaw=0.1,
            duration=2.5,
        )

        np.testing.assert_array_equal(move.target_head_pose, target_pose)
        np.testing.assert_array_equal(move.start_head_pose, start_pose)
        assert move.target_antennas == (0.5, 0.6)
        assert move.start_antennas == (0.1, 0.2)
        assert move.target_body_yaw == 0.3
        assert move.start_body_yaw == 0.1
        assert move._duration == 2.5

    def test_init_defaults_none_start_antennas(self) -> None:
        """Test that None start_antennas defaults to (0, 0)."""
        from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

        target_pose = np.eye(4, dtype=np.float32)

        move = GotoQueueMove(target_head_pose=target_pose, start_antennas=None)

        assert move.start_antennas == (0, 0)

    def test_init_defaults_none_start_body_yaw(self) -> None:
        """Test that None start_body_yaw defaults to 0."""
        from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

        target_pose = np.eye(4, dtype=np.float32)

        move = GotoQueueMove(target_head_pose=target_pose, start_body_yaw=None)

        assert move.start_body_yaw == 0


class TestGotoQueueMoveDuration:
    """Tests for GotoQueueMove duration property."""

    def test_duration_returns_stored_duration(self) -> None:
        """Test that duration property returns the stored duration."""
        from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

        target_pose = np.eye(4, dtype=np.float32)

        move = GotoQueueMove(target_head_pose=target_pose, duration=3.5)

        assert move.duration == 3.5

    def test_duration_default_is_one(self) -> None:
        """Test that default duration is 1.0."""
        from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

        target_pose = np.eye(4, dtype=np.float32)

        move = GotoQueueMove(target_head_pose=target_pose)

        assert move.duration == 1.0


class TestGotoQueueMoveEvaluate:
    """Tests for GotoQueueMove evaluate method."""

    def test_evaluate_at_start(self) -> None:
        """Test evaluate at t=0 returns start pose."""
        target_pose = np.eye(4, dtype=np.float32)
        start_pose = np.eye(4, dtype=np.float32) * 0.5
        interpolated_pose = start_pose.copy()

        mock_utils = MagicMock()
        mock_utils.create_head_pose = MagicMock(return_value=start_pose)
        mock_interpolation = MagicMock()
        mock_interpolation.linear_pose_interpolation = MagicMock(return_value=interpolated_pose)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
                "reachy_mini.utils.interpolation": mock_interpolation,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

            move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=start_pose,
                target_antennas=(1.0, 1.0),
                start_antennas=(0.0, 0.0),
                target_body_yaw=1.0,
                start_body_yaw=0.0,
                duration=2.0,
            )
            result = move.evaluate(0.0)

            # At t=0, t_clamped = 0/2 = 0
            mock_interpolation.linear_pose_interpolation.assert_called_once()
            args = mock_interpolation.linear_pose_interpolation.call_args[0]
            np.testing.assert_array_equal(args[0], start_pose)
            np.testing.assert_array_equal(args[1], target_pose)
            assert args[2] == 0.0  # t_clamped

            # Antennas at t=0 should be start values
            assert result[1] is not None
            np.testing.assert_array_almost_equal(result[1], np.array([0.0, 0.0]))
            assert result[2] == 0.0  # body_yaw at t=0

    def test_evaluate_at_end(self) -> None:
        """Test evaluate at t=duration returns target pose."""
        target_pose = np.eye(4, dtype=np.float32)
        start_pose = np.eye(4, dtype=np.float32) * 0.5

        mock_utils = MagicMock()
        mock_utils.create_head_pose = MagicMock(return_value=start_pose)
        mock_interpolation = MagicMock()
        mock_interpolation.linear_pose_interpolation = MagicMock(return_value=target_pose)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
                "reachy_mini.utils.interpolation": mock_interpolation,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

            move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=start_pose,
                target_antennas=(1.0, 1.0),
                start_antennas=(0.0, 0.0),
                target_body_yaw=1.0,
                start_body_yaw=0.0,
                duration=2.0,
            )
            result = move.evaluate(2.0)

            # At t=duration, t_clamped = 2/2 = 1
            mock_interpolation.linear_pose_interpolation.assert_called_once()
            args = mock_interpolation.linear_pose_interpolation.call_args[0]
            assert args[2] == 1.0  # t_clamped

            # Antennas at t=duration should be target values
            assert result[1] is not None
            np.testing.assert_array_almost_equal(result[1], np.array([1.0, 1.0]))
            assert result[2] == 1.0  # body_yaw at t=duration

    def test_evaluate_at_midpoint(self) -> None:
        """Test evaluate at t=duration/2 returns midpoint values."""
        target_pose = np.eye(4, dtype=np.float32)
        start_pose = np.eye(4, dtype=np.float32) * 0.5
        mid_pose = np.eye(4, dtype=np.float32) * 0.75

        mock_utils = MagicMock()
        mock_utils.create_head_pose = MagicMock(return_value=start_pose)
        mock_interpolation = MagicMock()
        mock_interpolation.linear_pose_interpolation = MagicMock(return_value=mid_pose)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
                "reachy_mini.utils.interpolation": mock_interpolation,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

            move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=start_pose,
                target_antennas=(1.0, 1.0),
                start_antennas=(0.0, 0.0),
                target_body_yaw=1.0,
                start_body_yaw=0.0,
                duration=2.0,
            )
            result = move.evaluate(1.0)

            # At t=1, t_clamped = 1/2 = 0.5
            mock_interpolation.linear_pose_interpolation.assert_called_once()
            args = mock_interpolation.linear_pose_interpolation.call_args[0]
            assert args[2] == 0.5  # t_clamped

            # Antennas at midpoint
            assert result[1] is not None
            np.testing.assert_array_almost_equal(result[1], np.array([0.5, 0.5]))
            assert result[2] == 0.5  # body_yaw at midpoint

    def test_evaluate_clamps_t_to_zero(self) -> None:
        """Test that negative t is clamped to 0."""
        target_pose = np.eye(4, dtype=np.float32)
        neutral_pose = np.eye(4, dtype=np.float32)

        mock_utils = MagicMock()
        mock_utils.create_head_pose = MagicMock(return_value=neutral_pose)
        mock_interpolation = MagicMock()
        mock_interpolation.linear_pose_interpolation = MagicMock(return_value=neutral_pose)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
                "reachy_mini.utils.interpolation": mock_interpolation,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

            move = GotoQueueMove(target_head_pose=target_pose, duration=1.0)
            move.evaluate(-0.5)

            # t_clamped should be 0 (clamped from negative)
            args = mock_interpolation.linear_pose_interpolation.call_args[0]
            assert args[2] == 0.0

    def test_evaluate_clamps_t_to_one(self) -> None:
        """Test that t > duration is clamped to 1."""
        target_pose = np.eye(4, dtype=np.float32)
        neutral_pose = np.eye(4, dtype=np.float32)

        mock_utils = MagicMock()
        mock_utils.create_head_pose = MagicMock(return_value=neutral_pose)
        mock_interpolation = MagicMock()
        mock_interpolation.linear_pose_interpolation = MagicMock(return_value=neutral_pose)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
                "reachy_mini.utils.interpolation": mock_interpolation,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

            move = GotoQueueMove(target_head_pose=target_pose, duration=1.0)
            move.evaluate(2.0)

            # t_clamped should be 1 (clamped from 2/1 = 2)
            args = mock_interpolation.linear_pose_interpolation.call_args[0]
            assert args[2] == 1.0

    def test_evaluate_uses_neutral_when_no_start_pose(self) -> None:
        """Test that evaluate uses neutral pose when start_head_pose is None."""
        target_pose = np.eye(4, dtype=np.float32)
        neutral_pose = np.eye(4, dtype=np.float32)

        mock_utils = MagicMock()
        mock_utils.create_head_pose = MagicMock(return_value=neutral_pose)
        mock_interpolation = MagicMock()
        mock_interpolation.linear_pose_interpolation = MagicMock(return_value=neutral_pose)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
                "reachy_mini.utils.interpolation": mock_interpolation,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

            move = GotoQueueMove(target_head_pose=target_pose)  # No start_head_pose
            move.evaluate(0.5)

            mock_utils.create_head_pose.assert_called_once_with(0, 0, 0, 0, 0, 0, degrees=True)
            args = mock_interpolation.linear_pose_interpolation.call_args[0]
            np.testing.assert_array_equal(args[0], neutral_pose)

    def test_evaluate_returns_target_on_error(self) -> None:
        """Test that evaluate returns target pose on error."""
        target_pose = np.eye(4, dtype=np.float32)

        mock_utils = MagicMock()
        mock_utils.create_head_pose = MagicMock(side_effect=RuntimeError("Pose error"))
        mock_interpolation = MagicMock()
        mock_interpolation.linear_pose_interpolation = MagicMock(
            side_effect=RuntimeError("Interpolation error")
        )

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
                "reachy_mini.utils.interpolation": mock_interpolation,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

            move = GotoQueueMove(
                target_head_pose=target_pose,
                target_antennas=(0.5, 0.6),
                target_body_yaw=0.3,
            )
            result = move.evaluate(0.5)

            # Should return target values on error
            assert result[0] is not None
            assert result[0].dtype == np.float64
            assert result[1] is not None
            np.testing.assert_array_almost_equal(
                result[1], np.array([0.5, 0.6], dtype=np.float64)
            )
            assert result[2] == 0.3

    def test_evaluate_returns_correct_types(self) -> None:
        """Test that evaluate returns correct numpy types."""
        target_pose = np.eye(4, dtype=np.float32)
        interpolated_pose = np.eye(4, dtype=np.float64)

        mock_utils = MagicMock()
        mock_utils.create_head_pose = MagicMock(return_value=np.eye(4, dtype=np.float32))
        mock_interpolation = MagicMock()
        mock_interpolation.linear_pose_interpolation = MagicMock(return_value=interpolated_pose)

        with patch.dict(
            "sys.modules",
            {
                "reachy_mini.utils": mock_utils,
                "reachy_mini.utils.interpolation": mock_interpolation,
            },
        ):
            if "reachy_mini_conversation_app.dance_emotion_moves" in sys.modules:
                del sys.modules["reachy_mini_conversation_app.dance_emotion_moves"]

            from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

            move = GotoQueueMove(target_head_pose=target_pose)
            result = move.evaluate(0.5)

            # Check return types
            assert isinstance(result[1], np.ndarray)
            assert result[1].dtype == np.float64
            assert isinstance(result[2], float)
