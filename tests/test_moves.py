"""Unit tests for the moves module."""

import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np
from numpy.typing import NDArray


# Mock reachy_mini before importing moves
@pytest.fixture(autouse=True)
def mock_reachy_mini():
    """Mock reachy_mini module before importing moves."""
    mock_reachy = MagicMock()
    mock_utils = MagicMock()
    mock_motion = MagicMock()
    mock_interpolation = MagicMock()

    # Create a realistic head pose (4x4 identity matrix)
    identity_pose = np.eye(4, dtype=np.float32)

    def mock_create_head_pose(
        x=0, y=0, z=0, roll=0, pitch=0, yaw=0, degrees=True, mm=True
    ) -> NDArray[np.float32]:
        """Mock create_head_pose that returns a 4x4 matrix."""
        pose = np.eye(4, dtype=np.float32)
        # Simple translation for x, y, z
        pose[0, 3] = float(x)
        pose[1, 3] = float(y)
        pose[2, 3] = float(z)
        return pose

    mock_utils.create_head_pose = mock_create_head_pose

    def mock_compose_world_offset(
        primary: NDArray, secondary: NDArray, reorthonormalize: bool = False
    ) -> NDArray[np.float32]:
        """Mock compose_world_offset."""
        # Simple addition for testing
        return (primary + secondary).astype(np.float32)

    def mock_linear_pose_interpolation(
        start: NDArray, end: NDArray, t: float
    ) -> NDArray[np.float32]:
        """Mock linear interpolation."""
        return ((1 - t) * start + t * end).astype(np.float32)

    mock_interpolation.compose_world_offset = mock_compose_world_offset
    mock_interpolation.linear_pose_interpolation = mock_linear_pose_interpolation

    # Mock Move base class
    class MockMove:
        duration: float = 1.0

        def evaluate(self, t: float):
            return (np.eye(4, dtype=np.float32), np.array([0.0, 0.0]), 0.0)

    mock_motion.move = MagicMock()
    mock_motion.move.Move = MockMove

    with patch.dict(
        "sys.modules",
        {
            "reachy_mini": mock_reachy,
            "reachy_mini.utils": mock_utils,
            "reachy_mini.motion": mock_motion,
            "reachy_mini.motion.move": mock_motion.move,
            "reachy_mini.utils.interpolation": mock_interpolation,
        },
    ):
        # Now set module attributes
        mock_reachy.ReachyMini = MagicMock
        mock_reachy.utils = mock_utils
        mock_reachy.utils.create_head_pose = mock_create_head_pose
        mock_reachy.utils.interpolation = mock_interpolation

        yield {
            "reachy_mini": mock_reachy,
            "utils": mock_utils,
            "motion": mock_motion,
            "interpolation": mock_interpolation,
            "create_head_pose": mock_create_head_pose,
            "compose_world_offset": mock_compose_world_offset,
            "linear_pose_interpolation": mock_linear_pose_interpolation,
            "Move": MockMove,
        }


class TestBreathingMove:
    """Tests for BreathingMove class."""

    def test_init_default_params(self, mock_reachy_mini: dict) -> None:
        """Test BreathingMove initializes with default parameters."""
        from reachy_mini_conversation_app.moves import BreathingMove

        start_pose = np.eye(4, dtype=np.float32)
        start_antennas = (0.1, -0.1)

        move = BreathingMove(
            interpolation_start_pose=start_pose,
            interpolation_start_antennas=start_antennas,
        )

        assert move.interpolation_duration == 1.0
        np.testing.assert_array_equal(move.interpolation_start_pose, start_pose)
        np.testing.assert_array_almost_equal(
            move.interpolation_start_antennas, [0.1, -0.1]
        )

    def test_init_custom_duration(self, mock_reachy_mini: dict) -> None:
        """Test BreathingMove with custom interpolation duration."""
        from reachy_mini_conversation_app.moves import BreathingMove

        move = BreathingMove(
            interpolation_start_pose=np.eye(4, dtype=np.float32),
            interpolation_start_antennas=(0.0, 0.0),
            interpolation_duration=2.5,
        )

        assert move.interpolation_duration == 2.5

    def test_duration_is_infinite(self, mock_reachy_mini: dict) -> None:
        """Test BreathingMove duration is infinite."""
        from reachy_mini_conversation_app.moves import BreathingMove

        move = BreathingMove(
            interpolation_start_pose=np.eye(4, dtype=np.float32),
            interpolation_start_antennas=(0.0, 0.0),
        )

        assert move.duration == float("inf")

    def test_evaluate_interpolation_phase(self, mock_reachy_mini: dict) -> None:
        """Test evaluate during interpolation phase (t < interpolation_duration)."""
        from reachy_mini_conversation_app.moves import BreathingMove

        start_pose = np.eye(4, dtype=np.float32)
        start_pose[0, 3] = 0.1  # x offset

        move = BreathingMove(
            interpolation_start_pose=start_pose,
            interpolation_start_antennas=(0.5, -0.5),
            interpolation_duration=1.0,
        )

        # At t=0, should be at start position
        head, antennas, body_yaw = move.evaluate(0.0)
        assert head is not None
        assert antennas is not None
        assert body_yaw == 0.0

        # At t=0.5, should be midway
        head, antennas, body_yaw = move.evaluate(0.5)
        assert head is not None
        assert antennas is not None

    def test_evaluate_breathing_phase(self, mock_reachy_mini: dict) -> None:
        """Test evaluate during breathing phase (t >= interpolation_duration)."""
        from reachy_mini_conversation_app.moves import BreathingMove

        move = BreathingMove(
            interpolation_start_pose=np.eye(4, dtype=np.float32),
            interpolation_start_antennas=(0.0, 0.0),
            interpolation_duration=1.0,
        )

        # After interpolation, breathing patterns should apply
        head, antennas, body_yaw = move.evaluate(2.0)
        assert head is not None
        assert antennas is not None
        assert len(antennas) == 2
        assert body_yaw == 0.0

    def test_breathing_z_oscillation(self, mock_reachy_mini: dict) -> None:
        """Test that breathing creates z-axis oscillation."""
        from reachy_mini_conversation_app.moves import BreathingMove

        move = BreathingMove(
            interpolation_start_pose=np.eye(4, dtype=np.float32),
            interpolation_start_antennas=(0.0, 0.0),
            interpolation_duration=0.0,  # Skip interpolation
        )

        # Sample at different times
        z_values = []
        for t in [0.0, 2.5, 5.0, 7.5, 10.0]:
            head, _, _ = move.evaluate(t)
            z_values.append(head[2, 3])  # z is in position [2,3] of 4x4 matrix

        # Z should oscillate (not all same value)
        assert len(set(z_values)) > 1

    def test_antenna_sway_opposite(self, mock_reachy_mini: dict) -> None:
        """Test that antennas sway in opposite directions."""
        from reachy_mini_conversation_app.moves import BreathingMove

        move = BreathingMove(
            interpolation_start_pose=np.eye(4, dtype=np.float32),
            interpolation_start_antennas=(0.0, 0.0),
            interpolation_duration=0.0,
        )

        # At a non-zero time point during breathing
        _, antennas, _ = move.evaluate(0.5)

        # Antennas should have opposite signs (or both zero)
        if antennas[0] != 0.0:
            assert antennas[0] * antennas[1] <= 0  # Opposite signs or zero


class TestCombineFullBody:
    """Tests for combine_full_body function."""

    def test_combine_neutral_poses(self, mock_reachy_mini: dict) -> None:
        """Test combining two neutral poses."""
        from reachy_mini_conversation_app.moves import combine_full_body

        primary = (np.eye(4, dtype=np.float32), (0.0, 0.0), 0.0)
        secondary = (np.zeros((4, 4), dtype=np.float32), (0.0, 0.0), 0.0)

        result = combine_full_body(primary, secondary)

        assert result[1] == (0.0, 0.0)  # antennas
        assert result[2] == 0.0  # body_yaw

    def test_combine_antennas_sum(self, mock_reachy_mini: dict) -> None:
        """Test that antennas are summed."""
        from reachy_mini_conversation_app.moves import combine_full_body

        primary = (np.eye(4, dtype=np.float32), (0.1, -0.1), 0.0)
        secondary = (np.zeros((4, 4), dtype=np.float32), (0.05, 0.05), 0.0)

        result = combine_full_body(primary, secondary)

        assert result[1] == pytest.approx((0.15, -0.05))

    def test_combine_body_yaw_sum(self, mock_reachy_mini: dict) -> None:
        """Test that body_yaw is summed."""
        from reachy_mini_conversation_app.moves import combine_full_body

        primary = (np.eye(4, dtype=np.float32), (0.0, 0.0), 0.5)
        secondary = (np.zeros((4, 4), dtype=np.float32), (0.0, 0.0), 0.3)

        result = combine_full_body(primary, secondary)

        assert result[2] == pytest.approx(0.8)


class TestCloneFullBodyPose:
    """Tests for clone_full_body_pose function."""

    def test_clone_creates_copy(self, mock_reachy_mini: dict) -> None:
        """Test that clone creates independent copy."""
        from reachy_mini_conversation_app.moves import clone_full_body_pose

        original_head = np.eye(4, dtype=np.float32)
        original = (original_head, (0.1, -0.1), 0.5)

        cloned = clone_full_body_pose(original)

        # Modify original
        original_head[0, 0] = 999.0

        # Clone should be unchanged
        assert cloned[0][0, 0] == 1.0

    def test_clone_preserves_values(self, mock_reachy_mini: dict) -> None:
        """Test that clone preserves all values."""
        from reachy_mini_conversation_app.moves import clone_full_body_pose

        head = np.eye(4, dtype=np.float32)
        head[0, 3] = 0.1
        original = (head, (0.2, -0.3), 0.7)

        cloned = clone_full_body_pose(original)

        np.testing.assert_array_equal(cloned[0], original[0])
        assert cloned[1] == (0.2, -0.3)
        assert cloned[2] == 0.7

    def test_clone_converts_to_float(self, mock_reachy_mini: dict) -> None:
        """Test that clone converts antenna values to float."""
        from reachy_mini_conversation_app.moves import clone_full_body_pose

        head = np.eye(4, dtype=np.float32)
        # Use numpy values that need conversion
        antennas = (np.float32(0.1), np.float64(-0.2))
        original = (head, antennas, np.float32(0.5))

        cloned = clone_full_body_pose(original)

        assert isinstance(cloned[1][0], float)
        assert isinstance(cloned[1][1], float)
        assert isinstance(cloned[2], float)


class TestMovementState:
    """Tests for MovementState dataclass."""

    def test_default_values(self, mock_reachy_mini: dict) -> None:
        """Test MovementState default values."""
        from reachy_mini_conversation_app.moves import MovementState

        state = MovementState()

        assert state.current_move is None
        assert state.move_start_time is None
        assert state.last_activity_time == 0.0
        assert state.speech_offsets == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert state.face_tracking_offsets == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert state.last_primary_pose is None

    def test_update_activity(self, mock_reachy_mini: dict) -> None:
        """Test update_activity updates timestamp."""
        from reachy_mini_conversation_app.moves import MovementState

        state = MovementState()
        old_time = state.last_activity_time

        time.sleep(0.01)
        state.update_activity()

        assert state.last_activity_time > old_time


class TestLoopFrequencyStats:
    """Tests for LoopFrequencyStats dataclass."""

    def test_default_values(self, mock_reachy_mini: dict) -> None:
        """Test LoopFrequencyStats default values."""
        from reachy_mini_conversation_app.moves import LoopFrequencyStats

        stats = LoopFrequencyStats()

        assert stats.mean == 0.0
        assert stats.m2 == 0.0
        assert stats.min_freq == float("inf")
        assert stats.count == 0
        assert stats.last_freq == 0.0
        assert stats.potential_freq == 0.0

    def test_reset(self, mock_reachy_mini: dict) -> None:
        """Test reset clears accumulator values."""
        from reachy_mini_conversation_app.moves import LoopFrequencyStats

        stats = LoopFrequencyStats(
            mean=50.0, m2=100.0, min_freq=45.0, count=100, last_freq=48.0
        )

        stats.reset()

        assert stats.mean == 0.0
        assert stats.m2 == 0.0
        assert stats.min_freq == float("inf")
        assert stats.count == 0
        # last_freq and potential_freq are not reset


class TestMovementManagerInit:
    """Tests for MovementManager initialization."""

    def test_init_with_robot(self, mock_reachy_mini: dict) -> None:
        """Test MovementManager initializes with robot."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        assert manager.current_robot is mock_robot
        assert manager.camera_worker is None
        assert manager.target_frequency == 100.0

    def test_init_with_camera_worker(self, mock_reachy_mini: dict) -> None:
        """Test MovementManager initializes with camera worker."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_camera = MagicMock()
        manager = MovementManager(current_robot=mock_robot, camera_worker=mock_camera)

        assert manager.camera_worker is mock_camera

    def test_init_creates_state(self, mock_reachy_mini: dict) -> None:
        """Test MovementManager creates initial state."""
        from reachy_mini_conversation_app.moves import MovementManager, MovementState

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        assert isinstance(manager.state, MovementState)
        assert manager.state.last_primary_pose is not None

    def test_init_creates_locks(self, mock_reachy_mini: dict) -> None:
        """Test MovementManager creates threading primitives."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        assert isinstance(manager._stop_event, threading.Event)
        assert isinstance(manager._speech_offsets_lock, type(threading.Lock()))
        assert isinstance(manager._face_offsets_lock, type(threading.Lock()))


class TestMovementManagerQueueMove:
    """Tests for MovementManager queue_move method."""

    def test_queue_move_adds_to_queue(self, mock_reachy_mini: dict) -> None:
        """Test queue_move adds command to queue."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        mock_move = MagicMock()
        mock_move.duration = 1.0
        manager.queue_move(mock_move)

        # Check command was queued
        assert not manager._command_queue.empty()
        cmd, payload = manager._command_queue.get_nowait()
        assert cmd == "queue_move"
        assert payload is mock_move


class TestMovementManagerClearQueue:
    """Tests for MovementManager clear_move_queue method."""

    def test_clear_queue_adds_command(self, mock_reachy_mini: dict) -> None:
        """Test clear_move_queue adds command to queue."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager.clear_move_queue()

        cmd, payload = manager._command_queue.get_nowait()
        assert cmd == "clear_queue"
        assert payload is None


class TestMovementManagerSetSpeechOffsets:
    """Tests for MovementManager set_speech_offsets method."""

    def test_set_speech_offsets_updates_pending(self, mock_reachy_mini: dict) -> None:
        """Test set_speech_offsets updates pending offsets."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        offsets = (0.01, 0.02, 0.03, 0.1, 0.2, 0.3)
        manager.set_speech_offsets(offsets)

        assert manager._pending_speech_offsets == offsets
        assert manager._speech_offsets_dirty is True


class TestMovementManagerSetListening:
    """Tests for MovementManager set_listening method."""

    def test_set_listening_queues_command(self, mock_reachy_mini: dict) -> None:
        """Test set_listening queues command when state changes."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager.set_listening(True)

        # Check command was queued
        assert not manager._command_queue.empty()
        cmd, payload = manager._command_queue.get_nowait()
        assert cmd == "set_listening"
        assert payload is True

    def test_set_listening_no_change_no_queue(self, mock_reachy_mini: dict) -> None:
        """Test set_listening does not queue when state unchanged."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)
        manager._shared_is_listening = False

        manager.set_listening(False)

        # Should not queue anything
        assert manager._command_queue.empty()


class TestMovementManagerIsIdle:
    """Tests for MovementManager is_idle method."""

    def test_is_idle_true_when_inactive(self, mock_reachy_mini: dict) -> None:
        """Test is_idle returns True when inactive long enough."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Set activity time far in the past
        manager._shared_last_activity_time = time.monotonic() - 10.0

        assert manager.is_idle() is True

    def test_is_idle_false_when_listening(self, mock_reachy_mini: dict) -> None:
        """Test is_idle returns False when listening."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager._shared_is_listening = True
        manager._shared_last_activity_time = time.monotonic() - 10.0

        assert manager.is_idle() is False

    def test_is_idle_false_when_recently_active(self, mock_reachy_mini: dict) -> None:
        """Test is_idle returns False when recently active."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager._shared_last_activity_time = time.monotonic()

        assert manager.is_idle() is False


class TestMovementManagerHandleCommand:
    """Tests for MovementManager _handle_command method."""

    def test_handle_queue_move(self, mock_reachy_mini: dict) -> None:
        """Test handling queue_move command."""
        from reachy_mini_conversation_app.moves import MovementManager
        from reachy_mini.motion.move import Move

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Create a proper Move subclass instance
        mock_move = MagicMock(spec=Move)
        mock_move.duration = 2.0

        manager._handle_command("queue_move", mock_move, time.monotonic())

        assert len(manager.move_queue) == 1
        assert manager.move_queue[0] is mock_move

    def test_handle_clear_queue(self, mock_reachy_mini: dict) -> None:
        """Test handling clear_queue command."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Add some moves
        manager.move_queue.append(MagicMock())
        manager.state.current_move = MagicMock()

        manager._handle_command("clear_queue", None, time.monotonic())

        assert len(manager.move_queue) == 0
        assert manager.state.current_move is None

    def test_handle_set_listening_true(self, mock_reachy_mini: dict) -> None:
        """Test handling set_listening True command."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)
        manager._is_listening = False

        current_time = time.monotonic()
        manager._last_listening_toggle_time = current_time - 1.0  # Past debounce

        manager._handle_command("set_listening", True, current_time)

        assert manager._is_listening is True
        assert manager._antenna_unfreeze_blend == 0.0

    def test_handle_unknown_command(self, mock_reachy_mini: dict) -> None:
        """Test handling unknown command logs warning."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Should not raise
        manager._handle_command("unknown_cmd", None, time.monotonic())


class TestMovementManagerGetPrimaryPose:
    """Tests for MovementManager _get_primary_pose method."""

    def test_get_primary_pose_no_move(self, mock_reachy_mini: dict) -> None:
        """Test _get_primary_pose returns last pose when no move."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        pose = manager._get_primary_pose(time.monotonic())

        assert pose is not None
        assert len(pose) == 3  # (head, antennas, body_yaw)

    def test_get_primary_pose_with_move(self, mock_reachy_mini: dict) -> None:
        """Test _get_primary_pose evaluates current move."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Set up a mock move
        mock_move = MagicMock()
        mock_move.evaluate.return_value = (
            np.eye(4, dtype=np.float32),
            np.array([0.1, -0.1]),
            0.5,
        )
        manager.state.current_move = mock_move
        manager.state.move_start_time = time.monotonic()

        pose = manager._get_primary_pose(time.monotonic())

        mock_move.evaluate.assert_called_once()
        assert pose[2] == 0.5  # body_yaw


class TestMovementManagerGetSecondaryPose:
    """Tests for MovementManager _get_secondary_pose method."""

    def test_get_secondary_pose_neutral(self, mock_reachy_mini: dict) -> None:
        """Test _get_secondary_pose returns neutral when no offsets."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        pose = manager._get_secondary_pose()

        assert pose[1] == (0.0, 0.0)  # antennas
        assert pose[2] == 0.0  # body_yaw

    def test_get_secondary_pose_combines_offsets(self, mock_reachy_mini: dict) -> None:
        """Test _get_secondary_pose combines speech and face offsets."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager.state.speech_offsets = (0.01, 0.02, 0.03, 0.1, 0.2, 0.3)
        manager.state.face_tracking_offsets = (0.001, 0.002, 0.003, 0.01, 0.02, 0.03)

        pose = manager._get_secondary_pose()

        # Should be combined into head pose
        assert pose is not None


class TestMovementManagerUpdateFrequencyStats:
    """Tests for MovementManager _update_frequency_stats method."""

    def test_update_frequency_stats(self, mock_reachy_mini: dict) -> None:
        """Test frequency stats are updated correctly."""
        from reachy_mini_conversation_app.moves import (
            MovementManager,
            LoopFrequencyStats,
        )

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        stats = LoopFrequencyStats()
        prev_time = time.monotonic()
        time.sleep(0.01)
        current_time = time.monotonic()

        updated_stats = manager._update_frequency_stats(current_time, prev_time, stats)

        assert updated_stats.count == 1
        assert updated_stats.last_freq > 0
        assert updated_stats.mean > 0


class TestMovementManagerCalculateBlendedAntennas:
    """Tests for MovementManager _calculate_blended_antennas method."""

    def test_blended_antennas_listening(self, mock_reachy_mini: dict) -> None:
        """Test antennas frozen when listening."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager._is_listening = True
        manager._listening_antennas = (0.5, -0.5)

        result = manager._calculate_blended_antennas((0.0, 0.0))

        assert result == (0.5, -0.5)

    def test_blended_antennas_not_listening(self, mock_reachy_mini: dict) -> None:
        """Test antennas blend when not listening."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager._is_listening = False
        manager._antenna_unfreeze_blend = 1.0  # Fully blended
        manager._listening_antennas = (0.5, -0.5)

        result = manager._calculate_blended_antennas((0.1, -0.1))

        # Should be target when fully blended
        assert result == pytest.approx((0.1, -0.1))


class TestMovementManagerGetStatus:
    """Tests for MovementManager get_status method."""

    def test_get_status_returns_dict(self, mock_reachy_mini: dict) -> None:
        """Test get_status returns status dictionary."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        status = manager.get_status()

        assert isinstance(status, dict)
        assert "queue_size" in status
        assert "is_listening" in status
        assert "breathing_active" in status
        assert "last_commanded_pose" in status
        assert "loop_frequency" in status

    def test_get_status_correct_values(self, mock_reachy_mini: dict) -> None:
        """Test get_status returns correct values."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager._is_listening = True
        manager._breathing_active = True

        status = manager.get_status()

        assert status["is_listening"] is True
        assert status["breathing_active"] is True


class TestMovementManagerStartStop:
    """Tests for MovementManager start/stop methods."""

    def test_start_creates_thread(self, mock_reachy_mini: dict) -> None:
        """Test start creates and starts worker thread."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager.start()
        try:
            assert manager._thread is not None
            assert manager._thread.is_alive()
        finally:
            manager._stop_event.set()
            manager._thread.join(timeout=1.0)

    def test_start_twice_ignored(self, mock_reachy_mini: dict) -> None:
        """Test starting twice is ignored."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager.start()
        first_thread = manager._thread

        manager.start()  # Should be ignored

        assert manager._thread is first_thread

        manager._stop_event.set()
        manager._thread.join(timeout=1.0)

    def test_stop_joins_thread(self, mock_reachy_mini: dict) -> None:
        """Test stop joins the worker thread."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_robot.goto_target = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager.start()
        thread = manager._thread

        manager.stop()

        assert manager._stop_event.is_set()
        assert thread is not None
        assert not thread.is_alive()

    def test_stop_resets_to_neutral(self, mock_reachy_mini: dict) -> None:
        """Test stop resets robot to neutral position."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager.start()
        manager.stop()

        mock_robot.goto_target.assert_called()


class TestMovementManagerWorkingLoop:
    """Tests for MovementManager working_loop method."""

    def test_working_loop_runs(self, mock_reachy_mini: dict) -> None:
        """Test working_loop executes and can be stopped."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Run for a short time
        manager.start()
        time.sleep(0.05)
        manager._stop_event.set()
        manager._thread.join(timeout=1.0)

        # Should have called set_target at least once
        assert mock_robot.set_target.called

    def test_working_loop_processes_commands(self, mock_reachy_mini: dict) -> None:
        """Test working_loop processes queued commands."""
        from reachy_mini_conversation_app.moves import MovementManager
        from reachy_mini.motion.move import Move

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Create a proper Move subclass instance
        mock_move = MagicMock(spec=Move)
        mock_move.duration = 0.1
        mock_move.evaluate.return_value = (np.eye(4, dtype=np.float32), np.array([0.0, 0.0]), 0.0)

        manager.start()
        manager.queue_move(mock_move)
        time.sleep(0.15)  # Give time for processing
        manager._stop_event.set()
        manager._thread.join(timeout=1.0)

        # Move should have been processed
        mock_move.evaluate.assert_called()


class TestConstants:
    """Tests for module constants."""

    def test_control_loop_frequency(self, mock_reachy_mini: dict) -> None:
        """Test CONTROL_LOOP_FREQUENCY_HZ constant."""
        from reachy_mini_conversation_app.moves import CONTROL_LOOP_FREQUENCY_HZ

        assert CONTROL_LOOP_FREQUENCY_HZ == 100.0


class TestMovementManagerSetMovingState:
    """Tests for MovementManager set_moving_state method."""

    def test_set_moving_state_queues_command(self, mock_reachy_mini: dict) -> None:
        """Test set_moving_state adds command to queue."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager.set_moving_state(2.5)

        cmd, payload = manager._command_queue.get_nowait()
        assert cmd == "set_moving_state"
        assert payload == 2.5


class TestMovementManagerApplyPendingOffsets:
    """Tests for MovementManager _apply_pending_offsets method."""

    def test_apply_speech_offsets_when_dirty(self, mock_reachy_mini: dict) -> None:
        """Test _apply_pending_offsets updates speech offsets when dirty."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Set pending offsets
        offsets = (0.1, 0.2, 0.3, 0.01, 0.02, 0.03)
        manager._pending_speech_offsets = offsets
        manager._speech_offsets_dirty = True

        manager._apply_pending_offsets()

        assert manager.state.speech_offsets == offsets
        assert manager._speech_offsets_dirty is False

    def test_apply_face_offsets_when_dirty(self, mock_reachy_mini: dict) -> None:
        """Test _apply_pending_offsets updates face offsets when dirty."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Set pending face offsets
        offsets = (0.05, 0.06, 0.07, 0.005, 0.006, 0.007)
        manager._pending_face_offsets = offsets
        manager._face_offsets_dirty = True

        manager._apply_pending_offsets()

        assert manager.state.face_tracking_offsets == offsets
        assert manager._face_offsets_dirty is False


class TestMovementManagerHandleCommandExtended:
    """Extended tests for MovementManager _handle_command method."""

    def test_handle_queue_move_invalid_payload(self, mock_reachy_mini: dict) -> None:
        """Test handling queue_move with invalid payload logs warning."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Pass something that's not a Move instance
        manager._handle_command("queue_move", "not_a_move", time.monotonic())

        # Queue should remain empty
        assert len(manager.move_queue) == 0

    def test_handle_queue_move_with_duration_conversion_error(self, mock_reachy_mini: dict) -> None:
        """Test handling queue_move when duration conversion fails."""
        from reachy_mini_conversation_app.moves import MovementManager
        from reachy_mini.motion.move import Move

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        mock_move = MagicMock(spec=Move)
        mock_move.duration = "invalid"  # Can't convert to float

        manager._handle_command("queue_move", mock_move, time.monotonic())

        # Move should still be queued
        assert len(manager.move_queue) == 1

    def test_handle_queue_move_without_duration(self, mock_reachy_mini: dict) -> None:
        """Test handling queue_move without duration attribute."""
        from reachy_mini_conversation_app.moves import MovementManager
        from reachy_mini.motion.move import Move

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        mock_move = MagicMock(spec=Move)
        del mock_move.duration  # Remove duration attribute

        manager._handle_command("queue_move", mock_move, time.monotonic())

        assert len(manager.move_queue) == 1

    def test_handle_set_moving_state_valid(self, mock_reachy_mini: dict) -> None:
        """Test handling set_moving_state with valid duration."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)
        initial_activity = manager.state.last_activity_time

        time.sleep(0.01)
        manager._handle_command("set_moving_state", 1.5, time.monotonic())

        # Activity should be updated
        assert manager.state.last_activity_time > initial_activity

    def test_handle_set_moving_state_invalid(self, mock_reachy_mini: dict) -> None:
        """Test handling set_moving_state with invalid duration."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)
        initial_activity = manager.state.last_activity_time

        manager._handle_command("set_moving_state", "invalid", time.monotonic())

        # Activity should not be updated
        assert manager.state.last_activity_time == initial_activity

    def test_handle_mark_activity(self, mock_reachy_mini: dict) -> None:
        """Test handling mark_activity command."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)
        initial_activity = manager.state.last_activity_time

        time.sleep(0.01)
        manager._handle_command("mark_activity", None, time.monotonic())

        assert manager.state.last_activity_time > initial_activity

    def test_handle_set_listening_debounce(self, mock_reachy_mini: dict) -> None:
        """Test set_listening is debounced."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        current_time = time.monotonic()
        # Set last toggle time to now (within debounce period)
        manager._last_listening_toggle_time = current_time

        manager._handle_command("set_listening", True, current_time)

        # Should still be False due to debounce
        assert manager._is_listening is False

    def test_handle_set_listening_same_state(self, mock_reachy_mini: dict) -> None:
        """Test set_listening does nothing when state unchanged."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        current_time = time.monotonic()
        manager._last_listening_toggle_time = current_time - 1.0  # Past debounce
        manager._is_listening = True  # Already listening

        # Try to set to True again
        manager._handle_command("set_listening", True, current_time)

        # Should remain True but nothing special happens
        assert manager._is_listening is True

    def test_handle_set_listening_unfreeze(self, mock_reachy_mini: dict) -> None:
        """Test set_listening unfreezing resets blend."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        current_time = time.monotonic()
        manager._last_listening_toggle_time = current_time - 1.0
        manager._is_listening = True  # Currently listening

        # Disable listening (unfreeze)
        manager._handle_command("set_listening", False, current_time)

        assert manager._is_listening is False
        assert manager._antenna_unfreeze_blend == 0.0


class TestMovementManagerBreathing:
    """Tests for MovementManager breathing functionality."""

    def test_breathing_starts_after_idle(self, mock_reachy_mini: dict) -> None:
        """Test breathing starts after idle delay."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_robot.get_current_joint_positions.return_value = (
            np.eye(4, dtype=np.float32),
            np.array([0.0, 0.0]),
        )
        mock_robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)

        manager = MovementManager(current_robot=mock_robot)
        manager.idle_inactivity_delay = 0.01  # Very short delay for testing

        # Set activity time far in the past
        manager.state.last_activity_time = time.monotonic() - 10.0

        manager._manage_breathing(time.monotonic())

        # Breathing should have been activated
        assert manager._breathing_active is True
        assert len(manager.move_queue) > 0

    def test_breathing_exception_handling(self, mock_reachy_mini: dict) -> None:
        """Test breathing handles exceptions gracefully."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_robot.get_current_joint_positions.side_effect = Exception("Robot error")

        manager = MovementManager(current_robot=mock_robot)
        manager.idle_inactivity_delay = 0.01

        # Set activity time far in the past
        manager.state.last_activity_time = time.monotonic() - 10.0

        # Should not raise
        manager._manage_breathing(time.monotonic())

        # Breathing should not be active due to error
        assert manager._breathing_active is False

    def test_breathing_stops_on_new_move(self, mock_reachy_mini: dict) -> None:
        """Test breathing stops when new move is queued."""
        from reachy_mini_conversation_app.moves import MovementManager, BreathingMove

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Set up breathing move as current
        breathing = BreathingMove(
            interpolation_start_pose=np.eye(4, dtype=np.float32),
            interpolation_start_antennas=(0.0, 0.0),
        )
        manager.state.current_move = breathing
        manager._breathing_active = True

        # Add a new move to queue
        mock_move = MagicMock()
        mock_move.duration = 1.0
        manager.move_queue.append(mock_move)

        manager._manage_breathing(time.monotonic())

        # Breathing should be stopped
        assert manager._breathing_active is False
        assert manager.state.current_move is None


class TestMovementManagerGetPrimaryPoseExtended:
    """Extended tests for _get_primary_pose."""

    def test_get_primary_pose_none_head(self, mock_reachy_mini: dict) -> None:
        """Test _get_primary_pose handles None head from move."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        mock_move = MagicMock()
        mock_move.evaluate.return_value = (None, np.array([0.0, 0.0]), 0.0)
        manager.state.current_move = mock_move
        manager.state.move_start_time = time.monotonic()

        pose = manager._get_primary_pose(time.monotonic())

        # Should have a valid pose despite None head
        assert pose is not None
        assert pose[0] is not None

    def test_get_primary_pose_none_antennas(self, mock_reachy_mini: dict) -> None:
        """Test _get_primary_pose handles None antennas from move."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        mock_move = MagicMock()
        mock_move.evaluate.return_value = (np.eye(4, dtype=np.float32), None, 0.0)
        manager.state.current_move = mock_move
        manager.state.move_start_time = time.monotonic()

        pose = manager._get_primary_pose(time.monotonic())

        # Should have default antennas
        assert pose[1] == (0.0, 0.0)

    def test_get_primary_pose_none_body_yaw(self, mock_reachy_mini: dict) -> None:
        """Test _get_primary_pose handles None body_yaw from move."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        mock_move = MagicMock()
        mock_move.evaluate.return_value = (np.eye(4, dtype=np.float32), np.array([0.0, 0.0]), None)
        manager.state.current_move = mock_move
        manager.state.move_start_time = time.monotonic()

        pose = manager._get_primary_pose(time.monotonic())

        # Should have default body_yaw
        assert pose[2] == 0.0

    def test_get_primary_pose_no_last_pose(self, mock_reachy_mini: dict) -> None:
        """Test _get_primary_pose creates neutral when no last pose."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Clear last primary pose
        manager.state.last_primary_pose = None

        pose = manager._get_primary_pose(time.monotonic())

        # Should create neutral pose
        assert pose is not None
        assert manager.state.last_primary_pose is not None


class TestMovementManagerBlendedAntennasExtended:
    """Extended tests for _calculate_blended_antennas."""

    def test_blended_antennas_zero_duration(self, mock_reachy_mini: dict) -> None:
        """Test antennas blending with zero duration."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        manager._is_listening = False
        manager._antenna_blend_duration = 0.0  # Zero duration
        manager._listening_antennas = (0.5, -0.5)
        manager._antenna_unfreeze_blend = 0.0

        result = manager._calculate_blended_antennas((0.1, -0.1))

        # Should immediately be at target
        assert result == pytest.approx((0.1, -0.1))


class TestMovementManagerIssueControlCommand:
    """Tests for _issue_control_command error handling."""

    def test_issue_control_command_error(self, mock_reachy_mini: dict) -> None:
        """Test _issue_control_command handles errors."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_robot.set_target.side_effect = Exception("Robot error")

        manager = MovementManager(current_robot=mock_robot)
        manager._last_set_target_err = 0  # Long time ago

        # Should not raise
        manager._issue_control_command(
            np.eye(4, dtype=np.float32), (0.0, 0.0), 0.0
        )

        # Error should be logged
        mock_robot.set_target.assert_called_once()

    def test_issue_control_command_error_suppression(self, mock_reachy_mini: dict) -> None:
        """Test _issue_control_command suppresses repeated errors."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_robot.set_target.side_effect = Exception("Robot error")

        manager = MovementManager(current_robot=mock_robot)
        # Set last error to very recent (within suppression interval)
        manager._last_set_target_err = time.monotonic()
        manager._set_target_err_suppressed = 0

        manager._issue_control_command(
            np.eye(4, dtype=np.float32), (0.0, 0.0), 0.0
        )

        # Error counter should be incremented
        assert manager._set_target_err_suppressed == 1

    def test_issue_control_command_error_suppression_logged(self, mock_reachy_mini: dict) -> None:
        """Test _issue_control_command logs suppressed error count."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_robot.set_target.side_effect = Exception("Robot error")

        manager = MovementManager(current_robot=mock_robot)
        # Set last error to past suppression interval
        manager._last_set_target_err = time.monotonic() - 10.0
        manager._set_target_err_suppressed = 5  # Some suppressed errors

        manager._issue_control_command(
            np.eye(4, dtype=np.float32), (0.0, 0.0), 0.0
        )

        # Suppressed count should be reset
        assert manager._set_target_err_suppressed == 0


class TestMovementManagerMaybeLogFrequency:
    """Tests for _maybe_log_frequency."""

    def test_maybe_log_frequency_logs_at_interval(self, mock_reachy_mini: dict) -> None:
        """Test _maybe_log_frequency logs at correct interval."""
        from reachy_mini_conversation_app.moves import MovementManager, LoopFrequencyStats

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        stats = LoopFrequencyStats(mean=50.0, m2=10.0, min_freq=45.0, count=100, last_freq=48.0)
        print_interval = 200

        # Should log at interval
        manager._maybe_log_frequency(200, print_interval, stats)

        # Stats should be reset
        assert stats.count == 0

    def test_maybe_log_frequency_skips_non_interval(self, mock_reachy_mini: dict) -> None:
        """Test _maybe_log_frequency skips non-interval loops."""
        from reachy_mini_conversation_app.moves import MovementManager, LoopFrequencyStats

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        stats = LoopFrequencyStats(mean=50.0, m2=10.0, min_freq=45.0, count=100, last_freq=48.0)
        print_interval = 200

        # Should not log
        manager._maybe_log_frequency(199, print_interval, stats)

        # Stats should not be reset
        assert stats.count == 100


class TestMovementManagerUpdateFaceTracking:
    """Tests for _update_face_tracking."""

    def test_update_face_tracking_with_camera_worker(self, mock_reachy_mini: dict) -> None:
        """Test _update_face_tracking with camera worker."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_camera = MagicMock()
        mock_camera.get_face_tracking_offsets.return_value = (0.1, 0.2, 0.3, 0.01, 0.02, 0.03)

        manager = MovementManager(current_robot=mock_robot, camera_worker=mock_camera)

        manager._update_face_tracking(time.monotonic())

        assert manager.state.face_tracking_offsets == (0.1, 0.2, 0.3, 0.01, 0.02, 0.03)


class TestMovementManagerStopExtended:
    """Extended tests for stop method."""

    def test_stop_when_not_running(self, mock_reachy_mini: dict) -> None:
        """Test stop does nothing when not running."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Stop without starting
        manager.stop()

        # goto_target should not be called
        mock_robot.goto_target.assert_not_called()

    def test_stop_goto_target_exception(self, mock_reachy_mini: dict) -> None:
        """Test stop handles goto_target exception."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        mock_robot.goto_target.side_effect = Exception("Robot error")

        manager = MovementManager(current_robot=mock_robot)

        manager.start()
        time.sleep(0.05)

        # Should not raise
        manager.stop()

        # Thread should be stopped despite error
        assert manager._thread is None or not manager._thread.is_alive()



class TestMovementManagerBranchCoverage:
    """Tests for edge case branch coverage."""

    def test_update_frequency_stats_zero_period(self, mock_reachy_mini: dict) -> None:
        """Test _update_frequency_stats with zero period (branch 659->666)."""
        from reachy_mini_conversation_app.moves import MovementManager, LoopFrequencyStats

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        stats = LoopFrequencyStats()
        initial_count = stats.count

        # Call with same timestamps (period=0)
        current_time = time.monotonic()
        result = manager._update_frequency_stats(current_time, current_time, stats)

        # Stats should not be updated
        assert result.count == initial_count

    def test_update_frequency_stats_negative_period(self, mock_reachy_mini: dict) -> None:
        """Test _update_frequency_stats with negative period (branch 659->666)."""
        from reachy_mini_conversation_app.moves import MovementManager, LoopFrequencyStats

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        stats = LoopFrequencyStats()
        initial_count = stats.count

        # Call with prev > current (negative period)
        current_time = time.monotonic()
        result = manager._update_frequency_stats(current_time, current_time + 1.0, stats)

        # Stats should not be updated
        assert result.count == initial_count

    def test_blended_antennas_partial_blend(self, mock_reachy_mini: dict) -> None:
        """Test _calculate_blended_antennas with partial blend (branch 627->633)."""
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Set up for partial blend (blend < 1.0 and not listening)
        manager._is_listening = False
        manager._antenna_blend_duration = 10.0  # Long duration
        manager._listening_antennas = (0.5, -0.5)
        manager._antenna_unfreeze_blend = 0.3  # Partial blend
        manager._antenna_blend_start = time.monotonic() - 1.0

        target = (0.1, -0.1)
        result = manager._calculate_blended_antennas(target)

        # Blend should be partially mixed
        # The new_blend will advance from 0.3 but not reach 1.0
        # So _listening_antennas should NOT be updated to target
        # (because new_blend < 1.0)
        assert manager._listening_antennas == (0.5, -0.5)

    def test_schedule_next_tick_zero_sleep(self, mock_reachy_mini: dict) -> None:
        """Test _schedule_next_tick when computation takes longer than target (branch 846->812)."""
        from reachy_mini_conversation_app.moves import MovementManager, LoopFrequencyStats

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Set a very short target period
        manager.target_period = 0.001  # 1ms

        stats = LoopFrequencyStats()

        # Mock _now to simulate long computation time
        # _schedule_next_tick calls self._now() once at line 670
        # computation_time = self._now() - loop_start
        # loop_start will be passed as 0.0, so we need _now() to return 1.0
        manager._now = lambda: 1.0  # Simulates computation_time = 1.0

        sleep_time, result_stats = manager._schedule_next_tick(0.0, stats)

        # Sleep time should be 0 (max(0.0, 0.001 - 1.0) = 0.0)
        assert sleep_time == 0.0

    def test_working_loop_zero_sleep_skips_sleep(self) -> None:
        """Test working_loop skips sleep when sleep_time <= 0 (branch 846->812)."""
        from reachy_mini_conversation_app.moves import MovementManager, LoopFrequencyStats
        import threading

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        loop_iterations = [0]

        def mock_schedule_returning_zero(loop_start: float, stats: LoopFrequencyStats) -> tuple:
            """Return 0 sleep time to skip sleep call."""
            loop_iterations[0] += 1
            # After a few iterations, stop the loop
            if loop_iterations[0] >= 3:
                manager._stop_event.set()
            return (0.0, stats)

        manager._schedule_next_tick = mock_schedule_returning_zero

        # Run the control loop in a thread
        manager._stop_event.clear()
        thread = threading.Thread(target=manager.working_loop, daemon=True)
        thread.start()

        # Wait for thread to finish
        thread.join(timeout=2.0)

        # Should have run at least 3 iterations
        assert loop_iterations[0] >= 3

    def test_stop_thread_becomes_none_between_check_and_join(self) -> None:
        """Test stop() when _thread becomes None between initial check and join (branch 741->744).

        This tests the defensive code path where _thread could theoretically become None
        after passing the initial is_alive() check but before the join() call.
        """
        from reachy_mini_conversation_app.moves import MovementManager

        mock_robot = MagicMock()
        manager = MovementManager(current_robot=mock_robot)

        # Create a mock thread that is "alive"
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        manager._thread = mock_thread

        # When clear_move_queue is called (line 737), set _thread to None
        # This simulates a race condition where thread becomes None between checks
        def clear_and_nullify():
            manager._thread = None

        manager.clear_move_queue = clear_and_nullify

        # stop() should handle this gracefully
        manager.stop()

        # Thread should be None (was set by our side effect)
        assert manager._thread is None
