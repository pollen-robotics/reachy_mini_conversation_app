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
