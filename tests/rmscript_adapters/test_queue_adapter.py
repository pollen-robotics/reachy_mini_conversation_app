"""Tests for QueueExecutionAdapter."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from rmscript.ir import IRAction, IRWaitAction, IRPictureAction, IRPlaySoundAction

from reachy_mini_conversation_app.rmscript_adapters.queue_adapter import (
    QueueExecutionAdapter,
    QueueAdapterContext,
)
from reachy_mini_conversation_app.rmscript_adapters.queue_moves import (
    GotoQueueMove,
    SoundQueueMove,
    PictureQueueMove,
)


@pytest.fixture
def mock_robot():
    """Create mock robot with all required methods."""
    robot = Mock()
    robot.get_current_head_pose.return_value = np.eye(4, dtype=np.float32)
    robot.get_current_joint_positions.return_value = (
        [0.0, 0.0, 0.0],  # head joints (body_yaw, ...)
        [0.0, 0.0],  # antenna joints
    )
    return robot


@pytest.fixture
def mock_movement_manager():
    """Create mock movement manager."""
    manager = Mock()
    manager.queue_move = Mock()
    manager.set_moving_state = Mock()
    return manager


@pytest.fixture
def mock_camera_worker():
    """Create mock camera worker."""
    camera = Mock()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera.get_latest_frame.return_value = mock_frame
    return camera


@pytest.fixture
def adapter_context(mock_robot, mock_movement_manager, mock_camera_worker):
    """Create QueueAdapterContext with mocks."""
    return QueueAdapterContext(
        script_name="test_script",
        script_description="Test script",
        source_file_path=None,
        reachy_mini=mock_robot,
        movement_manager=mock_movement_manager,
        camera_worker=mock_camera_worker,
    )


class TestQueueExecutionAdapter:
    """Test QueueExecutionAdapter execution."""

    def test_execute_empty_ir(self, adapter_context, mock_movement_manager):
        """Test executing empty IR."""
        adapter = QueueExecutionAdapter()
        result = adapter.execute([], adapter_context)

        assert result["status"] == "Queued 0 moves from 'test_script'"
        assert result["total_duration"] == "0.0s"
        mock_movement_manager.queue_move.assert_not_called()
        mock_movement_manager.set_moving_state.assert_called_once_with(0.0)

    def test_execute_single_action(self, adapter_context, mock_movement_manager):
        """Test executing single IRAction."""
        adapter = QueueExecutionAdapter()

        # Create a simple head movement
        head_pose = np.eye(4, dtype=np.float64)
        head_pose[0, 3] = 1.0  # Move forward
        ir = [IRAction(head_pose=head_pose, duration=2.0)]

        result = adapter.execute(ir, adapter_context)

        assert result["status"] == "Queued 1 moves from 'test_script'"
        assert result["total_duration"] == "2.0s"
        mock_movement_manager.queue_move.assert_called_once()
        mock_movement_manager.set_moving_state.assert_called_once_with(2.0)

        # Check that queued move is GotoQueueMove
        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert isinstance(queued_move, GotoQueueMove)
        assert queued_move.duration == 2.0

    def test_execute_wait_action(self, adapter_context, mock_movement_manager):
        """Test executing IRWaitAction creates hold move."""
        adapter = QueueExecutionAdapter()
        ir = [IRWaitAction(duration=3.0)]

        result = adapter.execute(ir, adapter_context)

        assert result["total_duration"] == "3.0s"
        mock_movement_manager.queue_move.assert_called_once()

        # Check that queued move is GotoQueueMove with same start and target
        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert isinstance(queued_move, GotoQueueMove)
        assert queued_move.duration == 3.0
        # Hold move should have same start and target
        np.testing.assert_array_equal(queued_move.start_head_pose, queued_move.target_head_pose)

    def test_execute_multiple_actions(self, adapter_context, mock_movement_manager):
        """Test executing multiple actions in sequence."""
        adapter = QueueExecutionAdapter()

        head_pose1 = np.eye(4, dtype=np.float64)
        head_pose1[0, 3] = 1.0
        head_pose2 = np.eye(4, dtype=np.float64)
        head_pose2[0, 3] = 2.0

        ir = [
            IRAction(head_pose=head_pose1, duration=1.0),
            IRWaitAction(duration=0.5),
            IRAction(head_pose=head_pose2, duration=1.5),
        ]

        result = adapter.execute(ir, adapter_context)

        assert result["status"] == "Queued 3 moves from 'test_script'"
        assert result["total_duration"] == "3.0s"  # 1.0 + 0.5 + 1.5
        assert mock_movement_manager.queue_move.call_count == 3

    def test_execute_action_with_antennas(self, adapter_context, mock_movement_manager):
        """Test executing action with antenna control."""
        adapter = QueueExecutionAdapter()

        ir = [IRAction(antennas=[0.5, 0.5], duration=1.0)]

        result = adapter.execute(ir, adapter_context)

        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert isinstance(queued_move, GotoQueueMove)
        assert queued_move.target_antennas == (0.5, 0.5)

    def test_execute_action_with_body_yaw(self, adapter_context, mock_movement_manager):
        """Test executing action with body yaw."""
        adapter = QueueExecutionAdapter()

        ir = [IRAction(body_yaw=1.0, duration=1.0)]

        result = adapter.execute(ir, adapter_context)

        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert isinstance(queued_move, GotoQueueMove)
        assert queued_move.target_body_yaw == 1.0

    def test_execute_action_maintains_unspecified_state(self, adapter_context, mock_movement_manager):
        """Test that unspecified state is maintained from previous action."""
        adapter = QueueExecutionAdapter()

        # First action sets body_yaw
        # Second action doesn't specify body_yaw, should maintain it
        ir = [
            IRAction(body_yaw=1.5, duration=1.0),
            IRAction(head_pose=np.eye(4, dtype=np.float64), duration=1.0),
        ]

        result = adapter.execute(ir, adapter_context)

        # Get second queued move
        second_move = mock_movement_manager.queue_move.call_args_list[1][0][0]
        assert second_move.start_body_yaw == 1.5  # Should use previous target

    @patch("reachy_mini_conversation_app.rmscript_adapters.queue_adapter.find_sound_file")
    def test_execute_sound_action_async(self, mock_find_sound, adapter_context, mock_movement_manager):
        """Test executing async sound playback."""
        test_sound_path = Path(__file__).parent / "test_sound.wav"
        mock_find_sound.return_value = test_sound_path

        adapter = QueueExecutionAdapter()
        ir = [IRPlaySoundAction(sound_name="test_sound", blocking=False)]

        result = adapter.execute(ir, adapter_context)

        mock_movement_manager.queue_move.assert_called_once()
        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert isinstance(queued_move, SoundQueueMove)
        assert not queued_move.blocking
        assert queued_move.duration == pytest.approx(0.01, abs=0.001)  # Instant

    @patch("reachy_mini_conversation_app.rmscript_adapters.queue_adapter.get_sound_duration")
    @patch("reachy_mini_conversation_app.rmscript_adapters.queue_adapter.find_sound_file")
    def test_execute_sound_action_blocking(self, mock_find_sound, mock_get_duration, adapter_context, mock_movement_manager):
        """Test executing blocking sound playback."""
        test_sound_path = Path(__file__).parent / "test_sound.wav"
        mock_find_sound.return_value = test_sound_path
        mock_get_duration.return_value = 3.5

        adapter = QueueExecutionAdapter()
        ir = [IRPlaySoundAction(sound_name="test_sound", blocking=True)]

        result = adapter.execute(ir, adapter_context)

        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert isinstance(queued_move, SoundQueueMove)
        assert queued_move.blocking
        assert queued_move.duration == 3.5

    @patch("reachy_mini_conversation_app.rmscript_adapters.queue_adapter.find_sound_file")
    def test_execute_sound_action_with_duration(self, mock_find_sound, adapter_context, mock_movement_manager):
        """Test executing sound with explicit duration."""
        test_sound_path = Path(__file__).parent / "test_sound.wav"
        mock_find_sound.return_value = test_sound_path

        adapter = QueueExecutionAdapter()
        ir = [IRPlaySoundAction(sound_name="test_sound", blocking=True, duration=5.0)]

        result = adapter.execute(ir, adapter_context)

        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert queued_move.duration == 5.0

    @patch("reachy_mini_conversation_app.rmscript_adapters.queue_adapter.find_sound_file")
    def test_execute_sound_action_loop(self, mock_find_sound, adapter_context, mock_movement_manager):
        """Test executing looped sound."""
        test_sound_path = Path(__file__).parent / "test_sound.wav"
        mock_find_sound.return_value = test_sound_path

        adapter = QueueExecutionAdapter()
        ir = [IRPlaySoundAction(sound_name="test_sound", loop=True, duration=10.0)]

        result = adapter.execute(ir, adapter_context)

        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert isinstance(queued_move, SoundQueueMove)
        assert queued_move.loop
        assert queued_move.duration == 10.0

    @patch("reachy_mini_conversation_app.rmscript_adapters.sound_player.find_sound_file")
    def test_execute_sound_not_found_logs_warning(self, mock_find_sound, adapter_context, mock_movement_manager):
        """Test that missing sound file logs warning."""
        mock_find_sound.return_value = None  # Sound not found

        adapter = QueueExecutionAdapter()
        ir = [IRPlaySoundAction(sound_name="missing_sound")]

        result = adapter.execute(ir, adapter_context)

        # Should not queue a move for missing sound
        mock_movement_manager.queue_move.assert_not_called()

    @patch("cv2.imwrite")
    def test_execute_picture_action(self, mock_imwrite, adapter_context, mock_movement_manager, mock_camera_worker):
        """Test executing picture capture."""
        mock_imwrite.return_value = True

        adapter = QueueExecutionAdapter()
        ir = [IRPictureAction()]

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.return_value = b"fake_image"
            mock_open.return_value = mock_file

            with patch("time.sleep") as mock_sleep:
                # Manually trigger picture capture before sleep
                def trigger_capture(duration):
                    # Get the picture move and evaluate it
                    for call in mock_movement_manager.queue_move.call_args_list:
                        move = call[0][0]
                        if isinstance(move, PictureQueueMove):
                            move.evaluate(0.0)

                mock_sleep.side_effect = trigger_capture
                result = adapter.execute(ir, adapter_context)

        mock_movement_manager.queue_move.assert_called_once()
        queued_move = mock_movement_manager.queue_move.call_args[0][0]
        assert isinstance(queued_move, PictureQueueMove)

        # Should return base64 for single picture
        assert "b64_im" in result  # Single picture format

    @patch("cv2.imwrite")
    def test_execute_multiple_pictures(self, mock_imwrite, adapter_context, mock_movement_manager, mock_camera_worker):
        """Test executing multiple picture captures."""
        mock_imwrite.return_value = True

        adapter = QueueExecutionAdapter()
        ir = [IRPictureAction(), IRWaitAction(duration=1.0), IRPictureAction()]

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.return_value = b"fake_image"
            mock_open.return_value = mock_file

            with patch("time.sleep") as mock_sleep:
                # Manually trigger picture captures before sleep
                def trigger_captures(duration):
                    for call in mock_movement_manager.queue_move.call_args_list:
                        move = call[0][0]
                        if isinstance(move, PictureQueueMove):
                            move.evaluate(0.0)

                mock_sleep.side_effect = trigger_captures
                result = adapter.execute(ir, adapter_context)

        assert mock_movement_manager.queue_move.call_count == 3  # 2 pictures + 1 wait

        # Multiple pictures should use array format
        assert "pictures" in result
        assert result["picture_count"] == 2

    def test_sound_search_paths_includes_script_dir(self, mock_robot, mock_movement_manager, mock_camera_worker):
        """Test that sound search includes script directory."""
        script_path = Path("/some/path/to/script.rmscript")
        context = QueueAdapterContext(
            script_name="test",
            script_description="Test",
            source_file_path=str(script_path),
            reachy_mini=mock_robot,
            movement_manager=mock_movement_manager,
            camera_worker=mock_camera_worker,
        )

        adapter = QueueExecutionAdapter()
        ir = [IRPlaySoundAction(sound_name="test_sound")]

        with patch("reachy_mini_conversation_app.rmscript_adapters.queue_adapter.find_sound_file") as mock_find:
            mock_find.return_value = None  # Not found
            result = adapter.execute(ir, context)

            # Check that search_paths includes script directory
            mock_find.assert_called_once()
            search_paths = mock_find.call_args[0][1]
            assert script_path.parent in search_paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
