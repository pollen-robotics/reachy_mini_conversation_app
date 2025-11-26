"""Tests for rmscript queue move implementations."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.rmscript_adapters.queue_moves import (
    GotoQueueMove,
    SoundQueueMove,
    PictureQueueMove,
)


class TestGotoQueueMove:
    """Test GotoQueueMove interpolation."""

    def test_duration_property(self):
        """Test that duration property returns correct value."""
        target_pose = np.eye(4, dtype=np.float32)
        move = GotoQueueMove(target_pose, duration=2.5)
        assert move.duration == 2.5

    def test_evaluate_at_start(self):
        """Test evaluation at t=0 returns start pose."""
        start_pose = np.eye(4, dtype=np.float32)
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[0, 3] = 1.0  # Move forward 1m

        move = GotoQueueMove(
            target_head_pose=target_pose,
            start_head_pose=start_pose,
            target_antennas=(0.5, 0.5),
            start_antennas=(0.0, 0.0),
            target_body_yaw=1.0,
            start_body_yaw=0.0,
            duration=2.0,
        )

        head, antennas, body_yaw = move.evaluate(0.0)

        # At t=0, should be at start pose
        assert head is not None
        np.testing.assert_array_almost_equal(head[:3, 3], start_pose[:3, 3])
        assert antennas[0] == pytest.approx(0.0, abs=0.01)
        assert antennas[1] == pytest.approx(0.0, abs=0.01)
        assert body_yaw == pytest.approx(0.0, abs=0.01)

    def test_evaluate_at_end(self):
        """Test evaluation at t=duration returns target pose."""
        start_pose = np.eye(4, dtype=np.float32)
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[0, 3] = 1.0  # Move forward 1m

        move = GotoQueueMove(
            target_head_pose=target_pose,
            start_head_pose=start_pose,
            target_antennas=(0.5, 0.5),
            start_antennas=(0.0, 0.0),
            target_body_yaw=1.0,
            start_body_yaw=0.0,
            duration=2.0,
        )

        head, antennas, body_yaw = move.evaluate(2.0)

        # At t=duration, should be at target pose
        assert head is not None
        np.testing.assert_array_almost_equal(head[:3, 3], target_pose[:3, 3])
        assert antennas[0] == pytest.approx(0.5, abs=0.01)
        assert antennas[1] == pytest.approx(0.5, abs=0.01)
        assert body_yaw == pytest.approx(1.0, abs=0.01)

    def test_evaluate_at_midpoint(self):
        """Test evaluation at t=duration/2 returns midpoint."""
        start_pose = np.eye(4, dtype=np.float32)
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[0, 3] = 1.0  # Move forward 1m

        move = GotoQueueMove(
            target_head_pose=target_pose,
            start_head_pose=start_pose,
            target_antennas=(1.0, 1.0),
            start_antennas=(0.0, 0.0),
            target_body_yaw=2.0,
            start_body_yaw=0.0,
            duration=2.0,
        )

        head, antennas, body_yaw = move.evaluate(1.0)

        # At t=1.0 (midpoint), should be halfway
        assert head is not None
        assert head[0, 3] == pytest.approx(0.5, abs=0.01)  # Halfway in X
        assert antennas[0] == pytest.approx(0.5, abs=0.01)
        assert antennas[1] == pytest.approx(0.5, abs=0.01)
        assert body_yaw == pytest.approx(1.0, abs=0.01)

    def test_evaluate_beyond_duration_clamps(self):
        """Test evaluation beyond duration clamps to target."""
        start_pose = np.eye(4, dtype=np.float32)
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[0, 3] = 1.0

        move = GotoQueueMove(
            target_head_pose=target_pose,
            start_head_pose=start_pose,
            target_antennas=(0.5, 0.5),
            start_antennas=(0.0, 0.0),
            target_body_yaw=1.0,
            start_body_yaw=0.0,
            duration=1.0,
        )

        head, antennas, body_yaw = move.evaluate(10.0)  # Way beyond duration

        # Should clamp to target
        assert head is not None
        assert head[0, 3] == pytest.approx(1.0, abs=0.01)
        assert antennas[0] == pytest.approx(0.5, abs=0.01)
        assert body_yaw == pytest.approx(1.0, abs=0.01)

    def test_uses_default_start_pose_when_none(self):
        """Test that None start_head_pose uses neutral pose."""
        target_pose = np.eye(4, dtype=np.float32)
        target_pose[0, 3] = 1.0

        move = GotoQueueMove(
            target_head_pose=target_pose,
            start_head_pose=None,  # Should use create_head_pose(0,0,0,0,0,0)
            target_antennas=(0.5, 0.5),
            start_antennas=None,  # Should use (0, 0)
            target_body_yaw=1.0,
            start_body_yaw=None,  # Should use 0
            duration=1.0,
        )

        head, antennas, body_yaw = move.evaluate(0.0)

        # At t=0 with None start, should be at neutral/zero
        assert head is not None
        assert body_yaw == pytest.approx(0.0, abs=0.01)
        assert antennas[0] == pytest.approx(0.0, abs=0.01)


class TestSoundQueueMove:
    """Test SoundQueueMove playback."""

    def test_duration_async_mode(self):
        """Test that async mode has instant duration."""
        move = SoundQueueMove(
            sound_file_path="/tmp/test.wav",
            duration=0.0,
            blocking=False,
            loop=False,
        )
        assert move.duration == pytest.approx(0.01, abs=0.001)  # Instant

    def test_duration_blocking_mode(self):
        """Test that blocking mode uses specified duration."""
        move = SoundQueueMove(
            sound_file_path="/tmp/test.wav",
            duration=3.5,
            blocking=True,
            loop=False,
        )
        assert move.duration == 3.5

    def test_duration_loop_mode(self):
        """Test that loop mode uses specified duration."""
        move = SoundQueueMove(
            sound_file_path="/tmp/test.wav",
            duration=10.0,
            blocking=False,
            loop=True,
        )
        assert move.duration == 10.0

    @patch("reachy_mini_conversation_app.rmscript_adapters.sound_player.play_sound_async")
    def test_plays_sound_once_at_start(self, mock_play_async):
        """Test that sound plays only once at t=0."""
        move = SoundQueueMove(
            sound_file_path="/tmp/test.wav",
            duration=0.0,
            blocking=False,
            loop=False,
        )

        # First evaluation should play
        move.evaluate(0.0)
        assert mock_play_async.call_count == 1

        # Subsequent evaluations should not play again
        move.evaluate(0.5)
        move.evaluate(1.0)
        assert mock_play_async.call_count == 1  # Still 1

    @patch("reachy_mini_conversation_app.rmscript_adapters.sound_player.play_sound_blocking")
    def test_blocking_mode_calls_blocking(self, mock_play_blocking):
        """Test that blocking mode uses play_sound_blocking."""
        move = SoundQueueMove(
            sound_file_path="/tmp/test.wav",
            duration=3.0,
            blocking=True,
            loop=False,
        )

        move.evaluate(0.0)
        mock_play_blocking.assert_called_once()
        assert str(mock_play_blocking.call_args[0][0]).endswith("test.wav")

    @patch("reachy_mini_conversation_app.rmscript_adapters.sound_player.play_sound_loop")
    def test_loop_mode_calls_loop(self, mock_play_loop):
        """Test that loop mode uses play_sound_loop."""
        move = SoundQueueMove(
            sound_file_path="/tmp/test.wav",
            duration=10.0,
            blocking=False,
            loop=True,
        )

        move.evaluate(0.0)
        mock_play_loop.assert_called_once()

    def test_evaluate_returns_none_pose(self):
        """Test that evaluate returns None for all joints (maintains pose)."""
        move = SoundQueueMove(
            sound_file_path="/tmp/test.wav",
            duration=0.0,
            blocking=False,
            loop=False,
        )

        with patch("reachy_mini_conversation_app.rmscript_adapters.sound_player.play_sound_async"):
            head, antennas, body_yaw = move.evaluate(0.0)
            assert head is None
            assert antennas is None
            assert body_yaw is None


class TestPictureQueueMove:
    """Test PictureQueueMove capture."""

    def test_duration_is_instant(self):
        """Test that picture move has instant duration."""
        move = PictureQueueMove(camera_worker=None, duration=0.01)
        assert move.duration == 0.01

    def test_captures_picture_once_at_start(self):
        """Test that picture is captured only once at t=0."""
        mock_camera = Mock()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.get_latest_frame.return_value = mock_frame

        move = PictureQueueMove(camera_worker=mock_camera, duration=0.01)

        # First evaluation should capture
        move.evaluate(0.0)
        assert mock_camera.get_latest_frame.call_count == 1
        assert move._captured is True

        # Subsequent evaluations should not capture again
        move.evaluate(0.5)
        move.evaluate(1.0)
        assert mock_camera.get_latest_frame.call_count == 1  # Still 1

    def test_no_camera_worker_logs_warning(self):
        """Test that no camera worker produces warning."""
        move = PictureQueueMove(camera_worker=None, duration=0.01)

        # Should not crash, just log warning
        head, antennas, body_yaw = move.evaluate(0.0)

        assert move.picture_base64 is None
        assert move.saved_path is None
        assert head is None
        assert antennas is None
        assert body_yaw is None

    def test_no_frame_available_logs_warning(self):
        """Test that no frame from camera produces warning."""
        mock_camera = Mock()
        mock_camera.get_latest_frame.return_value = None  # No frame

        move = PictureQueueMove(camera_worker=mock_camera, duration=0.01)
        move.evaluate(0.0)

        assert move.picture_base64 is None
        assert move.saved_path is None

    @patch("cv2.imwrite")
    def test_saves_picture_to_tmp(self, mock_imwrite):
        """Test that picture is saved to /tmp with timestamp."""
        mock_camera = Mock()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.get_latest_frame.return_value = mock_frame
        mock_imwrite.return_value = True

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.return_value = b"fake_image"
            mock_open.return_value = mock_file

            move = PictureQueueMove(camera_worker=mock_camera, duration=0.01)
            move.evaluate(0.0)

            # Check that cv2.imwrite was called
            mock_imwrite.assert_called_once()
            saved_path = mock_imwrite.call_args[0][0]
            assert saved_path.startswith("/tmp/reachy_picture_")
            assert saved_path.endswith(".jpg")
            assert move.saved_path == saved_path

    @patch("cv2.imwrite")
    @patch("builtins.open", create=True)
    def test_creates_base64_encoding(self, mock_open, mock_imwrite):
        """Test that picture is encoded to base64."""
        mock_camera = Mock()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.get_latest_frame.return_value = mock_frame
        mock_imwrite.return_value = True

        # Mock file read for base64 encoding
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = b"fake_image_data"
        mock_open.return_value = mock_file

        move = PictureQueueMove(camera_worker=mock_camera, duration=0.01)
        move.evaluate(0.0)

        # Should have base64 encoded data
        assert move.picture_base64 is not None
        assert isinstance(move.picture_base64, str)
        # Base64 of b"fake_image_data" should be "ZmFrZV9pbWFnZV9kYXRh"
        import base64
        expected = base64.b64encode(b"fake_image_data").decode("utf-8")
        assert move.picture_base64 == expected

    def test_evaluate_returns_none_pose(self):
        """Test that evaluate returns None for all joints (maintains pose)."""
        mock_camera = Mock()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera.get_latest_frame.return_value = mock_frame

        move = PictureQueueMove(camera_worker=mock_camera, duration=0.01)

        with patch("cv2.imwrite"), patch("builtins.open", create=True):
            head, antennas, body_yaw = move.evaluate(0.0)
            assert head is None
            assert antennas is None
            assert body_yaw is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
