"""Tests for utility helpers."""

import argparse
from unittest.mock import MagicMock, patch

from reachy_mini_conversation_app.utils import initialize_camera


def test_initialize_camera_creates_worker_when_camera_enabled() -> None:
    """A camera worker is created when --no-camera is not set."""
    args = argparse.Namespace(no_camera=False)
    current_robot = MagicMock()

    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker:
        camera_worker = initialize_camera(args, current_robot)

    mock_camera_worker.assert_called_once_with(current_robot)
    assert camera_worker is mock_camera_worker.return_value


def test_initialize_camera_returns_none_when_disabled() -> None:
    """No camera worker is created when --no-camera is set."""
    args = argparse.Namespace(no_camera=True)

    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker:
        camera_worker = initialize_camera(args, MagicMock())

    mock_camera_worker.assert_not_called()
    assert camera_worker is None
