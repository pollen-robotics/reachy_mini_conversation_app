"""Tests for utility helpers."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.utils import (
    CameraVisionInitializationError,
    initialize_camera_and_vision,
)


def test_initialize_camera_and_vision_propagates_local_vision_init_failures() -> None:
    """Explicit local vision requests should preserve unexpected initialization errors."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker=None,
        local_vision=True,
    )

    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker, \
         patch("reachy_mini_conversation_app.utils.subprocess.run", return_value=MagicMock(returncode=0)), \
         patch(
             "reachy_mini_conversation_app.vision.local_vision.initialize_vision_processor",
             side_effect=RuntimeError("Vision processor initialization failed"),
         ):
        with pytest.raises(RuntimeError, match="Vision processor initialization failed"):
            initialize_camera_and_vision(args, MagicMock())

    mock_camera_worker.assert_called_once()


def test_initialize_camera_and_vision_raises_when_local_vision_import_crashes() -> None:
    """Explicit local vision requests should fail cleanly on native import crashes."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker=None,
        local_vision=True,
    )

    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker, \
         patch("reachy_mini_conversation_app.utils.subprocess.run", return_value=MagicMock(returncode=-4)):
        with pytest.raises(CameraVisionInitializationError, match="Local vision import crashed"):
            initialize_camera_and_vision(args, MagicMock())

    mock_camera_worker.assert_called_once()


def test_initialize_camera_and_vision_raises_when_head_tracker_init_fails() -> None:
    """Head-tracker startup failures should be reported through the clean init error path."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker="yolo",
        local_vision=False,
    )

    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker, patch(
        "reachy_mini_conversation_app.vision.head_tracking.yolo_process.YoloHeadTrackerProcess",
        side_effect=RuntimeError("tracker init failed"),
    ):
        with pytest.raises(
            CameraVisionInitializationError,
            match="Failed to initialize yolo head tracker: tracker init failed",
        ):
            initialize_camera_and_vision(args, MagicMock())

    mock_camera_worker.assert_not_called()


def test_initialize_camera_and_vision_uses_mediapipe_head_tracker_in_process() -> None:
    """MediaPipe head tracking should use the in-process toolbox tracker."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker="mediapipe",
        local_vision=False,
    )

    current_robot = MagicMock()
    mediapipe_head_tracker = MagicMock()
    with patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker, patch(
        "reachy_mini_conversation_app.vision.head_tracking.mediapipe.MediapipeHeadTracker",
        return_value=mediapipe_head_tracker,
    ):
        initialize_camera_and_vision(args, current_robot)

    mock_camera_worker.assert_called_once_with(current_robot, mediapipe_head_tracker)
